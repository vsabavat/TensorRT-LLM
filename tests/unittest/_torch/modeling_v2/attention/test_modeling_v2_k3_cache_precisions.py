# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3 fused FP8 projections around BF16, FP8 and NVFP4 paged MLA caches."""

import copy
import weakref

import pytest
import torch
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from _torch.modeling.test_kimi_k3_mla import _checkpoint_weights

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.implementation import (
    DecodeStep,
    K3DecodeMLA,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures, KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="requires an SM100-family GPU",
)


@pytest.mark.parametrize(
    "cache_dtype", ["bf16", "fp8", "nvfp4", "nvfp4_ssm_fp32", "nvfp4_ssm_bf16"]
)
@pytest.mark.parametrize("batch,tokens", [(1, 1), (2, 1), (4, 1), (1, 4), (2, 4), (4, 4)])
@torch.no_grad()
def test_fp8_mla_cache_precisions(monkeypatch, cache_dtype: str, batch: int, tokens: int) -> None:
    import tensorrt_llm._torch.modules.kimi_k3_mla as mla_package

    hybrid = cache_dtype.startswith("nvfp4_ssm_")
    state_dtype = torch.bfloat16 if cache_dtype.endswith("bf16") else torch.float32
    if hybrid:
        cache_dtype = "nvfp4"
    torch.manual_seed(451)
    monkeypatch.setattr(mla_package, "KimiK3MLAAttention", K3DecodeMLA)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 6},
        num_attention_heads=6,
        num_key_value_heads=6,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8=True)
    quant_algo = {"bf16": None, "fp8": QuantAlgo.FP8, "nvfp4": QuantAlgo.NVFP4}[cache_dtype]
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=quant_algo),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                group_size=128,
                kv_cache_quant_algo=quant_algo,
            )
            for name in weights
        },
        use_cute_dsl_blockscaling_mm=torch.cuda.get_device_capability() == (10, 7),
        use_cute_dsl_blockscaling_bmm=True,
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn.mixer
    runtime.prepare_fp8_decode_weights()
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    context_length = 128
    page_size = 128 if cache_dtype == "nvfp4" else 64
    manager_class = Fp4MlaKVCacheManagerV2 if cache_dtype == "nvfp4" else KVCacheManagerV2
    manager_kwargs = {}
    if hybrid:
        manager_class = MambaHybridCacheManagerV2
        manager_kwargs = dict(
            mamba_d_state=128,
            mamba_d_conv=4,
            mamba_num_heads=6,
            mamba_n_groups=6,
            mamba_head_dim=128,
            mamba_num_layers=1,
            mamba_layer_mask=[False, True],
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=state_dtype,
            layer_mask=[True, False],
            conv_state_layout="q_k_v",
        )
    manager = manager_class(
        kv_cache_config=KvCacheConfig(
            max_tokens=256 * batch,
            dtype="auto" if cache_dtype == "bf16" else cache_dtype,
            enable_block_reuse=False,
            host_cache_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=page_size,
        max_seq_len=256,
        max_batch_size=batch,
        mapping=mapping,
        dtype={"bf16": DataType.BF16, "fp8": DataType.FP8, "nvfp4": DataType.NVFP4}[cache_dtype],
        max_num_tokens=batch * context_length,
        pretrained_config=config,
        spec_config=MTPDecodingConfig(max_draft_len=tokens - 1) if tokens > 1 else None,
        **manager_kwargs,
    )
    try:
        requests = manager.add_dummy_requests(list(range(batch)), [context_length + tokens] * batch)
        assert requests is not None
        if hybrid:
            assert manager.get_ssm_states(1).dtype == state_dtype
            assert manager.get_buffers(1) is None
            assert manager.get_fp4_mla_page_table_spec(0).hp_is_paged
        for request in requests:
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.prompt_len = context_length
            request.py_prompt_len = context_length

        def metadata(context: bool):
            meta = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.full(
                    (batch,), context_length if context else tokens, dtype=torch.int
                ),
                request_ids=list(range(batch)),
                max_num_requests=batch,
                num_contexts=batch if context else 0,
                prompt_lens=[context_length] * batch,
                max_num_tokens=batch * context_length,
                kv_cache_manager=manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[0 if context else context_length] * batch,
                ),
                mapping=mapping,
                enable_context_mla_with_cached_kv=True,
                max_total_draft_tokens=tokens - 1,
                runtime_features=AttentionRuntimeFeatures(has_speculative_draft_tokens=tokens > 1),
            )
            meta.prepare()
            return meta

        hidden = (
            torch.randn(batch, context_length + tokens, 256, device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        ctx = metadata(True)
        attrs = {
            "attention_metadata": weakref.ref(ctx),
            "mla_layers": {runtime.layer_idx_str: weakref.ref(runtime)},
        }
        with model_extra_attrs(attrs):
            runtime.forward_generic(
                None, hidden[:, :context_length].flatten(0, 1).contiguous(), ctx
            )
        gen = metadata(False)
        attrs["attention_metadata"] = weakref.ref(gen)
        x = hidden[:, context_length:].flatten(0, 1).contiguous()
        step = DecodeStep(batch * tokens, batch, tokens)
        assert runtime.will_run_decode_branch(gen, step)
        with model_extra_attrs(attrs):
            expected = runtime.forward_generic(None, x, gen)
            actual = runtime(None, x, gen, step)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = runtime(None, x, gen, step)
            graph.replay()
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.04)
        relative_error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-8)
        assert relative_error < 0.04, relative_error
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
        assert runtime.k_b_proj_trans.dtype == torch.float8_e4m3fn
        assert runtime.v_b_proj.dtype == torch.float8_e4m3fn
        assert runtime.k_b_proj_trans_dequant is None and runtime.v_b_proj_dequant is None
    finally:
        manager.shutdown()


@pytest.mark.parametrize("batch,tokens", [(1, 1), (2, 1), (4, 1), (1, 4), (2, 4), (4, 4)])
@torch.inference_mode()
def test_fp4_core_matches_dequantized_reference(monkeypatch, batch: int, tokens: int) -> None:
    from _torch.attention.test_fp4_mla import (
        _build_fp4_mla_attention_decode_case,
        _fp4_mla_attention_decode_reference,
    )

    from tensorrt_llm._torch.attention.backends.fp4_mla import run_fp4_mla_attention_decode

    monkeypatch.setenv("TRTLLM_FP4_MLA_ATTENTION_BACKEND", "triton")
    manager, meta, query, kv_rank, rope_dim = _build_fp4_mla_attention_decode_case(
        seq_lens=[131 + 2 * i for i in range(batch)],
        num_heads=6,
        seed=452,
        query_len_per_seq=tokens,
    )
    try:
        output = torch.empty_like(query[..., :kv_rank])
        state = meta.fp4_mla_state
        run_fp4_mla_attention_decode(
            meta,
            layer_idx=0,
            local_layer=0,
            q=query,
            output=output,
            sm_scale=0.1,
            kv_lora_rank=kv_rank,
            qk_rope_head_dim=rope_dim,
            prequantized_q=state.prequantized_q,
            prequantized_q_sf=state.prequantized_q_sf,
            q_batch_capacity=state.q_batch_capacity,
        )
        reference_meta = copy.copy(meta)
        reference_meta.fp4_mla_state = copy.copy(state)
        reference_meta.fp4_mla_state.workspaces = dict(state.workspaces)
        reference_meta.fp4_mla_state.workspaces.pop("_fp4_mla_attention_p_buf", None)
        expected, _ = _fp4_mla_attention_decode_reference(
            reference_meta,
            query[..., :kv_rank],
            query[..., kv_rank:],
            sm_scale=0.1,
            kv_lora_rank=kv_rank,
            qk_rope_head_dim=rope_dim,
        )
        actual = output.float().flatten(1)
        expected = expected.flatten(1)
        relative = (actual - expected).norm(dim=1) / expected.norm(dim=1)
        cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=1)
        assert bool((relative < 0.15).all()), relative
        assert bool((cosine > 0.99).all()), cosine
        torch.testing.assert_close(actual, expected, atol=0.15, rtol=0.15)
    finally:
        manager.shutdown()
