# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 targets share one implementation and leave the original MXFP4 targets independent."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import (
    implementation as modeling,
)
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


def test_nvfp4_k3_selects_hybrid_v2_cache_manager(monkeypatch):
    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.pyexecutor.config_utils import uses_fp4_mla_attention
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_fp4_mla_config
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    config = ModelConfig(
        pretrained_config=KimiLinearConfig(
            num_hidden_layers=2,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [2],
                "head_dim": 128,
                "num_heads": 6,
                "short_conv_kernel_size": 4,
            },
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
    )
    cache = KvCacheConfig(dtype="nvfp4", use_kv_cache_manager_v2=True)
    assert uses_fp4_mla_attention(config)
    validate_fp4_mla_config(
        config, SimpleNamespace(kv_cache_config=cache, enable_chunked_prefill=False)
    )
    assert get_kv_cache_manager_cls(config, cache) is MambaHybridCacheManagerV2
    captured = {}

    class RecordingManager(MambaHybridCacheManagerV2):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingManager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=cache,
        tokens_per_block=128,
        max_seq_len=256,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=config,
        dtype=torch.bfloat16,
    )
    assert captured["pretrained_config"] is config.pretrained_config
    with pytest.raises(ValueError, match="use_kv_cache_manager_v2"):
        get_kv_cache_manager_cls(
            config, cache.model_copy(update={"use_kv_cache_manager_v2": False})
        )
    monkeypatch.setenv("TRTLLM_USE_PY_MAMBA", "1")
    with pytest.raises(ValueError, match="MambaHybridCacheManagerV2"):
        get_kv_cache_manager_cls(config, cache)


@pytest.mark.parametrize("moe_tp,moe_ep", [(4, 4), (16, 1)])
def test_sm100_mxfp4_keeps_its_own_model(moe_tp: int, moe_ep: int) -> None:
    module = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        f"kimi_k3_mxfp4__sm_100__tp16_moetp{moe_tp}ep{moe_ep}.modeling"
    )
    target = getattr(module, f"ModelingV2KimiK3Mxfp4Sm100Tp16Moetp{moe_tp}ep{moe_ep}")
    assert target.forward.__module__ == module.__name__
    assert target.forward is not modeling.ModelingV2KimiK3.forward


def test_fp8_mla_preserves_raw_query_weights_before_backend_transformation() -> None:
    attention = modeling.K3DecodeMLA.__new__(modeling.K3DecodeMLA)
    torch.nn.Module.__init__(attention)
    codes = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scales = torch.full((1, 1), 0.125)
    attention.q_b_proj = SimpleNamespace(
        weight=codes,
        weight_scale=scales,
        out_features=128,
        in_features=128,
        use_cute_dsl_blockscaling_mm=True,
        disable_deep_gemm=False,
    )
    attention.prepare_fp8_decode_weights()
    codes.zero_()
    scales.zero_()
    assert attention._fp8_qb_weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(attention._fp8_qb_weight.float(), torch.ones(128, 128))
    torch.testing.assert_close(attention._fp8_qb_scale, torch.full((1, 1), 0.125))


@pytest.mark.parametrize(
    "backend,ue8m0",
    [
        ({}, True),
        ({"use_cute_dsl_blockscaling_mm": False, "disable_deep_gemm": False}, True),
        ({"use_cute_dsl_blockscaling_mm": True, "disable_deep_gemm": False}, False),
        ({"use_cute_dsl_blockscaling_mm": False, "disable_deep_gemm": True}, False),
    ],
)
def test_fp8_mla_query_recipe_handles_weight_read_specialization(monkeypatch, backend, ue8m0):
    attention = modeling.K3DecodeMLA.__new__(modeling.K3DecodeMLA)
    torch.nn.Module.__init__(attention)
    codes = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scales = torch.full((1, 1), 0.125)
    attention.q_b_proj = SimpleNamespace(
        weight=codes, weight_scale=scales, out_features=128, in_features=128, **backend
    )
    calls = []
    monkeypatch.setattr(modeling, "resmooth_to_fp8_e8m0", lambda w, s: calls.append((w, s)))
    attention.prepare_fp8_decode_weights()
    assert attention._fp8_qb_ue8m0 == ue8m0
    assert len(calls) == int(ue8m0)
    if calls:
        assert calls[0][0].data_ptr() != codes.data_ptr()
        assert calls[0][1].data_ptr() != scales.data_ptr()


@pytest.mark.parametrize("sm", [(10, 3), (10, 7)])
def test_nvfp4_kda_selects_rubin_projection_backend(monkeypatch, sm) -> None:
    from tensorrt_llm._torch.model_config import ModelConfig

    captured = []

    def initialize(module, *args, model_config, **kwargs):
        torch.nn.Module.__init__(module)
        captured.append(model_config)

    monkeypatch.setattr(modeling.KimiKDALinearAttention, "__init__", initialize)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: sm)
    config = ModelConfig(use_cute_dsl_blockscaling_mm=False)
    modeling.K3DecodeKDA(model_config=config)
    assert captured[0].use_cute_dsl_blockscaling_mm == (sm == (10, 7))
    assert not config.use_cute_dsl_blockscaling_mm


@pytest.mark.parametrize(
    "sm,fmt,moe_tp,moe_ep",
    [
        (103, "nvfp4", 4, 4),
        (107, "nvfp4", 4, 4),
    ],
)
def test_targets_share_the_model_lifecycle(sm: int, fmt: str, moe_tp: int, moe_ep: int) -> None:
    module = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        f"kimi_k3_{fmt}__sm_{sm}__tp16_moetp{moe_tp}ep{moe_ep}.modeling"
    )
    target = getattr(module, f"ModelingV2KimiK3{fmt.capitalize()}Sm{sm}Tp16Moetp{moe_tp}ep{moe_ep}")
    assert target.forward is modeling.ModelingV2KimiK3.forward
    assert target.load_weights is modeling.ModelingV2KimiK3.load_weights
    assert target.cache_derived_state is modeling.ModelingV2KimiK3.cache_derived_state
    assert target.post_load_weights is modeling.ModelingV2KimiK3.post_load_weights
    assert target.target_spec.sm == (sm // 10, sm % 10)
    assert (target.target_spec.moe_tp, target.target_spec.moe_ep) == (moe_tp, moe_ep)


@pytest.mark.parametrize("algo", [QuantAlgo.W4A8_MXFP4_MXFP8, QuantAlgo.NVFP4])
def test_streaming_loader_preserves_checkpoint_tensors(algo: QuantAlgo) -> None:
    """Each loader passes the exact packed tensors and scales to its quant method."""
    calls = []

    def record(backend: object, **kwargs: object) -> None:
        calls.append(kwargs)

    backend = SimpleNamespace(
        quant_method=SimpleNamespace(
            load_packed_mxfp4_expert=record,
            load_streaming_nvfp4_expert=record,
        )
    )
    spec = modeling._k3_expert_ckpt_spec(algo)
    tensors = {
        f"experts.7.{weight}.{kind}": object()
        for weight in ("w1", "w2", "w3")
        for kind in spec.kinds
    }
    spec.loader(backend, "experts", 7, 2, tensors.__getitem__)
    assert len(calls) == 1
    loaded = calls[0]
    assert loaded.pop("global_expert_id") == 7
    assert loaded.pop("local_slot_id") == 2
    for weight in ("w1", "w2", "w3"):
        for kind in spec.kinds:
            argument = "weight" if kind == "weight_packed" else kind
            assert loaded.pop(f"{weight}_{argument}") is tensors[f"experts.7.{weight}.{kind}"]
    assert not loaded
    assert spec.needs_layer_finalize == (algo == QuantAlgo.NVFP4)


@pytest.mark.parametrize("sm", [(10, 0), (10, 3), (10, 7)])
def test_mxfp4_fused_kernel_gates_hardware_and_scale_layout(monkeypatch, sm) -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import is_supported

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: sm)
    weights = torch.empty(224, 1536, 1792, dtype=torch.uint8, device="meta")
    fc1_scales = torch.empty(224, 1536, 112, dtype=torch.uint8, device="meta")
    down = torch.empty(224, 3584, 384, dtype=torch.uint8, device="meta")
    fc2_scales = torch.empty(224, 3584, 24, dtype=torch.uint8, device="meta")
    supported, _ = is_supported(weights, fc1_scales, down, fc2_scales, 224)
    assert supported == (sm in ((10, 0), (10, 3)))
    # Group-16 scales cannot be consumed by this group-32 MXFP4 kernel.
    nvfp4_scales = torch.empty(224, 1536, 224, dtype=torch.uint8, device="meta")
    supported, _ = is_supported(weights, nvfp4_scales, down, fc2_scales, 224)
    assert not supported
