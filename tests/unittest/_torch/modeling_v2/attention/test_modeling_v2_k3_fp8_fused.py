# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and paged-cache parity for NVFP4's FP8 fused attention kernels."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _ops():
    """Load the standalone kernels without requiring an initialized inference engine."""
    name = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.attention"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[5] / (
            "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/"
            "kimi_k3_nvfp4/attention.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _rand(*shape, scale=1.0):
    return (torch.randn(shape, device="cuda") * scale).bfloat16()


def _quantize(x, ue8m0=False):
    groups = x.float().unflatten(-1, (-1, 128))
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(4.48e-10) / 448
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    codes = (groups / scale).to(torch.float8_e4m3fn)
    return (codes.float() * scale).flatten(-2)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_fp8_kda_fused_decode(batch, state_dtype):
    torch.manual_seed(413)
    h, d, slots_count = 6, 768, batch + 3
    slots = torch.randperm(slots_count, device="cuda")[:batch].int()
    qkvg = _rand(batch, 4 * d, scale=0.2)
    bfa = _rand(batch, 136, scale=0.2)
    fb = _rand(d, 128, scale=0.05)
    weights = [_rand(d, 4, scale=0.3).float() for _ in range(3)]
    a_log = torch.zeros(h, device="cuda")
    dt = torch.zeros(d, device="cuda")
    norm = torch.ones(128, device="cuda")
    conv = _rand(slots_count, 3 * d, 3, scale=0.2)
    # The live pool has dense rows and padding between slots.
    state = torch.randn(slots_count, h * 16384 + 16, device="cuda", dtype=state_dtype)[
        :, : h * 16384
    ].view(slots_count, h, 128, 128)
    expected_conv, expected_state = conv.clone(), state.clone()
    initial_conv, initial_state = conv.clone(), state.clone()
    output = []
    for row, slot in enumerate(slots.tolist()):
        parts = []
        for section in range(3):
            raw = qkvg[row, section * d : (section + 1) * d].float()
            old = expected_conv[slot, section * d : (section + 1) * d].float()
            value = (old * weights[section][:, :3]).sum(-1) + raw * weights[section][:, 3]
            parts.append(F.silu(value).view(h, 128))
            expected_conv[slot, section * d : (section + 1) * d] = torch.cat(
                (old[:, 1:], raw[:, None]), -1
            )
        q, k, v = parts
        raw_q = parts[0]
        q = raw_q * torch.rsqrt(raw_q.square().sum(-1, keepdim=True) + 1.0e-6) / 128**0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)
        gate = F.linear(bfa[row, :128].float(), fb.float()).bfloat16().float().view(h, 128)
        decay = torch.exp(-5 * torch.sigmoid(gate))
        s = state[slot].float() * decay[:, None, :]
        beta = bfa[row, 128 : 128 + h].float().sigmoid()
        innovation = (v - (s * k[:, None, :]).sum(-1)) * beta[:, None]
        s += innovation[:, :, None] * k[:, None, :]
        expected_state[slot] = s
        o = (s * q[:, None, :]).sum(-1)
        o *= torch.rsqrt(o.square().mean(-1, keepdim=True) + 1.0e-6)
        o *= qkvg[row, 3 * d :].float().view(h, 128).sigmoid()
        output.append(o.bfloat16())
    actual = _ops().kda_decode(
        qkvg, bfa, fb, *weights, a_log, dt, norm, conv, state, slots, -5.0, 128**-0.5, 1.0e-6
    )
    torch.testing.assert_close(actual, torch.stack(output), atol=0.008, rtol=0.01)
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(
        state.float(),
        expected_state.float(),
        atol=0.016 if state_dtype == torch.bfloat16 else 2.0e-5,
        rtol=0.01 if state_dtype == torch.bfloat16 else 2.0e-5,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _ops().kda_decode(
            qkvg,
            bfa,
            fb,
            *weights,
            a_log,
            dt,
            norm,
            conv,
            state,
            slots,
            -5.0,
            128**-0.5,
            1.0e-6,
        )
    conv.copy_(initial_conv)
    state.copy_(initial_state)
    graph.replay()
    torch.testing.assert_close(graph_output, actual, atol=0, rtol=0)
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)


@pytest.mark.parametrize("batch,tokens", [(1, 1), (2, 1), (4, 1), (1, 4), (4, 8)])
@pytest.mark.parametrize("qb_ue8m0", [False, True])
def test_fp8_mla_fused_query_cache_and_attention(batch, tokens, qb_ue8m0):
    torch.manual_seed(422)
    torch.backends.cuda.matmul.allow_tf32 = False
    h, m = 6, batch * tokens
    ag = _rand(m, 2880, scale=0.2)
    qa_norm, kv_norm = _rand(1536, scale=0.2), _rand(512, scale=0.2)
    qb = _rand(h * 192, 1536, scale=0.05).to(torch.float8_e4m3fn)
    qb_scale = torch.rand(9, 12, device="cuda") * 0.2 + 0.1
    if qb_ue8m0:
        qb_scale = torch.exp2(torch.ceil(torch.log2(qb_scale)))
    kb = _rand(h, 512, 128, scale=0.05).to(torch.float8_e4m3fn)
    kb_scale = torch.rand(h, 4, 1, device="cuda") * 0.2 + 0.1
    vb = _rand(h, 128, 512, scale=0.05).to(torch.float8_e4m3fn)
    vb_scale = torch.rand(h, 1, 4, device="cuda") * 0.2 + 0.1
    lengths = torch.arange(batch, device="cuda", dtype=torch.int32) * 67 + tokens + 5
    pages_per_request = (int(lengths.max()) + 63) // 64
    pages = (
        torch.randperm(batch * pages_per_request, device="cuda")
        .int()
        .view(batch, pages_per_request)
    )
    page_storage = torch.zeros(batch, 2 * pages_per_request, device="cuda", dtype=torch.int32)
    page_storage[:, :pages_per_request] = pages
    pages = page_storage[:, :pages_per_request]
    page_offset = 2
    pool = _rand(batch * pages_per_request + page_offset, 64, 576, scale=0.1)
    expected_pool = pool.clone()
    qa = ag[:, :1536].float()
    qa = (
        qa * torch.rsqrt(qa.square().mean(-1, keepdim=True) + 1.0e-6) * qa_norm.float()
    ).bfloat16()
    qb_dequant = qb.float() * qb_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    q = (_quantize(qa, qb_ue8m0) @ qb_dequant.T).bfloat16().view(m, h, 192)
    kb_dequant = kb.float() * kb_scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    absorbed = torch.einsum("mhk,hvk->mhv", _quantize(q[..., :128]), kb_dequant).bfloat16()
    expected_q = torch.cat((absorbed, q[..., 128:]), -1).flatten(1)
    for row in range(m):
        seq, offset = divmod(row, tokens)
        pos = int(lengths[seq]) - tokens + offset
        page = int(pages[seq, pos // 64]) + page_offset
        ckv = ag[row, 1536:2048].float()
        ckv *= torch.rsqrt(ckv.square().mean() + 1.0e-6) * kv_norm.float()
        expected_pool[page, pos % 64] = torch.cat((ckv, ag[row, 2048:2112].float())).bfloat16()
    actual_q = _ops().mla_qkv(
        ag,
        qa_norm,
        qb,
        qb_scale,
        kb,
        kb_scale,
        kv_norm,
        pool.flatten(),
        pages,
        lengths,
        576,
        page_offset,
        tokens,
        1.0e-6,
        1.0e-6,
        qb_ue8m0,
    )
    torch.testing.assert_close(actual_q, expected_q, atol=0.002, rtol=0.02)
    torch.testing.assert_close(pool, expected_pool, atol=0.002, rtol=0.02)
    projected_q, latent_cache = _ops().mla_project_qkv(
        ag, qa_norm, qb, qb_scale, kb, kb_scale, kv_norm, 1.0e-6, 1.0e-6, qb_ue8m0
    )
    torch.testing.assert_close(projected_q, actual_q, atol=0, rtol=0)
    latent = ag[:, 1536:2048].float()
    latent = (
        latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + 1.0e-6) * kv_norm.float()
    ).bfloat16()
    torch.testing.assert_close(latent_cache[:, :512], latent, atol=0.002, rtol=0.02)
    torch.testing.assert_close(latent_cache[:, 512:], ag[:, 2048:2112], atol=0, rtol=0)
    vb_dequant = vb.float() * vb_scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    expected_output = []
    for row in range(m):
        seq, offset = divmod(row, tokens)
        end = int(lengths[seq]) - tokens + offset + 1
        positions = torch.arange(end, device="cuda")
        kv = pool[pages[seq, positions // 64].long() + page_offset, positions % 64].float()
        logits = actual_q[row].view(h, 576).float() @ kv.T / 192**0.5
        latent = (logits.softmax(-1) @ kv[:, :512]).bfloat16()
        projected = torch.einsum("hk,hvk->hv", _quantize(latent), vb_dequant).bfloat16()
        gate = ag[row, 2112:].float().sigmoid()
        expected_output.append((projected.flatten().float() * gate).bfloat16())
    actual = _ops().mla_attn(
        actual_q,
        pool.flatten(),
        pages,
        lengths,
        vb,
        vb_scale,
        ag,
        576,
        page_offset,
        tokens,
        192**-0.5,
        2112,
    )
    torch.testing.assert_close(actual, torch.stack(expected_output), atol=0.002, rtol=0.02)
    latent_output = _rand(m, h * 512, scale=0.1)
    projected = torch.einsum(
        "mhk,hvk->mhv", _quantize(latent_output.view(m, h, 512)), vb_dequant
    ).bfloat16()
    expected = (projected.flatten(1).float() * ag[:, 2112:].float().sigmoid()).bfloat16()
    torch.testing.assert_close(
        _ops().mla_output(latent_output, vb, vb_scale, ag, 2112), expected, atol=0.002, rtol=0.02
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_projected_q, graph_latent = _ops().mla_project_qkv(
            ag, qa_norm, qb, qb_scale, kb, kb_scale, kv_norm, 1.0e-6, 1.0e-6, qb_ue8m0
        )
        graph_projected_output = _ops().mla_output(latent_output, vb, vb_scale, ag, 2112)
        graph_q = _ops().mla_qkv(
            ag,
            qa_norm,
            qb,
            qb_scale,
            kb,
            kb_scale,
            kv_norm,
            pool.flatten(),
            pages,
            lengths,
            576,
            page_offset,
            tokens,
            1.0e-6,
            1.0e-6,
            qb_ue8m0,
        )
        graph_output = _ops().mla_attn(
            graph_q,
            pool.flatten(),
            pages,
            lengths,
            vb,
            vb_scale,
            ag,
            576,
            page_offset,
            tokens,
            192**-0.5,
            2112,
        )
    graph.replay()
    torch.testing.assert_close(graph_q, actual_q, atol=0, rtol=0)
    torch.testing.assert_close(graph_output, actual, atol=0, rtol=0)
    torch.testing.assert_close(graph_projected_q, actual_q, atol=0, rtol=0)
    torch.testing.assert_close(graph_latent, latent_cache, atol=0, rtol=0)
    torch.testing.assert_close(graph_projected_output, expected, atol=0.002, rtol=0.02)
    lengths.add_(1)
    graph.replay()
    updated_q = _ops().mla_qkv(
        ag,
        qa_norm,
        qb,
        qb_scale,
        kb,
        kb_scale,
        kv_norm,
        pool.flatten(),
        pages,
        lengths,
        576,
        page_offset,
        tokens,
        1.0e-6,
        1.0e-6,
        qb_ue8m0,
    )
    updated_output = _ops().mla_attn(
        updated_q,
        pool.flatten(),
        pages,
        lengths,
        vb,
        vb_scale,
        ag,
        576,
        page_offset,
        tokens,
        192**-0.5,
        2112,
    )
    torch.testing.assert_close(graph_q, updated_q, atol=0, rtol=0)
    torch.testing.assert_close(graph_output, updated_output, atol=0, rtol=0)
