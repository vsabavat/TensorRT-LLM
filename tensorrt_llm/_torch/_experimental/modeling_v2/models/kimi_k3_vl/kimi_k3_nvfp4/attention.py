# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8-preserving fused decode kernels for the NVFP4 Kimi K3 targets."""

import torch
import triton
import triton.language as tl


@triton.jit
def _kda_decode(
    QKVG,
    BFA,
    FB,
    WQ,
    WK,
    WV,
    A_LOG,
    DT,
    NORM,
    CONV,
    STATE,
    SLOTS,
    OUT,
    H: tl.constexpr,
    D: tl.constexpr,
    PROJ_STRIDE: tl.constexpr,
    BFA_STRIDE: tl.constexpr,
    CONV_SLOT: tl.constexpr,
    CONV_CHANNEL: tl.constexpr,
    CONV_TIME: tl.constexpr,
    STATE_SLOT: tl.constexpr,
    LOWER: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + row).to(tl.int64)
    k = tl.arange(0, 128)
    channel = head * 128 + k
    fa = tl.load(BFA + row * BFA_STRIDE + k)
    fb = tl.load(FB + channel[None, :] * 128 + k[:, None])
    gate_tile = tl.dot(tl.broadcast_to(fa[None, :], (16, 128)), fb)
    gate = tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, gate_tile, 0.0), 0)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    beta = tl.sigmoid(tl.load(BFA + row * BFA_STRIDE + 128 + head).to(tl.float32))
    q = tl.full((128,), 0, tl.float32)
    key = tl.full((128,), 0, tl.float32)
    value = tl.full((128,), 0, tl.float32)
    conv_base = slot * CONV_SLOT + channel * CONV_CHANNEL
    for tap in tl.static_range(3):
        cq = tl.load(CONV + conv_base + tap * CONV_TIME).to(tl.float32)
        ck = tl.load(CONV + conv_base + D * CONV_CHANNEL + tap * CONV_TIME).to(tl.float32)
        cv = tl.load(CONV + conv_base + 2 * D * CONV_CHANNEL + tap * CONV_TIME).to(tl.float32)
        q += cq * tl.load(WQ + channel * 4 + tap)
        key += ck * tl.load(WK + channel * 4 + tap)
        value += cv * tl.load(WV + channel * 4 + tap)
    xq = tl.load(QKVG + row * PROJ_STRIDE + channel)
    xk = tl.load(QKVG + row * PROJ_STRIDE + D + channel)
    xv = tl.load(QKVG + row * PROJ_STRIDE + 2 * D + channel)
    q += xq.to(tl.float32) * tl.load(WQ + channel * 4 + 3)
    key += xk.to(tl.float32) * tl.load(WK + channel * 4 + 3)
    value += xv.to(tl.float32) * tl.load(WV + channel * 4 + 3)
    q *= tl.sigmoid(q)
    key *= tl.sigmoid(key)
    value *= tl.sigmoid(value)
    q *= tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
    key *= tl.rsqrt(tl.sum(key * key, 0) + 1.0e-6)
    decay = tl.exp(
        LOWER * tl.sigmoid(tl.exp(tl.load(A_LOG + head)) * (gate + tl.load(DT + channel)))
    )
    offsets = slot * STATE_SLOT + head * 16384 + k[:, None] * 128 + k[None, :]
    state = tl.load(STATE + offsets).to(tl.float32) * decay[None, :]
    innovation = (value - tl.sum(state * key[None, :], 1)) * beta
    state += innovation[:, None] * key[None, :]
    tl.store(STATE + offsets, state)
    output = tl.sum(state * q[None, :], 1)
    output *= tl.rsqrt(tl.sum(output * output, 0) / 128 + EPS)
    output *= tl.load(NORM + k)
    output *= tl.sigmoid(tl.load(QKVG + row * PROJ_STRIDE + 3 * D + channel).to(tl.float32))
    tl.store(OUT + row * D + channel, output)
    for section in tl.static_range(3):
        base = conv_base + section * D * CONV_CHANNEL
        old1 = tl.load(CONV + base + CONV_TIME)
        old2 = tl.load(CONV + base + 2 * CONV_TIME)
        new = tl.load(QKVG + row * PROJ_STRIDE + section * D + channel)
        tl.store(CONV + base, old1)
        tl.store(CONV + base + CONV_TIME, old2)
        tl.store(CONV + base + 2 * CONV_TIME, new)


@torch.library.custom_op("trtllm::k3_fp8_kda_decode", mutates_args=("conv", "state"))
def kda_decode(
    qkvg: torch.Tensor,
    bfa: torch.Tensor,
    fb: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    a_log: torch.Tensor,
    dt: torch.Tensor,
    norm: torch.Tensor,
    conv: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    lower: float,
    scale: float,
    eps: float,
) -> torch.Tensor:
    """Fuse f_b, convolution, normalized delta recurrence and gated RMSNorm."""
    h = state.shape[1]
    if state.shape[2:] != (128, 128) or conv.shape[1:] != (3 * h * 128, 3):
        raise ValueError("Fused FP8 KDA requires 128-wide heads and width-four convolution")
    out = qkvg.new_empty((qkvg.shape[0], h, 128))
    _kda_decode[(qkvg.shape[0], h)](
        qkvg,
        bfa,
        fb,
        wq,
        wk,
        wv,
        a_log,
        dt,
        norm,
        conv,
        state,
        slots,
        out,
        h,
        h * 128,
        qkvg.stride(0),
        bfa.stride(0),
        *conv.stride(),
        state.stride(0),
        lower,
        scale,
        eps,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return out


@kda_decode.register_fake
def _kda_fake(qkvg, bfa, fb, wq, wk, wv, a_log, dt, norm, conv, state, slots, lower, scale, eps):
    return qkvg.new_empty((qkvg.shape[0], state.shape[1], 128))


@triton.jit
def _mla_qkv(
    AG,
    QA_NORM,
    QB,
    QB_SCALE,
    KB,
    KB_SCALE,
    KV_NORM,
    POOL,
    PAGES,
    LENGTHS,
    OUT,
    M: tl.constexpr,
    H: tl.constexpr,
    AG_STRIDE: tl.constexpr,
    QB_SCALE_STRIDE: tl.constexpr,
    KB_S0: tl.constexpr,
    KB_S1: tl.constexpr,
    KB_S2: tl.constexpr,
    KS_S0: tl.constexpr,
    KS_S1: tl.constexpr,
    KS_S2: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_OFFSET: tl.constexpr,
    T: tl.constexpr,
    QA_EPS: tl.constexpr,
    KV_EPS: tl.constexpr,
    STORE_PAGED: tl.constexpr = True,
    QB_UE8M0: tl.constexpr = False,
):
    head, chunk = tl.program_id(0), tl.program_id(1)
    rows = chunk * 16 + tl.arange(0, 16)
    qa_cols = tl.arange(0, 2048)
    qa = tl.load(
        AG + rows[:, None] * AG_STRIDE + qa_cols[None, :],
        (rows[:, None] < M) & (qa_cols[None, :] < 1536),
        0,
    ).to(tl.float32)
    rms = tl.rsqrt(tl.sum(qa * qa, 1) / 1536 + QA_EPS)
    n = tl.arange(0, 256)
    k = tl.arange(0, 128)
    qb_rows = head * 192 + n
    q = tl.full((16, 256), 0, tl.float32)
    for block in range(12):
        cols = block * 128 + k
        a = tl.load(AG + rows[:, None] * AG_STRIDE + cols[None, :], rows[:, None] < M, 0).to(
            tl.float32
        )
        a *= rms[:, None] * tl.load(QA_NORM + cols)[None, :].to(tl.float32)
        a = a.to(tl.bfloat16).to(tl.float32)
        a_scale = tl.maximum(tl.max(tl.abs(a), 1) / 448.0, 1.0e-12)
        if QB_UE8M0:
            a_scale = tl.exp2(tl.ceil(tl.log2(a_scale)))
        a8 = (a / a_scale[:, None]).to(tl.float8e4nv)
        b = tl.load(QB + qb_rows[None, :] * 1536 + cols[:, None], n[None, :] < 192, 0.0)
        bs = tl.load(QB_SCALE + (qb_rows // 128) * QB_SCALE_STRIDE + block, n < 192, 0)
        q += tl.dot(a8, b) * a_scale[:, None] * bs[None, :]
    q = q.to(tl.bfloat16).to(tl.float32)
    # Only the NoPE portion is absorbed; the PE portion is copied unchanged.
    qn = tl.gather(q, tl.broadcast_to(k[None, :], (16, 128)), 1)
    q_scale = tl.maximum(tl.max(tl.abs(qn), 1) / 448.0, 1.0e-12)
    q8 = (qn / q_scale[:, None]).to(tl.float8e4nv)
    latent = tl.arange(0, 512)
    kb = tl.load(KB + head * KB_S0 + latent[None, :] * KB_S1 + k[:, None] * KB_S2)
    ks = tl.load(KB_SCALE + head * KS_S0 + (latent // 128) * KS_S1)
    absorbed = tl.dot(q8, kb) * q_scale[:, None] * ks[None, :]
    base = rows[:, None] * (H * 576) + head * 576
    tl.store(OUT + base + latent[None, :], absorbed, rows[:, None] < M)
    pe = tl.arange(0, 64)
    qpe = tl.gather(q, tl.broadcast_to((128 + pe)[None, :], (16, 64)), 1)
    tl.store(OUT + base + 512 + pe[None, :], qpe, rows[:, None] < M)
    if head == 0:
        if STORE_PAGED:
            seq = rows // T
            length = tl.load(LENGTHS + seq, rows < M, 0)
            pos = length - T + rows % T
            page = tl.load(PAGES + seq * PAGE_STRIDE + pos // 64, (rows < M) & (pos >= 0), 0)
            cache_row = ((page.to(tl.int64) + PAGE_OFFSET) * 64 + pos % 64) * ROW_STRIDE
            valid = (rows < M) & (pos >= 0)
        else:
            cache_row = rows * 576
            valid = rows < M
        ckv = tl.load(
            AG + rows[:, None] * AG_STRIDE + 1536 + latent[None, :], rows[:, None] < M, 0
        ).to(tl.float32)
        kv_rms = tl.rsqrt(tl.sum(ckv * ckv, 1) / 512 + KV_EPS)
        ckv *= kv_rms[:, None] * tl.load(KV_NORM + latent)[None, :].to(tl.float32)
        tl.store(
            POOL + cache_row[:, None] + latent[None, :],
            ckv,
            valid[:, None],
        )
        kpe = tl.load(AG + rows[:, None] * AG_STRIDE + 2048 + pe[None, :], rows[:, None] < M, 0)
        tl.store(
            POOL + cache_row[:, None] + 512 + pe[None, :],
            kpe,
            valid[:, None],
        )


@torch.library.custom_op("trtllm::k3_fp8_mla_qkv", mutates_args=("pool",))
def mla_qkv(
    ag: torch.Tensor,
    qa_norm: torch.Tensor,
    qb: torch.Tensor,
    qb_scale: torch.Tensor,
    kb: torch.Tensor,
    kb_scale: torch.Tensor,
    kv_norm: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    lengths: torch.Tensor,
    row_stride: int,
    page_offset: int,
    tokens_per_request: int,
    qa_eps: float,
    kv_eps: float,
    qb_ue8m0: bool = False,
) -> torch.Tensor:
    """Fuse Q-A norm, FP8 Q-B/absorption and the paged latent-cache update."""
    h = kb.shape[0]
    out = ag.new_empty((ag.shape[0], h * 576))
    _mla_qkv[(h, triton.cdiv(ag.shape[0], 16))](
        ag,
        qa_norm,
        qb,
        qb_scale,
        kb,
        kb_scale,
        kv_norm,
        pool,
        pages,
        lengths,
        out,
        ag.shape[0],
        h,
        ag.stride(0),
        qb_scale.stride(0),
        *kb.stride(),
        *kb_scale.stride(),
        row_stride,
        pages.stride(0),
        page_offset,
        tokens_per_request,
        qa_eps,
        kv_eps,
        QB_UE8M0=qb_ue8m0,
        num_warps=8,
    )
    return out


@mla_qkv.register_fake
def _qkv_fake(
    ag,
    qa_norm,
    qb,
    qb_scale,
    kb,
    kb_scale,
    kv_norm,
    pool,
    pages,
    lengths,
    row_stride,
    page_offset,
    tokens_per_request,
    qa_eps,
    kv_eps,
    qb_ue8m0=False,
):
    return ag.new_empty((ag.shape[0], kb.shape[0] * 576))


@torch.library.custom_op("trtllm::k3_fp8_mla_project_qkv", mutates_args=())
def mla_project_qkv(
    ag: torch.Tensor,
    qa_norm: torch.Tensor,
    qb: torch.Tensor,
    qb_scale: torch.Tensor,
    kb: torch.Tensor,
    kb_scale: torch.Tensor,
    kv_norm: torch.Tensor,
    qa_eps: float,
    kv_eps: float,
    qb_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce absorbed Q and normalized latent KV for quantized cache kernels."""
    h = kb.shape[0]
    query = ag.new_empty((ag.shape[0], h * 576))
    latent = ag.new_empty((ag.shape[0], 576))
    _mla_qkv[(h, triton.cdiv(ag.shape[0], 16))](
        ag,
        qa_norm,
        qb,
        qb_scale,
        kb,
        kb_scale,
        kv_norm,
        latent,
        ag,
        ag,
        query,
        ag.shape[0],
        h,
        ag.stride(0),
        qb_scale.stride(0),
        *kb.stride(),
        *kb_scale.stride(),
        576,
        0,
        0,
        1,
        qa_eps,
        kv_eps,
        STORE_PAGED=False,
        QB_UE8M0=qb_ue8m0,
        num_warps=8,
    )
    return query, latent


@mla_project_qkv.register_fake
def _project_qkv_fake(
    ag, qa_norm, qb, qb_scale, kb, kb_scale, kv_norm, qa_eps, kv_eps, qb_ue8m0=False
):
    return ag.new_empty((ag.shape[0], kb.shape[0] * 576)), ag.new_empty((ag.shape[0], 576))


@triton.jit
def _mla_output(
    LATENT,
    VB,
    VS,
    GATE,
    OUT,
    H: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    GATE_COL: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 128)
    projected_tile = tl.full((16, 128), 0, tl.float32)
    for group in tl.static_range(4):
        latent = tl.load(LATENT + (row * H + head) * 512 + group * 128 + k).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(latent), 0) / 448.0, 1.0e-12)
        activation = tl.broadcast_to((latent / scale)[None, :], (16, 128)).to(tl.float8e4nv)
        weights = tl.load(VB + head * 128 * 512 + k[None, :] * 512 + group * 128 + k[:, None])
        projected_tile += tl.dot(activation, weights) * scale * tl.load(VS + head * 4 + group)
    projected = tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, projected_tile, 0), 0)
    projected = projected.to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(tl.load(GATE + row * GATE_STRIDE + GATE_COL + head * 128 + k).to(tl.float32))
    tl.store(OUT + (row * H + head) * 128 + k, projected * gate)


@torch.library.custom_op("trtllm::k3_fp8_mla_output", mutates_args=())
def mla_output(
    latent: torch.Tensor,
    vb: torch.Tensor,
    vb_scale: torch.Tensor,
    gate: torch.Tensor,
    gate_col: int,
) -> torch.Tensor:
    """Fuse FP8 V-B absorption and output gating after FP4/FP8 attention."""
    h = vb.shape[0]
    output = latent.new_empty((latent.shape[0], h * 128))
    _mla_output[(latent.shape[0], h)](
        latent, vb, vb_scale, gate, output, h, gate.stride(0), gate_col, num_warps=4
    )
    return output


@mla_output.register_fake
def _output_fake(latent, vb, vb_scale, gate, gate_col):
    return latent.new_empty((latent.shape[0], vb.shape[0] * 128))


@triton.jit
def _mla_attn(
    Q,
    POOL,
    PAGES,
    LENGTHS,
    VB,
    VS,
    GATE,
    OUT,
    H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_OFFSET: tl.constexpr,
    T: tl.constexpr,
    SCALE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    GATE_COL: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    seq = row // T
    length = tl.load(LENGTHS + seq)
    end = length - T + row % T + 1
    d = tl.arange(0, 1024)
    v = tl.arange(0, 512)
    q = tl.load(Q + row * H * 576 + head * 576 + d, d < 576, 0).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    output = tl.full((512,), 0.0, tl.float32)
    for start in range(0, tl.cdiv(tl.maximum(end, 0), 32)):
        pos = start * 32 + tl.arange(0, 32)
        page = tl.load(PAGES + seq * PAGE_STRIDE + pos // 64, pos < end, 0)
        offset = ((page.to(tl.int64) + PAGE_OFFSET) * 64 + pos % 64) * ROW_STRIDE
        kv = tl.load(
            POOL + offset[:, None] + d[None, :], (pos[:, None] < end) & (d[None, :] < 576), 0
        ).to(tl.float32)
        scores = tl.sum(kv * q[None, :], 1) * SCALE
        scores = tl.where(pos < end, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 0))
        alpha = tl.exp(maximum - new_max)
        probability = tl.exp(scores - new_max)
        value = tl.gather(kv, tl.broadcast_to(v[None, :], (32, 512)), 1)
        output = output * alpha + tl.sum(probability[:, None] * value, 0)
        denominator = denominator * alpha + tl.sum(probability, 0)
        maximum = new_max
    output = (output / tl.maximum(denominator, 1.0e-20)).to(tl.bfloat16).to(tl.float32)
    # Quantize each 128-wide latent group before the FP8 V-B projection.
    groups = tl.reshape(output, (4, 128))
    scales = tl.maximum(tl.max(tl.abs(groups), 1) / 448.0, 1.0e-12)
    quantized = (groups / scales[:, None]).to(tl.float8e4nv).to(tl.float32)
    n = tl.arange(0, 128)
    projected_tile = tl.full((16, 128), 0.0, tl.float32)
    for group in tl.static_range(4):
        latent = tl.sum(tl.where(tl.arange(0, 4)[:, None] == group, quantized, 0.0), 0)
        activation_scale = tl.sum(tl.where(tl.arange(0, 4) == group, scales, 0.0), 0)
        activation = tl.broadcast_to(latent[None, :], (16, 128)).to(tl.float8e4nv)
        weights = tl.load(VB + head * 128 * 512 + n[None, :] * 512 + group * 128 + n[:, None])
        weight_scale = tl.load(VS + head * 4 + group)
        projected_tile += tl.dot(activation, weights) * activation_scale * weight_scale
    projected = tl.sum(tl.where(tl.arange(0, 16)[:, None] == 0, projected_tile, 0.0), 0)
    projected = projected.to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(tl.load(GATE + row * GATE_STRIDE + GATE_COL + head * 128 + n).to(tl.float32))
    tl.store(OUT + row * H * 128 + head * 128 + n, projected * gate)


@torch.library.custom_op("trtllm::k3_fp8_mla_attn", mutates_args=())
def mla_attn(
    q: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    lengths: torch.Tensor,
    vb: torch.Tensor,
    vb_scale: torch.Tensor,
    gate: torch.Tensor,
    row_stride: int,
    page_offset: int,
    tokens_per_request: int,
    softmax_scale: float,
    gate_col: int,
) -> torch.Tensor:
    """Fuse paged causal attention, FP8 V-B projection and output gating."""
    h = vb.shape[0]
    out = q.new_empty((q.shape[0], h * 128))
    _mla_attn[(q.shape[0], h)](
        q,
        pool,
        pages,
        lengths,
        vb,
        vb_scale,
        gate,
        out,
        h,
        row_stride,
        pages.stride(0),
        page_offset,
        tokens_per_request,
        softmax_scale,
        gate.stride(0),
        gate_col,
        num_warps=8,
    )
    return out


@mla_attn.register_fake
def _attn_fake(
    q,
    pool,
    pages,
    lengths,
    vb,
    vb_scale,
    gate,
    row_stride,
    page_offset,
    tokens_per_request,
    softmax_scale,
    gate_col,
):
    return q.new_empty((q.shape[0], vb.shape[0] * 128))
