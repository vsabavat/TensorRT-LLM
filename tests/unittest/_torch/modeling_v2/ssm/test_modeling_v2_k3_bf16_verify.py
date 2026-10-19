# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 state storage with FP32 recurrence, draft replay and CUDA graphs."""

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.verify import (
    k3_fp8_kda_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="requires an SM100-family GPU",
)


def _run(weights: dict, pools: dict, projection: torch.Tensor, slots: torch.Tensor, drafts: int):
    return k3_fp8_kda_verify(
        projection,
        weights["w_fb"],
        weights["w_q"],
        weights["w_k"],
        weights["w_v"],
        weights["a_log"],
        weights["dt_bias"],
        weights["onorm_w"],
        pools["cs_q"],
        pools["cs_k"],
        pools["cs_v"],
        pools["ssm"],
        pools["state_tok"],
        slots,
        pools["pending"],
        drafts,
        kc.LOWER_BOUND,
        kc.SCALE,
        kc.EPS,
    )


@pytest.mark.parametrize("batch,steps", [(1, 2), (2, 4), (4, 8)])
@torch.inference_mode()
def test_bf16_verify_replay_and_graph(batch: int, steps: int) -> None:
    manager = kc.build_manager(2, num_spec=steps - 1, state_dtype=torch.bfloat16)
    try:
        slots = kc.request_slots(manager, batch, first_id=700)
        pools = kc.layer_pools(manager, 1)
        kc.fill_pools(pools, seed=431)
        reference = kc.clone_pools(pools)
        reference["ssm"] = pools["ssm"].float()
        weights = kc.make_weights(432)
        initial = kc.snapshot(pools)
        generator = torch.Generator(device="cuda").manual_seed(433)
        projection = torch.randn(
            batch * steps, kc.PROJ, generator=generator, device="cuda"
        ).bfloat16()
        # Compile both storage dtypes before capture, then restore every live buffer.
        _run(weights, reference, projection, slots, steps - 1)
        _run(weights, pools, projection, slots, steps - 1)
        for name, value in initial.items():
            pools[name].copy_(value)
            reference[name].copy_(value)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _run(weights, pools, projection, slots, steps - 1)
        for rnd in range(5):
            projection.copy_(torch.randn(projection.shape, generator=generator, device="cuda"))
            graph.replay()
            expected = _run(weights, reference, projection, slots, steps - 1)
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.01)
            torch.testing.assert_close(pools["ssm"], reference["ssm"].bfloat16(), atol=0, rtol=0)
            torch.testing.assert_close(
                pools["state_tok"], reference["state_tok"], atol=1e-5, rtol=1e-5
            )
            # Both paths start the next round from a BF16 committed golden state.
            reference["ssm"].copy_(reference["ssm"].bfloat16())
            accepted = torch.tensor(
                kc.pending_schedule(batch, steps - 1, rnd), device="cuda", dtype=torch.int32
            )
            for p in (pools, reference):
                p["pending"][slots.long()] = accepted
        untouched = kc.other_slots(pools["ssm"].shape[0], slots.tolist())
        assert kc.same_rows(pools, initial, untouched, ("ssm", "state_tok", "cs_q", "cs_k", "cs_v"))
    finally:
        manager.shutdown()
