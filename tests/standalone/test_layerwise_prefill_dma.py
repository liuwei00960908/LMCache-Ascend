# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the layerwise prefill DMA planner."""

import importlib.util
from pathlib import Path
import sys

import torch


def _load_dma_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend"
        / "v1"
        / "npu_connector"
        / "layerwise_dma.py"
    )
    spec = importlib.util.spec_from_file_location("layerwise_dma_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_block_id_plan_and_incremental_binding():
    dma = _load_dma_module()
    starts = [0, 4, 8, 12]
    ends = [4, 8, 12, 16]
    cycle = dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)
    plan = cycle.plan_block_id_ranges([10, 11, 12, 13], 4, starts, ends)

    assert int(plan.tokens.sum()) == 16
    assert plan.slot.tolist() == [40, 44, 48, 52]

    rows = dma.bind_copy_addresses(
        plan,
        [100, 200, 300, 400],
        [1000],
        [4, 4, 4, 4],
        [2],
        2,
        device_to_host=False,
        host_chunk_tokens=[4, 4, 4, 4],
    )
    assert len(rows) == len(plan)

    objects = [object() for _ in starts]
    bound = dma.bind_incremental_copy_addresses(
        plan,
        objects,
        starts,
        ends,
        [1000],
        [2],
        2,
        lambda _obj: 100,
        lambda _obj: 4,
        None,
        slot_prefix_unchanged=False,
    )
    assert len(bound.rows) == len(rows)


def test_empty_block_range_is_allocation_free():
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)
    plan = cycle.plan_block_id_ranges([], 4, [], [])
    assert plan.chunk.numel() == 0
    assert plan.tokens.numel() == 0
    assert torch.equal(plan.slot, torch.empty(0, dtype=torch.long))
