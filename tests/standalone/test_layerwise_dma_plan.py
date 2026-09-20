"""No-NPU checks for the P-node bundle/chunk DMA address plan."""

import importlib.util
import sys
import torch
import pytest
import ast
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "lmcache_ascend/v1/npu_connector/layerwise_dma.py"
)
spec = importlib.util.spec_from_file_location("layerwise_dma_standalone", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
bind_copy_addresses = module.bind_copy_addresses
plan_bundle_copies = module.plan_bundle_copies


def test_latent_bundle_is_split_only_at_512_token_bundle_boundaries():
    plan = plan_bundle_copies(list(range(2048)), [1024, 1024], 512)
    assert list(
        zip(plan.chunk.tolist(), plan.slot.tolist(), plan.tokens.tolist(), strict=False)
    ) == [
        (0, 0, 512),
        (0, 512, 512),
        (1, 1024, 512),
        (1, 1536, 512),
    ]


def test_index_bundle_crosses_1024_token_lmcache_boundary():
    plan = plan_bundle_copies(list(range(4096)), [1024] * 4, 2304)
    assert list(
        zip(plan.chunk.tolist(), plan.slot.tolist(), plan.tokens.tolist(), strict=False)
    ) == [
        (0, 0, 1024),
        (1, 1024, 1024),
        (2, 2048, 256),
        (2, 2304, 768),
        (3, 3072, 1024),
    ]


def test_noncontiguous_npu_bundles_are_separate_and_planes_are_correct():
    plan = plan_bundle_copies([512, 513, 700, 701], [4], 512)
    assert list(zip(plan.slot.tolist(), plan.tokens.tolist(), strict=False)) == [
        (512, 2),
        (700, 2),
    ]
    copies = bind_copy_addresses(
        plan,
        [10000],
        [20000, 30000],
        [4],
        [512, 64],
        2,
        device_to_host=False,
    )
    assert tuple(copies[0]) == (20000 + 512 * 1024, 10000, 2 * 1024)
    assert tuple(copies[1]) == (30000 + 512 * 128, 10000 + 4 * 1024, 2 * 128)
    assert copies[2][1] == 10000 + 2 * 1024


def test_d2h_reverses_pointers_and_partial_chunk_is_exact():
    plan = plan_bundle_copies([0, 1, 2, 3, 4], [3, 2], 4)
    copies = bind_copy_addresses(
        plan,
        [100, 200],
        [300],
        [3, 2],
        [128],
        2,
        device_to_host=True,
    )
    assert list(map(tuple, copies)) == [
        (100, 300, 3 * 256),
        (200, 300 + 3 * 256, 256),
        (456, 300 + 4 * 256, 256),
    ]


def test_second_plane_uses_physical_host_chunk_size_not_copied_tail():
    plan = plan_bundle_copies([4, 5], [2], 8)
    copies = bind_copy_addresses(
        plan,
        [1000],
        [2000, 3000],
        [2],
        [512, 64],
        2,
        device_to_host=False,
        host_chunk_tokens=[1024],
    )
    assert list(map(tuple, copies)) == [
        (2000 + 4 * 1024, 1000, 2 * 1024),
        (3000 + 4 * 128, 1000 + 1024 * 1024, 2 * 128),
    ]


@pytest.mark.parametrize(
    "bundle,chunk", [(512, 1024), (2304, 1024), (768, 1536), (1152, 4096), (7, 11)]
)
def test_periodic_plan_matches_token_reference(bundle, chunk):
    cycle = module.DmaCycle.build(bundle, chunk)
    length = cycle.period * 3 + chunk + 3
    logical = torch.arange(length)
    # Reverse and separate the physical bundles; never assume contiguous HBM.
    slots = (length // bundle + 2 - logical // bundle) * 2 * bundle + logical % bundle
    starts = list(range(0, length, chunk))
    ends = [min(x + chunk, length) for x in starts]
    # Exercise nonzero start, skipped chunks, and a partial final chunk.
    for first in (0, 1):
        ss, ee = starts[first::2], ends[first::2]
        selected = torch.cat([slots[s:e] for s, e in zip(ss, ee, strict=False)])
        sizes = [e - s for s, e in zip(ss, ee, strict=False)]
        expected = plan_bundle_copies(selected, sizes, bundle)
        actual = cycle.plan_ranges(slots, ss, ee)
        for field in ("chunk", "slot", "chunk_token", "tokens"):
            assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_cycles_derive_actual_layout():
    latent = torch.empty((2, 128, 576))
    index = torch.empty((2, 128, 128))
    a, b = module.build_group_cycles(latent, index, 1536, 2)
    assert (a.bundle_tokens, b.bundle_tokens) == (512, 2304)
    c, d = module.build_group_cycles(latent, index, 2048, 4)
    assert (c.bundle_tokens, d.bundle_tokens) == (1024, 4608)
    assert a.chunk_tokens == 1536
    assert c.chunk_tokens == 2048


def test_no_python_loops_in_dma_planner_or_binding():
    tree = ast.parse(MODULE_PATH.read_text())
    assert not any(
        isinstance(
            n,
            (
                ast.For,
                ast.While,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        )
        for n in ast.walk(tree)
    )


def test_incremental_binding_reuses_history_and_matches_full_rebind():
    cycle = module.DmaCycle.build(bundle_tokens=4, chunk_tokens=3)
    slots = torch.arange(12)
    owners = [object() for _ in range(3)]
    host_addresses = {id(owner): 10000 + i * 1000 for i, owner in enumerate(owners)}
    observed = []

    def host_ptr(owner):
        observed.append(owner)
        return host_addresses[id(owner)]

    def host_tokens(_owner):
        return 3

    first_plan = cycle.plan_ranges(slots, [0, 3], [3, 6])
    first = module.bind_incremental_copy_addresses(
        first_plan,
        owners[:2],
        [0, 3],
        [3, 6],
        [20000],
        [16],
        2,
        host_ptr,
        host_tokens,
        None,
        slot_prefix_unchanged=False,
    )
    assert observed == owners[:2]
    observed.clear()
    full_plan = cycle.plan_ranges(slots, [0, 3, 6], [3, 6, 9])
    second = module.bind_incremental_copy_addresses(
        full_plan,
        owners,
        [0, 3, 6],
        [3, 6, 9],
        [20000],
        [16],
        2,
        host_ptr,
        host_tokens,
        first,
        slot_prefix_unchanged=True,
    )
    assert observed == owners[2:]
    expected = bind_copy_addresses(
        full_plan,
        list(host_addresses.values()),
        [20000],
        [3, 3, 3],
        [16],
        2,
        device_to_host=False,
    )
    assert second.rows == expected


def test_incremental_binding_rebinds_changed_tail_or_npu_address():
    cycle = module.DmaCycle.build(bundle_tokens=4, chunk_tokens=3)
    slots = torch.arange(9)
    owners = [object() for _ in range(3)]
    addresses = {id(owner): 10000 + i * 1000 for i, owner in enumerate(owners)}
    observed = []

    def host_ptr(owner):
        observed.append(owner)
        return addresses[id(owner)]

    def bind(current_owners, starts, ends, npu_ptr, previous, stable):
        plan = cycle.plan_ranges(slots, starts, ends)
        return module.bind_incremental_copy_addresses(
            plan,
            current_owners,
            starts,
            ends,
            [npu_ptr],
            [16],
            2,
            host_ptr,
            lambda _owner: 3,
            previous,
            slot_prefix_unchanged=stable,
        )

    first = bind(owners[:2], [0, 3], [3, 6], 20000, None, False)
    observed.clear()
    replaced = bind([owners[0], owners[2]], [0, 3], [3, 6], 20000, first, True)
    assert observed == [owners[0], owners[2]]
    observed.clear()
    bind([owners[0], owners[2]], [0, 3], [3, 6], 30000, replaced, True)
    assert observed == [owners[0], owners[2]]


@pytest.mark.parametrize("bundle,internal", [(512, None), (2304, [2048])])
def test_incremental_binding_matches_80k_full_plan(bundle, internal):
    cycle = module.DmaCycle.build(bundle, 1024, internal)
    starts = list(range(0, 81920, 1024))
    ends = [start + 1024 for start in starts]
    slots = torch.arange(81920)
    owners = [object() for _ in starts]
    ptrs = {id(owner): 1000000 + i * 16384 for i, owner in enumerate(owners)}

    def bind(count, previous):
        plan = cycle.plan_ranges(slots, starts[:count], ends[:count])
        return module.bind_incremental_copy_addresses(
            plan,
            owners[:count],
            starts[:count],
            ends[:count],
            [2000000, 3000000],
            [576, 128],
            2,
            lambda owner: ptrs[id(owner)],
            lambda _owner: 1024,
            previous,
            slot_prefix_unchanged=True,
        )

    previous = bind(76, None)
    current = bind(80, previous)
    full = bind_copy_addresses(
        cycle.plan_ranges(slots, starts, ends),
        [ptrs[id(owner)] for owner in owners],
        [2000000, 3000000],
        [1024] * 80,
        [576, 128],
        2,
        device_to_host=False,
    )
    assert current.rows == full


def test_indexer_split_slabs_never_cross_a_dma_segment():
    _, cycle = module.build_group_cycles(
        torch.empty((2, 128, 576)), torch.empty((2, 128, 128)), 1024, 2, (512, 64)
    )
    logical = torch.arange(11003)
    bundle = logical // 2304 + 1
    offset = logical % 2304
    slots = torch.where(
        offset < 2048, bundle * 2048 + offset, 100000 + bundle * 256 + offset - 2048
    )
    starts = list(range(4096, 11003, 1024))
    ends = [min(s + 1024, 11003) for s in starts]
    plan = cycle.plan_ranges(slots, starts, ends)
    reconstructed = torch.cat(
        [
            torch.arange(s, s + n)
            for s, n in zip(plan.slot.tolist(), plan.tokens.tolist(), strict=False)
        ]
    )
    assert torch.equal(reconstructed, slots[4096:11003])
    assert plan.tokens.sum() == 11003 - 4096
