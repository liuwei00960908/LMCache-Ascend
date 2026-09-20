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
