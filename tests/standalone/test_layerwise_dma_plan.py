"""No-NPU checks for the P-node bundle/chunk DMA address plan."""

import importlib.util
import sys
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
    assert [(s.chunk, s.slot, s.tokens) for s in plan] == [
        (0, 0, 512), (0, 512, 512), (1, 1024, 512), (1, 1536, 512)
    ]


def test_index_bundle_crosses_1024_token_lmcache_boundary():
    plan = plan_bundle_copies(list(range(4096)), [1024] * 4, 2304)
    assert [(s.chunk, s.slot, s.tokens) for s in plan] == [
        (0, 0, 1024), (1, 1024, 1024), (2, 2048, 256),
        (2, 2304, 768), (3, 3072, 1024),
    ]


def test_noncontiguous_npu_bundles_are_separate_and_planes_are_correct():
    plan = plan_bundle_copies([512, 513, 700, 701], [4], 512)
    assert [(s.slot, s.tokens) for s in plan] == [(512, 2), (700, 2)]
    copies = bind_copy_addresses(
        plan, [10000], [20000, 30000], [4], [512, 64], 2,
        device_to_host=False,
    )
    assert copies[0] == (20000 + 512 * 1024, 10000, 2 * 1024)
    assert copies[1] == (30000 + 512 * 128, 10000 + 4 * 1024, 2 * 128)
    assert copies[2][1] == 10000 + 2 * 1024


def test_d2h_reverses_pointers_and_partial_chunk_is_exact():
    plan = plan_bundle_copies([0, 1, 2, 3, 4], [3, 2], 4)
    copies = bind_copy_addresses(
        plan, [100, 200], [300], [3, 2], [128], 2,
        device_to_host=True,
    )
    assert copies == [
        (100, 300, 3 * 256),
        (200, 300 + 3 * 256, 256),
        (456, 300 + 4 * 256, 256),
    ]


def test_second_plane_uses_physical_host_chunk_size_not_copied_tail():
    plan = plan_bundle_copies([4, 5], [2], 8)
    copies = bind_copy_addresses(
        plan, [1000], [2000, 3000], [2], [512, 64], 2,
        device_to_host=False, host_chunk_tokens=[1024],
    )
    assert copies == [
        (2000 + 4 * 1024, 1000, 2 * 1024),
        (3000 + 4 * 128, 1000 + 1024 * 1024, 2 * 128),
    ]
