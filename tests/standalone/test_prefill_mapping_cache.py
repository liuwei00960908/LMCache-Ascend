# SPDX-License-Identifier: Apache-2.0
"""Execute production slot-map slicing/cache on CPU, including graph tensors."""

import ast
from pathlib import Path

import pytest
import torch


@pytest.fixture
def api():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    names = {"_slice_layerwise_slot_mapping", "_cached_layerwise_slot_mapping"}
    functions = [
        n
        for n in ast.parse(path.read_text(encoding="utf8")).body
        if getattr(n, "name", None) in names
    ]
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    ns = {"torch": torch}
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), ns)
    return ns["_cached_layerwise_slot_mapping"]


@pytest.mark.parametrize("layers", [22, 79])
def test_two_banks_are_prepared_once_including_draft_layer(api, monkeypatch, layers):
    # 78 target latent layers + one MTP layer; sparse indexer has 22 physical rows.
    with torch.inference_mode():
        banks = (torch.arange(12), torch.arange(12) + 100)
    monkeypatch.setattr(
        torch, "cat", lambda *a, **kw: pytest.fail("contiguous prefix copied")
    )
    cache, prepared, previous = {}, 0, {}
    for layer in range(layers):
        bank = layer % 2
        chunks, full, new = api(cache, banks[bank], [4, 8], [8, 12], 0)
        prepared += new
        assert torch.equal(full, banks[bank][4:12])
        assert full.data_ptr() == banks[bank][4:].data_ptr()
        if bank in previous:
            assert full is previous[bank]
        previous[bank] = full
        assert len(chunks) == 2
    assert prepared == 2


def test_disjoint_ranges_cat_only_once_per_bank_and_refresh_next_forward(
    api, monkeypatch
):
    original_cat = torch.cat
    calls = []

    def cat(*args, **kwargs):
        calls.append(1)
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", cat)
    mapping, cache = torch.arange(8), {}
    for _ in range(10):
        _, full, _ = api(cache, mapping, [0, 6], [2, 8])
        assert full.tolist() == [0, 1, 6, 7]
    assert len(calls) == 1
    mapping.add_(10)  # runner may reuse tensor storage in a later forward
    _, full, new = api({}, mapping, [0, 6], [2, 8])
    assert new and full.tolist() == [10, 11, 16, 17]
    assert len(calls) == 2


def test_non_prefill_mapping_is_not_cached_and_keeps_snapshot_semantics(api):
    mapping = torch.arange(4)
    _, first, _ = api(None, mapping, [0, 2], [2, 4])
    mapping.add_(10)
    _, second, new = api(None, mapping, [0, 2], [2, 4])
    assert new and first.tolist() == [0, 1, 2, 3]
    assert second.tolist() == [10, 11, 12, 13]


def test_base_offsets_and_invalid_windows_still_checked_once(api):
    cache, mapping = {}, torch.arange(4)
    _, full, new = api(cache, mapping, [100, 102], [102, 104], 100)
    assert new and torch.equal(full, mapping)
    with pytest.raises(ValueError, match="outside"):
        api(cache, mapping, [100], [104], 101)
