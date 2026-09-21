# SPDX-License-Identifier: Apache-2.0
"""Verify the direct page-pointer DMA preparation is behavior-preserving."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DMA_PATH = ROOT / "lmcache_ascend/v1/npu_connector/layerwise_dma.py"
CONNECTOR_PATH = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"


def _load_dma_module():
    spec = importlib.util.spec_from_file_location(
        "layerwise_dma_pointer_equivalence", DMA_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_host_pointer_helper(page_type):
    tree = ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_layer_memory_host_ptr"
    )
    namespace = {"LayerPageMemoryObj": page_type}
    source = ast.Module(body=[function], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(source), str(CONNECTOR_PATH), "exec"),
        namespace,
    )
    return namespace["_layer_memory_host_ptr"]


class _Page:
    """Small page double exposing the production page-pointer contract."""

    def __init__(self, layers: int, layer_elements: int, valid_tokens: int):
        self.valid_tokens = valid_tokens
        self._storage = torch.empty(layers * layer_elements, dtype=torch.int8)
        self._layer_elements = layer_elements

    def layer_tensor(self, layer_id: int) -> torch.Tensor:
        start = layer_id * self._layer_elements
        return self._storage.narrow(0, start, self._layer_elements)

    def layer_data_ptr(self, layer_id: int) -> int:
        return int(self._storage.data_ptr()) + layer_id * self._layer_elements


class _LegacyObject:
    def __init__(self, elements: int):
        self.tensor = torch.empty(elements, dtype=torch.int8)

    @property
    def data_ptr(self) -> int:
        return int(self.tensor.data_ptr())


def test_page_pointer_and_legacy_pointer_match_tensor_view_addresses():
    helper = _load_host_pointer_helper(_Page)
    page = _Page(layers=3, layer_elements=257, valid_tokens=129)

    for layer_id in range(3):
        assert helper(page, layer_id) == page.layer_tensor(layer_id).data_ptr()

    legacy = _LegacyObject(257)
    assert helper(legacy, 0) == legacy.tensor.data_ptr()


def test_dma_rows_are_identical_for_old_and_direct_page_pointer_preparation():
    dma = _load_dma_module()
    helper = _load_host_pointer_helper(_Page)
    pages = [
        _Page(layers=2, layer_elements=257 * 4, valid_tokens=512),
        _Page(layers=2, layer_elements=257 * 4, valid_tokens=129),
    ]
    starts = [0, 512]
    ends = [512, 641]
    chunk_sizes = [512, 129]
    widths = [257]
    npu_ptrs = [0x200000]

    plan = dma.plan_bundle_copies(
        list(range(641)), chunk_sizes, bundle_tokens=512
    )
    old_host_ptrs = [
        int(page.layer_tensor(1).data_ptr()) for page in pages
    ]
    new_host_ptrs = [helper(page, 1) for page in pages]
    old_rows = dma.bind_copy_addresses(
        plan,
        old_host_ptrs,
        npu_ptrs,
        chunk_sizes,
        widths,
        1,
        device_to_host=True,
        host_chunk_tokens=[page.valid_tokens for page in pages],
    )
    new_rows = dma.bind_copy_addresses(
        plan,
        new_host_ptrs,
        npu_ptrs,
        chunk_sizes,
        widths,
        1,
        device_to_host=True,
        host_chunk_tokens=[page.valid_tokens for page in pages],
    )
    assert new_rows == old_rows
