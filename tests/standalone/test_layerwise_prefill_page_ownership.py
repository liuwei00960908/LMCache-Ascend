# SPDX-License-Identifier: Apache-2.0
"""Exercise request-scoped page ownership without importing NPU extensions."""

from __future__ import annotations

import ast
from pathlib import Path
import threading


class _Page:
    def __init__(self) -> None:
        self.refs = 1

    def ref_count_up(self) -> None:
        self.refs += 1

    def ref_count_down(self) -> None:
        self.refs -= 1

    @property
    def can_evict(self) -> bool:
        return self.refs == 1


def _owner() -> object:
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendLMCacheEngine"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_retain_layerwise_prefill_pages", "release_layerwise_prefill_pages"}
    ]
    source = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *methods,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)
    owner = type("Owner", (), namespace)()
    owner._engine_state_lock = threading.RLock()
    owner._layerwise_prefill_page_owners = {}
    return owner


def test_pages_remain_unevictable_until_request_finishes() -> None:
    owner = _owner()
    released_dma: list[str] = []
    owner.gpu_connector = type(
        "Connector",
        (),
        {
            "release_layerwise_prefill_dma_cache": lambda _self, req: (
                released_dma.append(req)
            )
        },
    )()
    first, second = _Page(), _Page()
    owner._retain_layerwise_prefill_pages("a", [first, first, second])
    owner._retain_layerwise_prefill_pages("a", [first])
    owner._retain_layerwise_prefill_pages("b", [first])

    assert (first.refs, second.refs) == (3, 2)
    assert not first.can_evict and not second.can_evict

    owner.release_layerwise_prefill_pages("a")
    assert (first.refs, second.refs) == (2, 1)
    assert not first.can_evict and second.can_evict

    owner.release_layerwise_prefill_pages("a")
    owner.release_layerwise_prefill_pages("b")
    assert (first.refs, second.refs) == (1, 1)
    assert first.can_evict and second.can_evict
    assert released_dma == ["a", "a", "b"]
