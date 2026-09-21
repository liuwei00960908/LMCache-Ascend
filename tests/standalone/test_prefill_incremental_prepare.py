# SPDX-License-Identifier: Apache-2.0
"""Regression checks for suffix-only P-node prefill preparation."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "lmcache_ascend/v1/cache_engine.py"


def _load_store_plan():
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendLMCacheEngine"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_layerwise_prefill_store_plan"
    )
    source = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(source), str(ENGINE_PATH), "exec"), namespace)
    return namespace["_layerwise_prefill_store_plan"]


class _TokenDatabase:
    def __init__(self) -> None:
        self.full_calls = 0
        self.suffix_calls: list[dict[str, object]] = []
        self.suffix_key = object()

    def process_tokens(self, **_kwargs):
        self.full_calls += 1
        return iter(())

    def process_tokens_from_prefix(self, tokens, **kwargs):
        self.suffix_calls.append({"tokens": tokens, **kwargs})
        return iter(((256, 512, self.suffix_key),))


def test_incremental_prepare_plans_only_the_new_suffix():
    plan_fn = _load_store_plan()
    token_database = _TokenDatabase()
    engine = SimpleNamespace(
        config=SimpleNamespace(chunk_size=256),
        token_database=token_database,
        _layerwise_prefill_store_frontiers={"request": {0: (256, 1234)}},
    )

    plan, base, frontier = plan_fn(
        engine,
        req_id="request",
        tokens=[0] * 512,
        mask=None,
        request_configs=None,
        kv_group=0,
        incremental=True,
    )

    assert base == 256
    assert frontier == (256, 1234)
    assert list(plan) == [(256, 512, token_database.suffix_key)]
    assert token_database.full_calls == 0
    assert token_database.suffix_calls == [
        {
            "tokens": [0] * 512,
            "prefix_token_count": 256,
            "prefix_hash": 1234,
            "request_configs": None,
            "kv_group": 0,
        }
    ]


def test_incremental_prepare_does_not_replan_an_unchanged_frontier():
    plan_fn = _load_store_plan()
    token_database = _TokenDatabase()
    engine = SimpleNamespace(
        config=SimpleNamespace(chunk_size=256),
        token_database=token_database,
        _layerwise_prefill_store_frontiers={"request": {0: (512, 1234)}},
    )

    plan, base, frontier = plan_fn(
        engine,
        req_id="request",
        tokens=[0] * 512,
        mask=None,
        request_configs=None,
        kv_group=0,
        incremental=True,
    )

    assert list(plan) == []
    assert base == 512
    assert frontier == (512, 1234)
    assert token_database.full_calls == 0
    assert token_database.suffix_calls == []
