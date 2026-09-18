# SPDX-License-Identifier: Apache-2.0
"""Startup instrumentation keeps reader/RemoteFill order and failure behavior."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("failure_stage", [None, "decoder_remote_fill"])
def test_startup_phases_preserve_initialization_and_rollback(passive, failure_stage):
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendLMCacheEngine"
    )
    cls.body = [n for n in cls.body if getattr(n, "name", None) == "post_init"]
    calls = []
    failure = RuntimeError("native initialization failed")

    class Base:
        def post_init(self, **kwargs):
            calls.append("base")

    @contextmanager
    def phase(name, **details):
        assert details == {"rank": 3 if passive else 0}
        calls.append(name + ":begin")
        yield
        calls.append(name + ":end")

    def reader(*args):
        assert calls[-1] == "group1_external_reader:begin"
        calls.append("reader")
        return object()

    def initialize():
        assert calls[-1] == "decoder_remote_fill:begin"
        calls.append("remote_fill")
        if failure_stage:
            raise failure

    ns = dict(
        LMCacheEngine=Base,
        startup_phase=phase,
        serving_perf_enabled=lambda: False,
        RemoteExternalPageReader=reader,
        logger=object(),
        log_remote_fill_diagnostic=lambda *a, **kw: None,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    engine = ns[cls.name]()
    engine.metadata = NS(worker_id=3 if passive else 0)
    engine.config = NS(pd_role="receiver")
    engine._persistent_direct_hbm_split_group_enabled = lambda: passive
    engine._is_passive = lambda: passive
    engine._initialize_decoder_remote_fill = initialize
    engine._rollback_group1_direct_hbm_startup = lambda: calls.append("rollback")
    engine.is_store_async = engine._direct_store_enabled = False
    if failure_stage:
        with pytest.raises(RuntimeError) as caught:
            engine.post_init()
        assert caught.value is failure
    else:
        engine.post_init()
    expected = ["base"]
    if passive:
        expected += [
            "group1_external_reader:begin",
            "reader",
            "group1_external_reader:end",
        ]
    expected += ["decoder_remote_fill:begin", "remote_fill"]
    expected += ["rollback" if failure_stage else "decoder_remote_fill:end"]
    assert calls == expected
