# SPDX-License-Identifier: Apache-2.0
"""Exercise production adapter routing without importing the NPU runtime."""

import ast
from collections import deque
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS
from weakref import WeakSet

import pytest


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"


def production_class(path, name, methods, namespace, base=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if getattr(n, "name", None) in methods]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())] if base else []
    cls.decorator_list = []
    if base:
        namespace["Base"] = base
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class BaseAdapter:
    def __init__(self, config, role, parent):
        self.config = config
        self.kv_role = "kv_both"
        self.lmcache_engine = NS()
        self._parent = parent
        self.normal_saves = []

    def save_kv_layer(self, layer, *_args, **_kwargs):
        self.normal_saves.append(layer)

    def start_load_kv(self, *_args, **_kwargs):
        self.at_prime = dict(
            self._parent._get_connector_metadata().requests[0].request_configs
        )

    def _release_finished_worker_requests(self, ids):
        self.released = tuple(ids)


def adapter(
    monkeypatch,
    *,
    p_node=True,
    direct=True,
    remote=False,
    store_async=True,
    queue_size=2,
    role="worker",
):
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", str(p_node).lower())
    extra = dict(
        mooncake_direct_npu_prefill_store=direct,
        mooncake_page_first_multi_buffer=True,
        mooncake_layer_merged_page_objects=True,
        save_only_first_rank=True,
        use_ascend_direct=True,
        save_chunk_meta=False,
    )
    config = NS(
        use_layerwise=True,
        store_async=store_async,
        store_async_max_queue_size=queue_size,
        remote_url="mooncakestore://localhost:58888/",
        enable_shared_cpu_cache=True,
        enable_remote_lmcache_store=remote,
        pd_role="sender",
        dsa_two_groups=True,
        extra_config=extra,
        get_extra_config_value=extra.get,
    )
    request = NS(
        req_id="r",
        token_ids=list(range(5)),
        block_allocation_mode="prefill_child",
        is_sparse_decode=False,
        is_decode_window_save=False,
        is_last_prefill=True,
        slot_mapping=[[0]],
        indexer_slot_mapping=[[0]],
        live_source_requested=False,
        request_configs={},
        save_spec=NS(can_save=True, can_save_indexer=True),
    )
    metadata = NS(requests=[request])
    parent = NS(_connector_metadata=metadata, _get_connector_metadata=lambda: metadata)
    cls = production_class(
        ADAPTER,
        "LMCacheAscendConnectorV1Impl",
        {
            "__init__",
            "_direct_prefill_requests",
            "save_kv_layer",
            "_finish_save_batch",
            "_finish_layerwise_prefill_batch",
            "_source_ready_event",
            "_submit_direct_prefill_requests",
            "start_load_kv",
            "_release_finished_worker_requests",
            "_forget_layerwise_store_results",
        },
        dict(
            os=os,
            logger=logging.getLogger("fallback-test"),
            KVConnectorRole=NS(SCHEDULER="scheduler"),
            Mapping=dict,
            _validate_remote_fill_sleep_mode=lambda *a: None,
            _remote_fill_request_qualified=lambda req: False,
            _prepare_remote_fill_persistent_placement=lambda cfg, **kw: cfg.update(
                {"placement": kw}
            ),
            _persistent_direct_hbm_enabled=lambda cfg: True,
            serving_perf_enabled=lambda: False,
        ),
        BaseAdapter,
    )
    obj = cls(config, role, parent)
    obj._latent_layer_names = ["layer0", "layer1", "layer2", "layer3"]
    obj._indexer_layer_names = ["index0", "index1"]
    obj._producer_fence_handoff_targets = lambda requests: ()
    return obj, request


@pytest.mark.parametrize("direct,remote", [(True, False), (False, True), (True, True)])
def test_banked_prefill_never_selects_all_layer_sources(monkeypatch, direct, remote):
    obj, _ = adapter(monkeypatch, direct=direct, remote=remote)
    assert obj._direct_prefill_requests() is None


def test_every_layer_save_reaches_original_banked_writer(monkeypatch, caplog):
    obj, _ = adapter(monkeypatch)
    obj._preflight_direct_store = lambda requests: True
    for layer in obj._latent_layer_names + obj._indexer_layer_names:
        obj.save_kv_layer(layer, None, None)
    assert obj.normal_saves == obj._latent_layer_names + obj._indexer_layer_names
    assert (
        len([r for r in caplog.records if "PREFILL_STORE_FALLBACK" in r.message]) == 1
    )


@pytest.mark.parametrize("p_node", [False, True])
def test_regular_direct_route_unchanged(monkeypatch, p_node):
    obj, request = adapter(monkeypatch, p_node=p_node)
    assert obj._direct_prefill_requests() == (None if p_node else [request])


@pytest.mark.parametrize("role", ["scheduler", "worker"])
def test_fallback_ignores_unused_direct_queue_constraints(monkeypatch, role):
    obj, _ = adapter(monkeypatch, queue_size=7, role=role)
    assert obj._direct_prefill_requests() is None


def test_normal_direct_mode_still_checks_queue_constraints(monkeypatch):
    with pytest.raises(ValueError, match="store_async_max_queue_size=2"):
        adapter(monkeypatch, p_node=False, queue_size=7)


def test_plain_layerwise_prefill_allows_store_async(monkeypatch):
    obj, _ = adapter(monkeypatch, direct=False, remote=False)
    assert obj._direct_prefill_requests() is None


def test_plain_non_prefill_layerwise_keeps_async_restriction(monkeypatch):
    with pytest.raises(ValueError, match="Layerwise storing"):
        adapter(monkeypatch, p_node=False, direct=False, remote=False)


def test_late_direct_submission_cannot_read_banks(monkeypatch):
    obj, request = adapter(monkeypatch)
    obj._direct_group_caches = lambda: pytest.fail("bank addresses revisited")
    obj._submit_direct_prefill_requests([request], finish_batch=True)


@pytest.mark.parametrize("final", [False, True])
def test_step_finalization_fences_persistence_before_handoff(monkeypatch, final):
    obj, request = adapter(monkeypatch, remote=True)
    request.is_last_prefill = final
    calls = []
    obj.lmcache_engine = NS(
        submit_layerwise_prefill_fills=lambda ids: calls.append(("fill", list(ids))),
        poll_layerwise_prefill_puts=lambda **kw: calls.append(("poll", kw)),
        wait_for_pending_sync_stores=lambda: calls.append("wait"),
        adopt_completed_layerwise_store=lambda r: calls.append(r),
        finish_layerwise_prefill_store=lambda *a, **kw: calls.append((a, kw)),
    )
    obj._completed_layerwise_stores = {("r", 0): "latent", ("r", 1): "index"}
    obj._finish_save_batch({})
    expected = [
        ("fill", ["r"]),
        ("poll", {"final": final, "req_ids": ("r",) if final else ()}),
        "wait",
    ]
    if final:
        expected.extend(["latent", "index"])
        expected.append(
            (
                ("r", {}),
                {
                    "required_store_end": 5,
                    "persistence_fenced": True,
                    "tokens": request.token_ids,
                },
            )
        )
    assert calls == expected
    assert obj._completed_layerwise_stores == {}


def test_remote_put_failure_never_publishes_handoff(monkeypatch):
    obj, _ = adapter(monkeypatch, remote=True)

    def failed_wait():
        raise RuntimeError("remote put failed")

    obj.lmcache_engine = NS(
        submit_layerwise_prefill_fills=lambda ids: None,
        poll_layerwise_prefill_puts=lambda **kw: None,
        wait_for_pending_sync_stores=failed_wait,
    )
    obj._completed_layerwise_stores = {("r", 0): object()}
    with pytest.raises(RuntimeError, match="remote put failed"):
        obj._finish_save_batch({})
    assert obj._completed_layerwise_stores == {}


def test_final_request_does_not_adopt_other_requests_local_progress(monkeypatch):
    obj, request = adapter(monkeypatch, remote=True)
    request.is_last_prefill = True
    adopted = []
    obj.lmcache_engine = NS(
        submit_layerwise_prefill_fills=lambda ids: None,
        poll_layerwise_prefill_puts=lambda **kw: None,
        wait_for_pending_sync_stores=lambda: None,
        adopt_completed_layerwise_store=adopted.append,
        finish_layerwise_prefill_store=lambda *a, **kw: None,
    )
    obj._completed_layerwise_stores = {
        ("r", 0): "latent-with-mtp",
        ("r", 1): "index-with-mtp",
    }
    obj._layerwise_local_store_results = {("unfinished", 0): "local-only"}
    obj._finish_save_batch({})
    assert adopted == ["latent-with-mtp", "index-with-mtp"]
    assert obj._layerwise_local_store_results == {("unfinished", 0): "local-only"}


def test_banked_placement_is_installed_before_generators_create_keys(monkeypatch):
    obj, request = adapter(monkeypatch, remote=True)
    obj.lmcache_engine.poll_layerwise_prefill_puts = lambda: None
    obj.start_load_kv(NS(attn_metadata={}))
    assert obj.at_prime == {"placement": {"group1_direct_hbm": True}}
    assert obj.at_prime == request.request_configs


def test_no_per_step_retirement_fence_for_empty_finished_set(monkeypatch):
    obj, _ = adapter(monkeypatch)
    obj.lmcache_engine.poll_layerwise_prefill_puts = lambda **kw: pytest.fail("wait")
    obj._release_finished_worker_requests([])
    assert obj.released == ()


def test_cancelled_request_is_fenced_before_its_local_progress_is_dropped(monkeypatch):
    obj, _ = adapter(monkeypatch)
    obj._layerwise_local_store_results = {("r", 0): "ready", ("other", 1): "keep"}
    calls = []
    obj.lmcache_engine.poll_layerwise_prefill_puts = lambda **kw: calls.append(kw)
    obj._release_finished_worker_requests(["r"])
    assert calls == [{"final": True, "req_ids": ("r",)}]
    assert obj._layerwise_local_store_results == {("other", 1): "keep"}
    assert obj.released == ("r",)


@dataclass
class DirectState:
    committed_end: dict = field(default_factory=dict)
    submitted_end: dict = field(default_factory=dict)
    remote_fill: object = None
    futures: deque = field(default_factory=deque)
    pending_keys: set = field(default_factory=set)


@dataclass
class ProducerState:
    handoff: object = None
    terminal: object = None
    session: object = None
    disabled_reason: str = ""
    futures: deque = field(default_factory=deque)
    viable_counted: bool = False
    active_counted: bool = False
    submitted_bytes: int = 0
    last_future: object = None
    metrics_started: bool = False


class BaseEngine:
    def __init__(self, config, metadata, *_args):
        self.config = config
        self.metadata = metadata
        self.kv_events_enabled = False

    def _is_passive(self):
        return self.metadata.worker_id != 0


def engine(monkeypatch, *, p_node=True, direct=True, remote=True, rank=0):
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", str(p_node).lower())
    extra = dict(
        mooncake_direct_npu_prefill_store=direct,
        use_ascend_direct=True,
        save_chunk_meta=False,
    )
    config = NS(
        use_layerwise=True,
        store_async=True,
        pd_role="sender",
        enable_remote_lmcache_store=remote,
        store_async_max_queue_size=2,
        chunk_size=256,
        blocking_timeout_secs=0,
        get_extra_config_value=extra.get,
    )
    cls = production_class(
        ROOT / "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {
            "__init__",
            "adopt_completed_layerwise_store",
            "finish_layerwise_prefill_store",
            "_recover_layerwise_prefill_persistence",
            "_republish_layerwise_cpu_chunk",
            "_track_sync_store_futures",
            "_remote_fill_prepare_request",
            "_finish_remote_fill",
            "wait_for_pending_sync_stores",
            "drain_remote_fill_terminal_results",
            "direct_prefill_store_enabled",
            "drop_direct_store_states",
            "poll_layerwise_prefill_puts",
            "_queue_layerwise_cpu_fill",
            "submit_layerwise_prefill_fills",
        },
        dict(
            os=os,
            threading=threading,
            deque=deque,
            WeakSet=WeakSet,
            mooncake_layer_pages_enabled=lambda cfg: True,
            _DirectStoreRequestState=DirectState,
            ProducerRequestState=ProducerState,
            REMOTE_FILL_REQUEST_CONFIG_KEY="lmcache.remote_fill",
            REMOTE_BACKEND_NAME="RemoteBackend",
            Mapping=dict,
            wait=wait,
        ),
        BaseEngine,
    )
    obj = cls(config, NS(worker_id=rank, world_size=4), None, None, None, None)

    # Use the real terminal result schema and coordinator completion code.
    path = ROOT / "lmcache_ascend/v1/remote_fill_producer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RemoteFillTerminalResult"
    )
    ns = dict(dataclass=dataclass, field=field)
    exec(compile(ast.Module(body=[result], type_ignores=[]), str(path), "exec"), ns)
    coordinator_cls = production_class(
        ROOT / "lmcache_ascend/v1/remote_fill_coordinator.py",
        "RemoteFillCoordinator",
        {"finish", "wait", "drain_terminal_results", "release"},
        dict(
            RemoteFillTerminalResult=ns[result.name],
            serving_perf_enabled=lambda: False,
            log_remote_fill_diagnostic=lambda *a, **k: None,
            logger=logging.getLogger("fallback-test"),
            RemoteFillFatalError=type("Fatal", (RuntimeError,), {}),
        ),
    )
    coordinator = coordinator_cls()
    metrics = NS(
        timing_enabled=False, finish_attempt=lambda *a: None, add_bytes=lambda *a: None
    )
    coordinator.get_metrics = lambda: metrics

    def prepare(req_id, configs, state):
        state.handoff = NS(transfer_id=configs["lmcache.remote_fill"])
        return True

    coordinator.prepare_request = prepare
    obj._remote_fill_coordinator = coordinator
    obj._get_remote_fill_coordinator = lambda: coordinator
    return obj


@pytest.mark.parametrize("p_node,expected", [(False, True), (True, False)])
@pytest.mark.parametrize("direct,remote", [(True, False), (False, True), (True, True)])
def test_engine_disables_all_layer_npu_jobs_only_for_prefill(
    monkeypatch,
    p_node,
    expected,
    direct,
    remote,
):
    obj = engine(monkeypatch, p_node=p_node, direct=direct, remote=remote)
    assert obj.direct_prefill_store_enabled() is expected


def adopt(obj, group, end):
    obj.adopt_completed_layerwise_store(
        NS(
            request_id="r",
            kv_group=group,
            committed_end=end,
            keys=[],
        )
    )


def test_persistent_only_receipt_uses_completed_both_group_frontiers(monkeypatch):
    obj = engine(monkeypatch)
    for group in (0, 1):
        adopt(obj, group, 256)
        adopt(obj, group, 511)
    future = Future()
    future.set_result(None)
    obj._pending_sync_store_futures.add(future)
    obj.finish_layerwise_prefill_store(
        "r",
        {"lmcache.remote_fill": "transfer"},
        required_store_end=511,
    )
    assert not obj._pending_sync_store_futures
    assert obj.drain_remote_fill_terminal_results() == {
        "r": {
            "transfer_id": "transfer",
            "outcome": "PERSISTENT_ONLY",
            "persistent_common_end": 511,
            "required_store_end": 511,
        }
    }
    assert obj._direct_store_states["r"].remote_fill.session is None
    obj.finish_layerwise_prefill_store(
        "r",
        {"lmcache.remote_fill": "transfer"},
        required_store_end=511,
    )
    assert obj.drain_remote_fill_terminal_results() == {}


@pytest.mark.parametrize("ends", [{0: 511}, {0: 511, 1: 256}])
def test_missing_or_short_group_refuses_completion(monkeypatch, ends):
    obj = engine(monkeypatch)
    for group, end in ends.items():
        adopt(obj, group, end)
    with pytest.raises(RuntimeError, match="persistence is incomplete"):
        obj.finish_layerwise_prefill_store(
            "r",
            {"lmcache.remote_fill": "transfer"},
            required_store_end=511,
        )
    assert obj.drain_remote_fill_terminal_results() == {}


@pytest.mark.parametrize("pending", [False, True])
def test_failed_or_pending_put_cannot_emit_success(monkeypatch, pending):
    obj = engine(monkeypatch)
    for group in (0, 1):
        adopt(obj, group, 511)
    future = Future()
    if not pending:
        future.set_exception(ValueError("failed put"))
    obj._pending_sync_store_futures.add(future)
    with pytest.raises(TimeoutError if pending else ValueError):
        obj.finish_layerwise_prefill_store(
            "r",
            {"lmcache.remote_fill": "transfer"},
            required_store_end=511,
        )
    assert obj.drain_remote_fill_terminal_results() == {}


@pytest.mark.parametrize(
    "remote,rank,configs",
    [
        (False, 0, {}),
        (True, 1, {"lmcache.remote_fill": "transfer"}),
        (True, 0, {}),
    ],
)
def test_no_handoff_for_passive_or_ordinary_requests(
    monkeypatch,
    remote,
    rank,
    configs,
):
    obj = engine(monkeypatch, remote=remote, rank=rank)
    obj.finish_layerwise_prefill_store(
        "r", configs, required_store_end=511, tokens=list(range(511))
    )
    assert obj.drain_remote_fill_terminal_results() == {}


def test_zero_length_frontier_and_cleanup_do_not_create_native_jobs(monkeypatch):
    obj = engine(monkeypatch)
    obj.finish_layerwise_prefill_store(
        "r",
        {"lmcache.remote_fill": "transfer"},
        required_store_end=0,
    )
    obj.drop_direct_store_states({"r"})
    assert obj._direct_store_states == {}
    assert obj.drain_remote_fill_terminal_results() == {
        "r": {
            "transfer_id": "transfer",
            "outcome": "PERSISTENT_ONLY",
            "persistent_common_end": 0,
            "required_store_end": 0,
        }
    }
