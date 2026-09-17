# SPDX-License-Identifier: Apache-2.0
"""Cache-hit PD completion: run production routing with CPU-only backends."""

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from test_layerwise_cpu_fill import (
    Key as PageKey,
    Page,
    Queue,
    module as queue_module,
    storage_manager,
)
from test_prefill_direct_fallback import adapter, adopt, engine, production_class


@dataclass(frozen=True)
class Key:
    group: int
    start: int
    end: int
    layer: int = -1

    def split_layers(self, count):
        return [Key(self.group, self.start, self.end, layer) for layer in range(count)]


class Storage:
    """Only the transport/allocator boundary is replaced, not engine logic."""

    def __init__(self):
        self.remote_pages = set()
        self.remote_layers = set()
        self.local_pages = {}
        self.local_layers = {}
        self.probes = []
        self.reads = []
        self.puts = []
        self.completion = Future()
        self.completion.set_result(None)
        self.storage_backends = {"RemoteBackend": self}

    def batched_external_pages_exist(self, keys):
        self.probes.append(("pages", list(keys)))
        return [key in self.remote_pages for key in keys]

    def batched_contains(self, keys):
        self.probes.append(("layers", list(keys)))
        return next(
            (index for index, key in enumerate(keys) if key not in self.remote_layers),
            len(keys),
        )

    def batched_get_layer_page_prefix(self, keys):
        assert len(keys) == 1
        key = keys[0]
        self.reads.append(key)
        page = self.local_pages.get(key)
        if page is None:
            return [], 0
        page.ref_count_up()
        return [page], 1

    def get_blocking(self, key):
        self.reads.append(key)
        obj = self.local_layers.get(key)
        if obj is not None:
            obj.ref_count_up()
        return obj

    def batched_put_layer_pages(self, keys, pages, **kwargs):
        assert kwargs == {"req_id": "r", "publish_local_early": True}
        self.puts.append(list(keys))
        # StorageManager consumes the caller references and holds its own
        # until completion; the initial LocalCPU references remain alive.
        for page in pages:
            page.ref_count_up()
            self.completion.add_done_callback(
                lambda _, page=page: page.ref_count_down()
            )
            page.ref_count_down()
        return [self.completion]

    def batched_submit_put_task(self, keys, objects):
        self.puts.append(list(keys))
        for obj in objects:
            obj.ref_count_up()
            self.completion.add_done_callback(lambda _, obj=obj: obj.ref_count_down())
        return [self.completion]


def setup(monkeypatch, length=9, chunk_size=4):
    obj = engine(monkeypatch)
    storage = Storage()
    calls = []
    plan = {
        group: [
            Key(group, start, min(start + chunk_size, length))
            for start in range(0, length, chunk_size)
        ]
        for group in (0, 1)
    }

    def process_tokens(*, tokens, request_configs, kv_group):
        assert tokens == list(range(length))
        assert request_configs == {"lmcache.remote_fill": "transfer"}
        calls.append(kv_group)
        return ((key.start, key.end, key) for key in plan[kv_group])

    obj.token_database = NS(process_tokens=process_tokens)
    obj.storage_manager = storage
    obj._shared_local_cpu_backend = lambda: storage
    # Different physical layer counts, including the MTP layer.
    obj._num_layers_for_kv_group = lambda group: {0: 79, 1: 22}[group]
    obj._direct_group_caches = lambda: pytest.fail("must not read reused NPU banks")
    storage.remote_pages.update(plan[0] + plan[1])
    return obj, storage, plan, calls


def finish(obj, length=9):
    obj.finish_layerwise_prefill_store(
        "r",
        {"lmcache.remote_fill": "transfer"},
        required_store_end=length,
        tokens=list(range(length)),
    )


def assert_receipt(obj, length=9):
    assert obj.drain_remote_fill_terminal_results() == {
        "r": {
            "transfer_id": "transfer",
            "outcome": "PERSISTENT_ONLY",
            "persistent_common_end": length,
            "required_store_end": length,
        }
    }


@pytest.mark.parametrize("length,chunk", [(22971, 1024), (1024, 1024), (256, 256)])
def test_full_cache_hit_skips_store_but_can_handoff(monkeypatch, length, chunk):
    obj, storage, plan, calls = setup(monkeypatch, length, chunk)
    impl, request = adapter(monkeypatch, remote=True)
    impl.lmcache_engine = obj
    request.token_ids = list(range(length))
    request.request_configs = {"lmcache.remote_fill": "transfer"}
    request.save_spec.skip_leading_tokens = length
    request.slot_mapping = [NS(to=lambda **kwargs: None)]
    impl.device = "cpu"
    # This is the real skip decision that made committed={} on the server.
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache/lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    base = production_class(
        path,
        "LMCacheConnectorV1Impl",
        {"_prepare_layerwise_store_inputs"},
        {"torch": NS(long="long")},
    )
    for group in (0, 1):
        assert (
            base._prepare_layerwise_store_inputs(
                impl, request, request.save_spec, group
            )
            is None
        )
    assert not impl._completed_layerwise_stores
    impl._finish_save_batch({})
    assert_receipt(obj, length)
    assert calls == [0, 1]
    assert storage.probes == [("pages", plan[0]), ("pages", plan[1])]
    assert not storage.reads and not storage.puts


def test_normal_completed_store_does_not_hash_or_probe(monkeypatch):
    obj, storage, _, calls = setup(monkeypatch)
    for group in (0, 1):
        adopt(obj, group, 9)
    finish(obj)
    assert_receipt(obj)
    finish(obj)  # Repeated finalization must not probe or emit another terminal.
    assert obj.drain_remote_fill_terminal_results() == {}
    assert not calls and not storage.probes and not storage.reads


def test_only_missing_group_and_suffix_are_probed(monkeypatch):
    obj, storage, plan, calls = setup(monkeypatch)
    adopt(obj, 0, 9)
    adopt(obj, 1, 4)
    finish(obj)
    assert_receipt(obj)
    assert calls == [1]
    assert storage.probes == [("pages", plan[1][1:])]


def test_legacy_remote_hit_requires_every_physical_layer(monkeypatch):
    obj, storage, plan, _ = setup(monkeypatch)
    storage.remote_pages.clear()
    for group, keys in plan.items():
        for key in keys:
            storage.remote_layers.update(
                key.split_layers(obj._num_layers_for_kv_group(group))
            )
    finish(obj)
    assert_receipt(obj)
    assert not storage.reads and not storage.puts


def test_republish_only_missing_cpu_page_not_later_remote_legacy_hit(monkeypatch):
    obj, storage, plan, _ = setup(monkeypatch)
    missing, later = plan[0][1:]
    storage.remote_pages.difference_update((missing, later))
    storage.remote_layers.update(later.split_layers(79))
    page = storage.local_pages[missing] = Page()
    finish(obj)
    assert_receipt(obj)
    assert storage.puts == [[missing]]
    assert storage.reads == [missing]
    assert page.refs == 1


@pytest.mark.parametrize("outcome", ["pending", "failed", "success"])
def test_cpu_hit_alone_is_not_remote_completion(monkeypatch, outcome):
    obj, storage, plan, _ = setup(monkeypatch)
    missing = plan[0][-1]  # unfull final chunk
    storage.remote_pages.remove(missing)
    page = storage.local_pages[missing] = Page()
    storage.completion = Future()
    if outcome == "success":
        storage.completion.set_result(None)
    elif outcome == "failed":
        storage.completion.set_exception(ValueError("remote put failed"))
    if outcome == "success":
        finish(obj)
        assert_receipt(obj)
    else:
        with pytest.raises(TimeoutError if outcome == "pending" else ValueError):
            finish(obj)
        assert obj.drain_remote_fill_terminal_results() == {}
        assert obj._direct_store_states["r"].committed_end == {}
    if outcome == "pending":
        assert page.refs == 2  # LocalCPU plus the pending DMA source
        storage.completion.set_result(None)
        obj.wait_for_pending_sync_stores()
    assert page.refs == 1


def test_republish_legacy_local_objects(monkeypatch):
    obj, storage, plan, _ = setup(monkeypatch)
    missing = plan[1][-1]
    storage.remote_pages.remove(missing)
    layers = missing.split_layers(22)
    storage.local_layers.update({key: Page() for key in layers})
    finish(obj)
    assert_receipt(obj)
    assert storage.puts == [layers]
    assert all(page.refs == 1 for page in storage.local_layers.values())


def test_partial_legacy_remote_chunk_and_missing_cpu_refuse_handoff(monkeypatch):
    obj, storage, plan, _ = setup(monkeypatch)
    missing = plan[0][1]
    storage.remote_pages.remove(missing)
    layers = missing.split_layers(79)
    storage.remote_layers.update(layers[:-1])
    storage.local_layers.update({key: Page() for key in layers[:-1]})
    with pytest.raises(RuntimeError, match="absent from both remote"):
        finish(obj)
    assert obj.drain_remote_fill_terminal_results() == {}
    assert not storage.puts
    assert all(page.refs == 1 for page in storage.local_layers.values())


def test_submission_failure_releases_get_reference(monkeypatch):
    obj, storage, plan, _ = setup(monkeypatch)
    missing = plan[0][0]
    storage.remote_pages.remove(missing)
    page = storage.local_pages[missing] = Page()

    def fail(*args, **kwargs):
        raise ValueError("submission failed")

    storage.batched_put_layer_pages = fail
    with pytest.raises(ValueError, match="submission failed"):
        finish(obj)
    assert page.refs == 1
    assert obj.drain_remote_fill_terminal_results() == {}


@pytest.mark.parametrize("failed", [False, True])
def test_repair_uses_real_storage_manager_completion_and_ownership(monkeypatch, failed):
    obj = engine(monkeypatch)
    manager, local, remote_future = storage_manager()
    obj.storage_manager = manager
    obj._shared_local_cpu_backend = lambda: local
    key, page = PageKey(), Page()

    def get(keys):
        assert keys == [key]
        page.ref_count_up()
        return [page], 1

    def publish(keys, pages):
        assert keys == [key] and pages == [page]
        # Already in LocalCPU; idempotent re-publication retains its one ref.

    local.batched_get_layer_page_prefix = get
    local.batched_submit_layer_pages = publish
    obj._republish_layerwise_cpu_chunk("r", key, [])
    assert page.refs == 2
    with pytest.raises(TimeoutError):
        obj.wait_for_pending_sync_stores()
    if failed:
        remote_future.set_exception(ValueError("failed native put"))
        with pytest.raises(ValueError, match="failed native put"):
            obj.wait_for_pending_sync_stores()
    else:
        remote_future.set_result(None)
        obj.wait_for_pending_sync_stores()
    assert page.refs == 1


def test_recovery_fences_reused_prefix_not_unrelated_puts(monkeypatch):
    obj, storage, plan, _ = setup(monkeypatch)
    prefix, unrelated = Future(), Future()
    queue = obj._layerwise_put_queue = Queue(100, 8, 0)
    queue.add(10, [prefix], req_id="old", keys=plan[0])
    queue.add(10, [unrelated], req_id="other", keys=("unrelated",))

    def complete(futures, timeout):
        assert set(futures) == {prefix}
        assert not storage.probes  # No persistence probe before the source put.
        prefix.set_result(None)
        return {prefix}, set()

    monkeypatch.setattr(queue_module, "wait", complete)
    finish(obj)
    assert_receipt(obj)
    assert not unrelated.done()
    assert not storage.puts


@pytest.mark.parametrize("failed", [False, True])
def test_reused_prefix_pending_or_failed_put_cannot_handoff(monkeypatch, failed):
    obj, storage, plan, _ = setup(monkeypatch)
    prefix = Future()
    queue = obj._layerwise_put_queue = Queue(100, 8, 0)
    queue.add(10, [prefix], req_id="old", keys=plan[0])
    if failed:
        prefix.set_exception(ValueError("old prefix put failed"))
    with pytest.raises(ValueError if failed else TimeoutError):
        finish(obj)
    assert not storage.probes
    assert obj.drain_remote_fill_terminal_results() == {}


def test_short_plan_does_not_claim_required_tail(monkeypatch):
    obj, storage, plan, _ = setup(monkeypatch)
    plan[0].pop()
    with pytest.raises(RuntimeError, match="Cannot recover complete"):
        finish(obj)
    assert not storage.probes
    assert obj.drain_remote_fill_terminal_results() == {}
