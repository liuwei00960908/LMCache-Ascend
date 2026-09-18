# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the real connector's dispatch; kernels/streams are mocked."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"


def connector_method(name, scope):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if getattr(n, "name", "") == "VLLMPagedMemLayerwiseNPUConnector"
    )
    fn = next(n for n in cls.body if getattr(n, "name", "") == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias("annotations")], level=0
    )
    unit = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
    exec(compile(unit, str(SOURCE), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("layer", [0, 1, 78])
def test_prepared_load_uses_one_call_and_preserves_full_metadata(enabled, layer):
    calls, recorded, waits = [], [], []
    normal = lambda *a, **kw: calls.append(("normal", a, kw))
    split = lambda *a, **kw: calls.append(("split", a, kw))
    fn = connector_method(
        "_run_dense_direct_kv_transfer_layer",
        {
            "dense_mla_dsa_batched_direct_kv_transfer_prepared": normal,
        },
    )
    owner = NS(
        enable_npu_transfer_validation=True,
        _stream_context_or_null=lambda _: nullcontext(),
    )
    transfer = NS(wait_stream=lambda other: waits.append(("producer", other)))
    compute = NS(wait_stream=lambda other: waits.append(("consumer", other)))

    def tensor(name, size):
        return NS(
            numel=lambda: size, record_stream=lambda s: recorded.append((name, s))
        )

    slots = tensor("slots", 131614)
    pointers = tensor("ptrs", 129)
    offsets = tensor("offsets", 129)
    sizes = tensor("sizes", 129)
    state = object()
    fn(
        owner,
        kvcaches_ref=[],
        kv_group=0,
        layer_id=layer,
        transfer_stream=transfer,
        current_stream=compute,
        slot_mapping_full=slots,
        chunk_ptrs_npu=pointers,
        chunk_offsets_npu=offsets,
        chunk_sizes_npu=sizes,
        total_tokens=131614,
        fixed_chunk_size=1024,
        dense_kv_format=5,
        dense_token_major=True,
        dense_vllm_two_major=False,
        dense_k_hidden_dims=512,
        dense_v_hidden_dims=64,
        dense_dsa_hidden_dims=0,
        dense_host_interleaved=True,
        layer_tensors=[],
        direction=False,
        destination_plan=NS(states={layer: state}),
        defer_consumer_wait=True,
        prefill_load_queue=NS(transfer_prepared=split) if enabled else None,
    )
    assert calls == [
        (
            "split" if enabled else "normal",
            (state, slots, pointers, offsets, sizes, 131614, True),
            dict(validate_inputs=layer == 0, fixed_chunk_size=1024),
        )
    ]
    assert waits == [("producer", compute)]  # no new eager consumer wait
    assert recorded == [("ptrs", transfer)] + (
        [("slots", transfer), ("offsets", transfer), ("sizes", transfer)]
        if layer == 0
        else []
    )


@pytest.mark.parametrize(
    "enabled,deferred", [(False, False), (False, True), (True, False), (True, True)]
)
def test_queue_is_p_only_and_cached(enabled, deferred):
    # Execute the real generator's queue-setup block independently of its NPU
    # allocator. D-side deferred readiness must not opt into the P-only queue.
    scope = {}
    fn = connector_method("batched_to_gpu", scope)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if getattr(n, "name", "") == "VLLMPagedMemLayerwiseNPUConnector"
    )
    node = next(n for n in cls.body if getattr(n, "name", "") == fn.__name__)
    index = next(
        i
        for i, n in enumerate(node.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "prefill_load_queue" for t in n.targets
        )
    )
    setup = compile(
        ast.Module(body=[node.body[index]], type_ignores=[]),
        str(SOURCE),
        "exec",
    )
    created, logs = [], []

    def create():
        queue = NS(priority=7, priority_verified=True)
        created.append(queue)
        return queue

    owner = NS(_prefill_split_load_enabled=enabled, _prefill_split_load_queue=None)
    scope.update(
        self=owner,
        deferred_dense_direct_get=deferred,
        torch=NS(npu=NS(device=lambda _: nullcontext())),
        layout=NS(kv_device="npu:0"),
        lmc_ops=NS(PrefillLoadQueue=create),
        logger=NS(info=lambda *a: logs.append(a)),
    )
    get_queue = connector_method("_get_prefill_transfer_queue", scope)
    owner._get_prefill_transfer_queue = lambda device: get_queue(owner, device)
    for _ in range(3):
        exec(setup, scope)
        assert scope["prefill_load_queue"] is (
            created[0] if enabled and deferred else None
        )
    assert len(created) == len(logs) == int(enabled and deferred)


def test_environment_is_read_only_during_connector_construction():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if getattr(n, "name", "") == "VLLMPagedMemLayerwiseNPUConnector"
    )
    users = [
        n.name
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(c, ast.Constant)
            and c.value == "LMCACHE_ASCEND_PREFILL_SPLIT_LOAD"
            for c in ast.walk(n)
        )
    ]
    assert users == ["__init__"]


@pytest.mark.parametrize("kv_group", [0, 1])
@pytest.mark.parametrize("has_history", [False, True])
def test_consumer_joins_only_its_bank_and_layer_not_the_fifo(kv_group, has_history):
    waits = []
    compute = NS(
        wait_event=waits.append,
        wait_stream=lambda _: pytest.fail("whole FIFO join risks a dependency cycle"),
    )
    save, load, future_load = object(), object(), object()
    saves = {(kv_group, 0): (3, save)}
    loads = {(kv_group, 4): (3, future_load)}
    if has_history:
        loads[(kv_group, 2)] = (3, load)
    fn = connector_method(
        "wait_for_layerwise_prefill_load",
        {"torch": NS(npu=NS(current_stream=lambda: compute))},
    )
    owner = NS(
        _layerwise_prefill_transfer_state=lambda: (
            {kv_group: 2},
            {kv_group: 3},
            saves,
            loads,
        ),
        _layerwise_prefill_bank=lambda layer, group: layer % 2,
    )
    fn(owner, 2, kv_group)
    assert waits == [save] + ([load] if has_history else [])
    assert loads == {(kv_group, 4): (3, future_load)}
