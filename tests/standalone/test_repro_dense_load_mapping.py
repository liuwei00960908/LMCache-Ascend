# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the NPU repro's controls, using the production mapping helpers."""

import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/repro_dense_load_mapping.py"
SPEC = importlib.util.spec_from_file_location("dense_mapping_repro", SCRIPT)
repro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repro)


@pytest.fixture
def mapping_helper():
    path = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_slice_layerwise_slot_mapping", "_cached_layerwise_slot_mapping"}
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["_cached_layerwise_slot_mapping"]


@pytest.mark.parametrize("reuse,layers_copied", [(False, 22), (True, 0)])
def test_multi_chunk_probe_preserves_values_and_counts(
    mapping_helper, reuse, layers_copied
):
    probe = repro.MappingProbe(mapping_helper, reuse=reuse)
    mapping = torch.arange(1115, -1, -1)
    setup = probe(None, mapping, [0, 1024], [1024, 1116])
    for _ in range(22):
        chunks, full, _ = probe(None, mapping, [0, 1024], [1024, 1116])
        assert [len(chunk) for chunk in chunks] == [1024, 92]
        assert torch.equal(full, mapping)
        assert (full is setup[1]) == reuse
    assert probe.report() == dict(
        calls=23, setup_copies=1, layer_copies=layers_copied, reused=22 if reuse else 0
    )


def test_current_probe_does_not_keep_mapping_alive(mapping_helper):
    probe = repro.MappingProbe(mapping_helper, reuse=False)
    result = probe(None, torch.arange(1116), [0, 1024], [1024, 1116])
    reference = weakref.ref(result[1])
    del result
    assert reference() is None  # An accidental extra owner would hide the real bug.


def test_reuse_owner_is_released_between_transfers(mapping_helper):
    probe = repro.MappingProbe(mapping_helper, reuse=True)
    result = probe(None, torch.arange(1116), [0, 1024], [1024, 1116])
    reference = weakref.ref(result[1])
    del result
    assert reference() is not None
    probe.reset()
    assert reference() is None


def test_single_chunk_does_not_allocate_full_mapping(mapping_helper):
    probe = repro.MappingProbe(mapping_helper, reuse=False)
    mapping = torch.arange(1024)
    for _ in range(23):
        probe(None, mapping, [0], [1024])
    assert probe.report() == dict(calls=23, setup_copies=0, layer_copies=0, reused=0)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("copy_to_device", [False, True])
@pytest.mark.parametrize(
    "ranges", [([0], [1024]), ([0, 1024], [1024, 1116]), ([0, 1100], [1024, 1192])]
)
@pytest.mark.parametrize("banked", [False, True])
def test_actual_deferred_generator_uses_probe_without_changing_protocol(
    mapping_helper, monkeypatch, reuse, copy_to_device, ranges, banked
):
    """Drive the production generator on CPU; only its NPU operations are stubbed."""
    path = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_resolve_layerwise_slot_mapping",
            "_layer_memory_tensor",
            "_layer_source_memory_objs",
        }
    ]
    connector_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    nodes.append(
        next(
            node
            for node in connector_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "batched_to_gpu"
        )
    )
    # The fixed-map test control must not override a P-node bank switch.
    probe = repro.MappingProbe(mapping_helper, reuse and not banked)
    scope = dict(
        torch=torch,
        _cached_layerwise_slot_mapping=probe,
        _DENSE_DIRECT_LOAD_DISABLE=False,
        LayerPageSource=type("LayerPageSource", (), {}),
        LayerPageMemoryObj=type("LayerPageMemoryObj", (), {}),
        logger=SimpleNamespace(isEnabledFor=lambda _: False),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    waits, synchronizations, validated = [], [], []
    stream = SimpleNamespace(
        wait_stream=lambda other: waits.append(other),
        synchronize=lambda: synchronizations.append(True),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=lambda: SimpleNamespace(record=lambda _: None)),
        raising=False,
    )
    layout = SimpleNamespace(
        kv_format=SimpleNamespace(value=6),
        vllm_two_major=False,
        k_hidden_dims=128,
        v_hidden_dims=0,
        dsa_hidden_dims=128,
        kv_device=torch.device("cpu"),
    )
    starts, ends = ranges
    mapping = torch.arange(max(ends) - 1, -1, -1)
    bank_mappings = [mapping, mapping + 2048]
    submitted, prepared_mappings, used_mappings = [], [], []

    def prepare_mapping(slots, _):
        # Simulate CPU -> NPU conversion returning a distinct tensor owner.
        prepared = slots.clone() if copy_to_device else slots
        prepared_mappings.append(prepared)
        return prepared

    def transfer(**kwargs):
        assert kwargs["direction"] is False
        assert kwargs["current_stream"] is stream
        assert kwargs["transfer_stream"] is stream
        selected = bank_mappings[kwargs["layer_id"] % 2] if banked else mapping
        expected = torch.cat(
            [selected[start:end] for start, end in zip(starts, ends, strict=True)]
        )
        assert torch.equal(kwargs["slot_mapping_full"], expected)
        if not banked:
            assert kwargs["slot_mapping_full"] is prepared_mappings[0]
        used_mappings.append(kwargs["slot_mapping_full"])
        submitted.append(kwargs["layer_id"])

    connector = SimpleNamespace(
        kvcaches=None,
        load_stream=stream,
        use_gpu=True,
        initialize_kvcaches_ptr=lambda **kw: None,
        _lazy_initialize_buffer_with_staging=lambda *a, **kw: layout,
        _is_mla_dsa_format=lambda _: True,
        _check_layerwise_transfer_invariants=lambda **kw: validated.append(kw),
        _layerwise_token_major=lambda _: True,
        _expected_memory_format=lambda _: "index",
        _sparse_lmc_host_interleaved=lambda _: True,
        _prepare_dense_direct_chunk_metadata=lambda *a, **kw: (
            1024,
            torch.tensor([0, 1024]),
            torch.tensor([1024, 92]),
        ),
        _slot_mapping_on_kv_device=prepare_mapping,
        _get_or_create_sparse_destination_plan=lambda **kw: object(),
        _expected_group_layers=lambda _: 3,
        _resolve_sparse_chunk_ptrs_npu=lambda *a, **kw: torch.zeros(
            2, dtype=torch.long
        ),
        _run_dense_direct_kv_transfer_layer=transfer,
        record_dense_load_readiness=lambda: "ready",
        _set_layerwise_prefill_bank_count=lambda *a: None,
        _layerwise_prefill_transfer_generation=lambda _: 1,
        _check_layerwise_prefill_transfer_generation=lambda *a: None,
        _layerwise_prefill_bank=lambda layer, group: layer % 2,
        _layerwise_prefill_transfer_state=lambda: ({}, {}, {}, {}),
        _stream_context_or_null=lambda _: nullcontext(),
    )
    readiness = []
    generator = scope["batched_to_gpu"](
        connector,
        starts,
        ends,
        slot_mapping=mapping,
        sync=True,
        kv_group=1,
        kvcaches=[object()] * 3,
        deferred_layerwise_get=banked,
        **({} if banked else {"_dense_load_readiness_out": readiness}),
        cached_chunk_ptrs_npu=[],
        cached_chunk_dev_ptrs=[],
    )
    next(generator)
    for layer in range(3):
        objects = [
            SimpleNamespace(
                metadata=SimpleNamespace(fmt="index"), tensor=torch.zeros(end - start)
            )
            for start, end in zip(starts, ends, strict=True)
        ]
        generator.send(
            {
                "memory_objs": objects,
                "layer_request": {"slot_mapping": bank_mappings[layer % 2]},
            }
            if banked
            else objects
        )
    for _ in generator:
        pass
    assert submitted == [0, 1, 2]
    if banked:
        assert readiness == []
        assert probe.calls == 4
        assert used_mappings[0] is used_mappings[2]
        assert used_mappings[0] is not used_mappings[1]
        assert synchronizations == [True]  # Existing terminal P-side fence only.
    else:
        assert readiness == ["ready"]
        assert probe.calls == 1
        assert probe.layer_copies == 0
        assert len(validated) == 1
        assert waits == [stream]  # Existing setup dependency; no per-layer wait.
        assert synchronizations == []


@pytest.mark.parametrize(
    "statuses,prefix",
    [
        (("PASS", "PASS", "PASS"), "NOT_REPRODUCED"),
        (("PASS", "PASS", "MISMATCH"), "REPRODUCED_DATA_MISMATCH"),
        (("PASS", "PASS", "ERROR"), "CURRENT_MULTI_ERROR"),
        (("MISMATCH", "PASS", "MISMATCH"), "INCONCLUSIVE"),
        (("ERROR", "PASS", "ERROR"), "INCONCLUSIVE"),
    ],
)
def test_verdict_requires_passing_controls(statuses, prefix):
    reports = {
        name: {"status": status}
        for name, status in zip(repro.CASES, statuses, strict=True)
    }
    assert repro.outcome(reports).startswith(prefix)


def test_defaults_match_reported_kernel_shape():
    args = repro.parse_args([])
    assert (args.tokens, args.chunk_size, args.layers) == (1116, 1024, 22)


def test_child_setup_error_is_reported_without_npu(monkeypatch, tmp_path):
    def fail(*args):
        raise ImportError("No NPU here")

    monkeypatch.setattr(repro, "run_npu_case", fail)
    args = repro.parse_args(["--child", "reuse_multi", "--run-dir", str(tmp_path)])
    assert repro.run_child(args) == 1
    report = json.loads((tmp_path / "reuse_multi.json").read_text())
    assert report["stage"] == "setup"
    assert report["status"] == "ERROR"
    assert "No NPU here" in report["error"]


def test_parent_does_not_run_more_cases_after_control_failure(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(repro.subprocess, "run", fake_run)
    args = repro.parse_args(["--run-dir", str(tmp_path)])
    assert repro.run_parent(args) == 1
    assert len(commands) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["verdict"].startswith("INCONCLUSIVE")
