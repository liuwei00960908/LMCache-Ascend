# SPDX-License-Identifier: Apache-2.0
"""Drive the production store generator without requiring an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("copy_to_device", [False, True])
@pytest.mark.parametrize("layers", [22, 79])
@pytest.mark.parametrize(
    "ranges,base",
    [
        (([0], [1024]), 0),
        (([0, 1024], [1024, 1116]), 0),
        (([2048, 3148], [3072, 3240]), 2048),
    ],
)
def test_ordinary_store_reuses_prepared_mapping(
    monkeypatch, copy_to_device, layers, ranges, base
):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helpers = {
        "_resolve_layerwise_slot_mapping",
        "_slice_layerwise_slot_mapping",
        "_cached_layerwise_slot_mapping",
        "_layer_memory_tensor",
    }
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers
    ]
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    nodes.append(
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "batched_from_gpu"
        )
    )
    scope = dict(
        torch=torch,
        _DENSE_DIRECT_STORE_DISABLE=False,
        LayerPageMemoryObj=type("LayerPageMemoryObj", (), {}),
        logger=NS(debug=lambda *a: None, error=lambda *a: None),
        _mtp_dw_deep_diag_enabled=lambda: False,
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), scope)

    starts, ends = ranges
    mapping = torch.arange(max(ends) - base - 1, -1, -1)
    expected = torch.cat(
        [mapping[s - base : e - base] for s, e in zip(starts, ends, strict=True)]
    )
    prepared, submitted, checked, concatenations, syncs = [], [], [], [], []
    stream = NS(synchronize=lambda: syncs.append(True))
    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda: stream), raising=False)
    real_cat = torch.cat

    def cat(*args, **kwargs):
        concatenations.append(True)
        return real_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", cat)

    def convert(slots, target_stream):
        assert target_stream is stream
        # Conversion has a distinct owner, as with CPU -> NPU, but remains CPU
        # here so the test can check identity and contents without device mocks.
        result = slots.clone() if copy_to_device else slots
        prepared.append(result)
        return result

    def transfer(**kwargs):
        assert kwargs["direction"] is True
        assert kwargs["defer_consumer_wait"] is False
        assert kwargs["slot_mapping_full"] is prepared[0]
        assert torch.equal(kwargs["slot_mapping_full"], expected)
        submitted.append(kwargs["layer_id"])

    layout = NS(
        kv_format=NS(value=6),
        vllm_two_major=False,
        k_hidden_dims=128,
        v_hidden_dims=0,
        dsa_hidden_dims=128,
    )
    connector = NS(
        kvcaches=None,
        use_gpu=True,
        store_stream=stream,
        initialize_kvcaches_ptr=lambda **kw: None,
        _lazy_initialize_buffer_with_staging=lambda *a, **kw: layout,
        _is_mla_dsa_format=lambda _: True,
        _check_layerwise_transfer_invariants=lambda **kw: checked.append(kw),
        _slot_mapping_on_kv_device=convert,
        _layerwise_token_major=lambda _: True,
        _expected_memory_format=lambda _: "index",
        _sparse_lmc_host_interleaved=lambda _: True,
        _prepare_dense_direct_chunk_metadata=lambda *a, **kw: (
            1024,
            torch.tensor(starts),
            torch.tensor(ends),
        ),
        _expected_group_layers=lambda _: layers,
        _resolve_sparse_chunk_ptrs_npu=lambda *a, **kw: torch.zeros(
            len(starts), dtype=torch.long
        ),
        _run_dense_direct_kv_transfer_layer=transfer,
    )
    memory = [
        [
            NS(metadata=NS(fmt="index"), tensor=torch.zeros(e - s, 128))
            for s, e in zip(starts, ends, strict=True)
        ]
        for _ in range(layers)
    ]
    list(
        scope["batched_from_gpu"](
            connector,
            memory,
            starts,
            ends,
            slot_mapping=mapping,
            slot_mapping_base=base,
            sync=True,
            kv_group=1,
            kvcaches=[object()] * layers,
        )
    )
    assert submitted == list(range(layers))
    assert len(prepared) == len(checked) == 1
    assert len(concatenations) == (1 if len(starts) > 1 else 0)
    assert len(syncs) == layers  # Preserve the ordinary store completion contract.
