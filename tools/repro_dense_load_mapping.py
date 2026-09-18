#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise D-side dense H2D slot-map ordering with the real Ascend kernel.

Run on an IDLE NPU: python -u tools/repro_dense_load_mapping.py --device 0
No model, distributed process group, Mooncake, or shared-cache service is used.
Each case has its own process because an NPU fault can invalidate its context.
Only the reuse control patches the mapping helper, inside that child process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any


PREFIX = "[DENSE_MAPPING_REPRO]"
CASES = ("reuse_multi", "current_single", "current_multi")


def log(message: str) -> None:
    print(f"{PREFIX} {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical, visible NPU id")
    parser.add_argument("--layers", type=int, default=22)
    parser.add_argument("--tokens", type=int, default=1116, help="Fault shape: 1024+92")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=10, help="Per phase, per case")
    parser.add_argument(
        "--delay-matmuls",
        type=int,
        default=32,
        help="Producer-stream matmuls per layer; 0 disables stress",
    )
    parser.add_argument("--timeout", type=float, default=300, help="Seconds per child")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--child", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device < 0 or args.layers < 2 or args.chunk_size < 1:
        parser.error("device must be >=0, layers >=2, chunk-size >=1")
    if args.tokens <= args.chunk_size:
        parser.error("tokens must exceed chunk-size to exercise multiple chunks")
    if args.repeats < 1 or args.delay_matmuls < 0 or args.timeout <= 0:
        parser.error("repeats/timeout must be positive; delay-matmuls must be >=0")
    if args.child and args.run_dir is None:
        parser.error("internal child requires run-dir")
    return args


class MappingProbe:
    """Count materializations without retaining temporaries in the current case."""

    def __init__(self, implementation: Any, reuse: bool) -> None:
        self.implementation = implementation
        self.reuse = reuse
        self.reset()

    def reset(self) -> None:
        self.saved: Any = None
        self.calls = 0
        self.setup_copies = 0
        self.layer_copies = 0
        self.reused = 0

    def __call__(
        self, cache: Any, mapping: Any, starts: Any, ends: Any, base: int = 0
    ) -> Any:
        setup = self.calls == 0
        self.calls += 1
        if self.reuse and self.saved is not None:
            self.reused += 1
            # The exact setup tensor stays owned until the generator completes.
            return self.saved[0], self.saved[1], False
        result = self.implementation(cache, mapping, starts, ends, base)
        full = result[1]
        copied = (
            full.untyped_storage().data_ptr() != mapping.untyped_storage().data_ptr()
        )
        if copied:
            if setup:
                self.setup_copies += 1
            else:
                self.layer_copies += 1
        if self.reuse:
            self.saved = result
        return result

    def report(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in ("calls", "setup_copies", "layer_copies", "reused")
        }


def outcome(reports: dict[str, dict]) -> str:
    """Never turn an unclassified device/setup error into a proven repro."""
    if any(reports.get(case, {}).get("status") != "PASS" for case in CASES[:2]):
        return "INCONCLUSIVE: control failed; inspect its log first"
    status = reports.get("current_multi", {}).get("status")
    if status == "MISMATCH":
        return "REPRODUCED_DATA_MISMATCH: controls passed; current multi-chunk failed"
    if status == "PASS":
        return "NOT_REPRODUCED: this run does not rule out a stream/lifetime race"
    return "CURRENT_MULTI_ERROR: controls passed; inspect traceback/plog"


def run_npu_case(args: argparse.Namespace, report: dict) -> None:
    # Lazy imports keep --help, the parent and CPU unit tests NPU-independent.
    import torch
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    import lmcache_ascend.c_ops as ascend_c_ops

    # Same registered-host allocator binding as the existing KV microbenchmark.
    sys.modules["lmcache.non_cuda_equivalents"] = ascend_c_ops
    import lmcache.v1.memory_management as mm

    mm.lmc_ops = ascend_c_ops
    from lmcache_ascend.v1.npu_connector import npu_connectors as nc

    if nc._DENSE_DIRECT_LOAD_DISABLE:
        raise RuntimeError("Dense-direct load is disabled in this environment")
    report["connector_file"] = nc.__file__
    report["torch"] = torch.__version__
    report["torch_npu"] = torch_npu.__version__
    report["environment"] = {
        key: os.environ.get(key)
        for key in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_LAUNCH_BLOCKING",
            "TASK_QUEUE_ENABLE",
            "PYTORCH_NPU_ALLOC_CONF",
        )
    }
    log(f"case={args.child} connector={nc.__file__}; env={report['environment']}")

    count = args.chunk_size if args.child == "current_single" else args.tokens
    starts = list(range(0, count, args.chunk_size))
    ends = [min(start + args.chunk_size, count) for start in starts]
    width, block_size = 128, 128
    blocks = (count + block_size - 1) // block_size + 2
    capacity = blocks * block_size
    device = torch.device(f"npu:{args.device}")
    connector = nc.VLLMPagedMemLayerwiseNPUConnector(
        hidden_dim_size=width,
        num_layers=args.layers,
        use_gpu=True,
        chunk_size=args.chunk_size,
        dtype=torch.bfloat16,
        device=device,
        use_mla=True,
        dsa_two_groups=True,
    )
    destinations = [
        (
            torch.empty(
                (blocks, block_size, 1, width), dtype=torch.bfloat16, device=device
            ),
        )
        for _ in range(args.layers)
    ]
    # Retain ALL registered CPU sources until every transfer has completed.
    pool_bytes = max(64 << 20, args.layers * (count * width * 2 + len(starts) * 4096))
    allocator = mm.PinMemoryAllocator(pool_bytes)
    rows, expected_rows = [], []
    for layer in range(args.layers):
        values = (
            (
                torch.arange(count)[:, None] * 7
                + torch.arange(width)[None, :]
                + layer * 11
            )
            % 97
        ).to(torch.bfloat16)
        expected_rows.append(values)
        objects = []
        for start, end in zip(starts, ends, strict=True):
            obj = allocator.allocate(
                torch.Size([(end - start) * width]),
                torch.bfloat16,
                mm.MemoryFormat.KV_DSA_INDEX_FMT,
            )
            if obj is None or obj.tensor is None:
                raise RuntimeError("Repro registered-host allocation failed")
            obj.metadata.valid_tokens = end - start
            obj.tensor.copy_(values[start:end].reshape(-1))
            objects.append(obj)
        rows.append(objects)

    original_helper = nc._cached_layerwise_slot_mapping
    original_kernel = nc.dense_mla_dsa_batched_direct_kv_transfer_prepared
    probe = MappingProbe(original_helper, reuse=args.child == "reuse_multi")
    launches: list[dict] = []

    def kernel_probe(*values: Any, **kwargs: Any) -> Any:
        # Store only scalar metadata, NOT tensors: retaining them would hide UAF.
        launches.append(
            {
                "tokens": values[1].numel(),
                "chunks": values[2].numel(),
                "fixed_chunk_size": kwargs.get("fixed_chunk_size"),
            }
        )
        report["last_submission"] = {
            "layer": len(launches) - 1,
            **launches[-1],
            **probe.report(),
        }
        return original_kernel(*values, **kwargs)

    nc._cached_layerwise_slot_mapping = probe
    nc.dense_mla_dsa_batched_direct_kv_transfer_prepared = kernel_probe
    work_a = torch.ones((512, 512), dtype=torch.bfloat16, device=device)
    work_b = torch.ones_like(work_a)
    work_out = torch.empty_like(work_a)
    torch.mm(work_a, work_b, out=work_out)  # Warm up the stress op before measuring.
    torch.npu.synchronize()
    report.update(
        tokens=count,
        chunks=[end - start for start, end in zip(starts, ends, strict=True)],
        iterations=[],
        stage="transfer",
    )
    log(
        f"case={args.child} setup complete; layers={args.layers} "
        f"chunks={report['chunks']}"
    )

    # First run without stress. Then delay only the producer stream and exercise
    # ordinary allocator reuse; never write into a live mapping or use bad slots.
    phases = ["natural"] + (["stress"] if args.delay_matmuls else [])
    for phase in phases:
        for iteration in range(args.repeats):
            report["active_iteration"] = {"phase": phase, "iteration": iteration}
            probe.reset()
            launches.clear()
            slots_cpu = torch.arange(count - 1, -1, -1).roll(iteration)
            slots = slots_cpu.to(device)
            for (destination,) in destinations:
                destination.fill_(-123)
            torch.npu.synchronize()
            readiness: list = []
            generator = connector.batched_to_gpu(
                starts,
                ends,
                slot_mapping=slots,
                sync=True,
                kv_group=1,
                kvcaches=destinations,
                _dense_load_readiness_out=readiness,
                cached_chunk_ptrs_npu=[],
                cached_chunk_dev_ptrs=[],
            )
            next(generator)
            # Setup is complete in both variants; only per-layer work can race.
            torch.npu.synchronize()
            for objects in rows:
                if phase == "stress":
                    for _ in range(args.delay_matmuls):
                        torch.mm(work_a, work_b, out=work_out)
                generator.send(objects)
                if phase == "stress":
                    scratch = torch.empty_like(slots)
                    scratch.fill_(0)  # Valid slot; no deliberately out-of-range data.
                    del scratch
            # Production cold-load contract: wait at the terminal readiness event,
            # NOT after every layer. No test hook adds cross-stream ordering.
            for _ in generator:
                pass
            if len(readiness) != 1 or len(launches) != args.layers:
                raise RuntimeError(
                    f"Wrong test path: readiness={len(readiness)}, "
                    f"native_launches={len(launches)}"
                )
            connector.synchronize_dense_load_readiness(readiness[0])
            torch.npu.synchronize()
            differences = []
            for layer, (destination,) in enumerate(destinations):
                expected = torch.full((capacity, width), -123, dtype=torch.bfloat16)
                expected[slots_cpu] = expected_rows[layer]
                actual = destination.cpu().reshape(capacity, width)
                mismatches = int((actual != expected).sum())
                if mismatches:
                    differences.append(
                        {
                            "layer": layer,
                            "mismatches": mismatches,
                            "max_abs_diff": float(
                                (actual.float() - expected.float()).abs().max()
                            ),
                        }
                    )
            entry = {
                "phase": phase,
                "iteration": iteration,
                **probe.report(),
                "native_launches": len(launches),
                "kernel_shape": launches[0],
                "differences": differences,
            }
            report["iterations"].append(entry)
            log(f"case={args.child} {json.dumps(entry)}")
            if differences:
                report["status"] = "MISMATCH"
                break
        if report.get("status") == "MISMATCH":
            break

    # Only unregister sources after successful stream completion. On device
    # exceptions the isolated child exits instead of attempting further NPU work.
    nc._cached_layerwise_slot_mapping = original_helper
    nc.dense_mla_dsa_batched_direct_kv_transfer_prepared = original_kernel
    for objects in rows:
        for obj in objects:
            obj.ref_count_down()
    allocator.close()
    report.setdefault("status", "PASS")


def run_child(args: argparse.Namespace) -> int:
    report: dict = {"case": args.child, "stage": "setup"}
    try:
        run_npu_case(args, report)
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    (args.run_dir / f"{args.child}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log(f"case={args.child} status={report['status']}")
    return 0 if report["status"] == "PASS" else 1


def run_parent(args: argparse.Namespace) -> int:
    root = args.run_dir or Path(tempfile.mkdtemp(prefix="dense-mapping-", dir="."))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log(f"results={root}; use an idle NPU; no model/Mooncake/TP is started")
    reports = {}
    for case in CASES:
        path = root / f"{case}.log"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--child",
            case,
            "--run-dir",
            str(root),
        ]
        for name in (
            "device",
            "layers",
            "tokens",
            "chunk_size",
            "repeats",
            "delay_matmuls",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        # Avoid reading stale reports when a caller reuses a run directory.
        result_path = root / f"{case}.json"
        if result_path.exists():
            raise FileExistsError(
                f"Use a fresh --run-dir; report exists: {result_path}"
            )
        log(f"starting {case}; log={path}")
        with path.open("w", encoding="utf-8") as output:
            try:
                completed = subprocess.run(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {
                "status": "ERROR",
                "error": "child timed out"
                if exit_code is None
                else "child exited without report",
            }
        )
        result["exit_code"] = exit_code
        if exit_code != 0 and result["status"] == "PASS":
            result.update(
                status="ERROR", error="Child failed after writing PASS report"
            )
        reports[case] = result
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(PREFIX):
                print(line, flush=True)
        if result["status"] != "PASS":
            log(f"{case}: {result['status']}; inspect {path}")
            if result.get("error"):
                log(str(result["error"]))
            if case != "current_multi":
                break  # A broken control is not evidence for the suspected race.
    verdict = outcome(reports)
    (root / "summary.json").write_text(
        json.dumps({"verdict": verdict, "cases": reports}, indent=2), encoding="utf-8"
    )
    log(f"{verdict}; summary={root / 'summary.json'}")
    return (
        0 if all(reports.get(case, {}).get("status") == "PASS" for case in CASES) else 1
    )


if __name__ == "__main__":
    arguments = parse_args()
    sys.exit(run_child(arguments) if arguments.child else run_parent(arguments))
