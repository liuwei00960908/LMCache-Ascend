#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NPU-only bitwise copy check, without model/Mooncake/distributed startup.

python -u tools/check_prefill_split_load.py --probe-only
python -u tools/check_prefill_split_load.py --device 0 --tokens 131614
Uses registered CPU memory, D2H then H2D, both layouts, ragged chunks and two banks.
This checks the native copy and event join, NOT allReduce overlap/performance.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=131614)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("tokens must be positive")
    import torch
    import torch_npu  # noqa: F401
    import lmcache_ascend.c_ops as ops

    torch.npu.set_device(args.device)
    queue = ops.PrefillLoadQueue()  # Fail BEFORE allocating buffers if unsupported.
    print(
        f"[PREFILL_SPLIT_LOAD] priority={queue.priority} "
        f"readback={queue.priority_verified}",
        flush=True,
    )
    if args.probe_only:
        return
    sys.modules["lmcache.non_cuda_equivalents"] = ops
    import lmcache.v1.memory_management as mm

    mm.lmc_ops = ops

    device = f"npu:{args.device}"
    stream = torch.npu.Stream()
    # Include an unaligned fragment boundary INSIDE a CPU chunk, plus short tail.
    for count in sorted({1, 4096, 16383, 16384, 16385, 32769, args.tokens}):
        for kdim, vdim, fmt in ((128, 0, 6), (512, 64, 5)):
            for interleaved in (False, True):
                width, block = kdim + vdim, 128
                blocks = (count + block - 1) // block + 1
                capacity = blocks * block
                sizes = [min(1009, count - start) for start in range(0, count, 1009)]
                offsets, cursor = [], 0
                for size in sizes:
                    offsets.append(cursor)
                    cursor += size
                allocator = mm.PinMemoryAllocator(
                    max(16 << 20, count * width * 2 + len(sizes) * 8192)
                )
                objects = []
                cpu = (
                    (torch.arange(count)[:, None] * 7 + torch.arange(width)[None, :])
                    % 97
                ).bfloat16()
                memory_format = (
                    mm.MemoryFormat.KV_DSA_INDEX_FMT
                    if vdim == 0
                    else mm.MemoryFormat.KV_MLA_FMT
                )
                for start, size in zip(offsets, sizes, strict=True):
                    obj = allocator.allocate(
                        torch.Size([size * width]), torch.bfloat16, memory_format
                    )
                    if obj is None or obj.tensor is None:
                        raise RuntimeError("Registered CPU source allocation failed")
                    obj.metadata.valid_tokens = size
                    data = cpu[start : start + size]
                    packed = (
                        data.reshape(-1)
                        if interleaved or vdim == 0
                        else torch.cat(
                            (data[:, :kdim].reshape(-1), data[:, kdim:].reshape(-1))
                        )
                    )
                    obj.tensor.copy_(packed)
                    objects.append(obj)
                pointers = torch.tensor(
                    [
                        ops.get_device_ptr(o.tensor.data_ptr(), o.tensor.numel() * 2)
                        for o in objects
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                starts_npu = torch.tensor(offsets, dtype=torch.int32, device=device)
                sizes_npu = torch.tensor(sizes, dtype=torch.int32, device=device)
                banks = [
                    tuple(
                        torch.empty(
                            (blocks, block, 1, dim), dtype=torch.bfloat16, device=device
                        )
                        for dim in (kdim, vdim)
                        if dim
                    )
                    for _ in range(2)
                ]
                baseline = tuple(torch.empty_like(part) for part in banks[0])
                for slot_dtype in (torch.int32, torch.int64):
                    slot_ref = torch.empty(0, dtype=slot_dtype, device=device)
                    states = [
                        ops.prepare_sparse_direct_destination_state(
                            b, slot_ref, fmt, kdim, vdim, 0
                        )
                        for b in banks
                    ]
                    base_state = ops.prepare_sparse_direct_destination_state(
                        baseline, slot_ref, fmt, kdim, vdim, 0
                    )
                    store_state = ops.prepare_sparse_direct_layer_state(
                        objects[0].tensor,
                        baseline,
                        slot_ref,
                        interleaved,
                        False,
                        fmt,
                        kdim,
                        vdim,
                        0,
                        count,
                    )
                    # Source objects were changed by the preceding dtype case.
                    # Its terminal synchronize above makes the CPU read safe.
                    current_values = torch.cat(
                        [
                            o.tensor.reshape(size, width)
                            if interleaved or vdim == 0
                            else torch.cat(
                                (
                                    o.tensor[: size * kdim].reshape(size, kdim),
                                    o.tensor[size * kdim :].reshape(size, vdim),
                                ),
                                dim=1,
                            )
                            for o, size in zip(objects, sizes, strict=True)
                        ]
                    ).clone()
                    snapshots, expected = [], []
                    for iteration in range(4):
                        fixed_chunk_size = 1009 if iteration % 2 == 0 else 0
                        bank = banks[iteration % 2]
                        slots_cpu = (
                            torch.arange(count - 1, -1, -1)
                            .roll(iteration)
                            .to(slot_dtype)
                        )
                        if count > 2:
                            slots_cpu[count // 2] = -1
                        # Metadata copy -> short kernels -> readiness -> consumer,
                        # with no host synchronize between layers/bank reuse.
                        slots = slots_cpu.to(device)
                        valid_slots = slots_cpu[slots_cpu >= 0].long().to(device)
                        for part in (*bank, *baseline):
                            part.fill_(-123)
                        stream.wait_stream(torch.npu.current_stream())
                        with torch.npu.stream(stream):
                            ops.dense_mla_dsa_batched_direct_kv_transfer_prepared(
                                base_state,
                                slots,
                                pointers,
                                starts_npu,
                                sizes_npu,
                                count,
                                interleaved,
                                True,
                                fixed_chunk_size,
                            )
                            # Emulate newly computed KV, then save and reload on
                            # the SAME dispatch/FIFO without a CPU wait between.
                            for part in baseline:
                                part.view(capacity, -1)[valid_slots] += 1
                            valid_slots.record_stream(stream)
                            queue.store_prepared(
                                store_state,
                                slots,
                                pointers,
                                starts_npu,
                                sizes_npu,
                                count,
                                interleaved,
                                True,
                                True,
                                fixed_chunk_size,
                            )
                            queue.transfer_prepared(
                                states[iteration % 2],
                                slots,
                                pointers,
                                starts_npu,
                                sizes_npu,
                                count,
                                interleaved,
                                True,
                                fixed_chunk_size,
                            )
                            slots.record_stream(stream)
                            ready = torch.npu.Event()
                            ready.record(stream)
                        torch.npu.current_stream().wait_event(ready)
                        snapshots.append(
                            (
                                tuple(x.clone() for x in bank),
                                tuple(x.clone() for x in baseline),
                            )
                        )
                        exp = torch.full((capacity, width), -123, dtype=torch.bfloat16)
                        valid = slots_cpu >= 0
                        current_values[valid] += 1
                        exp[slots_cpu[valid].long()] = current_values[valid]
                        expected.append(exp)
                    torch.npu.synchronize()
                    for obj, start, size in zip(objects, offsets, sizes, strict=True):
                        data = current_values[start : start + size]
                        packed = (
                            data.reshape(-1)
                            if interleaved or vdim == 0
                            else torch.cat(
                                (data[:, :kdim].reshape(-1), data[:, kdim:].reshape(-1))
                            )
                        )
                        if not torch.equal(obj.tensor, packed):
                            raise RuntimeError(
                                f"D2H_MISMATCH tokens={count} width={width}"
                            )
                    for (actual_parts, base_parts), exp in zip(
                        snapshots, expected, strict=True
                    ):
                        actual = torch.cat(
                            [x.cpu().reshape(capacity, -1) for x in actual_parts], dim=1
                        )
                        reference = torch.cat(
                            [x.cpu().reshape(capacity, -1) for x in base_parts], dim=1
                        )
                        if not torch.equal(actual, exp) or not torch.equal(
                            actual, reference
                        ):
                            raise RuntimeError(
                                f"COPY_MISMATCH tokens={count} width={width} "
                                f"interleaved={interleaved}"
                            )
                for obj in objects:
                    obj.ref_count_down()
                allocator.close()
                print(
                    f"[PREFILL_SPLIT_LOAD] PASS tokens={count} width={width} "
                    f"interleaved={interleaved}",
                    flush=True,
                )
    print("[PREFILL_SPLIT_LOAD] ALL_COPY_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
