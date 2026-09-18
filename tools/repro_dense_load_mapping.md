# D-side multi-chunk H2D reproduction

Run from an installed/built `LMCache-Ascend` checkout on an **idle NPU**:

```bash
python -u tools/repro_dense_load_mapping.py --device 0
```

`--device` is the logical id within `ASCEND_RT_VISIBLE_DEVICES`. No model,
TP process group, Mooncake, or shared-cache server is started. Nothing is deleted
from `/dev/shm`. This test uses the installed LMCache/LMCache-Ascend Python and
native code; its log records the actual connector module path.

The default shape follows the reported failing indexer transfer: BF16, width
128, NPU block size 128, 22 layers, 1116 tokens = 1024 + 92. This is the size of
the **transfer**, not the length of a model prompt. The host pool is 64 MiB at
these defaults. Production code is not edited by this test.

Three isolated child processes run in this order:

1. `reuse_multi`: two chunks; a test-only helper override retains the initial
   concatenated slot mapping and reuses that exact tensor across layers.
2. `current_single`: unmodified production helper; one 1024-token chunk.
3. `current_multi`: unmodified production helper; 1024 + 92 tokens.

All call the real `batched_to_gpu` deferred-readiness path and the native
`dense_mla_dsa_batched_direct_kv_transfer_prepared` H2D kernel. Each case first
runs naturally, then with producer-stream matmul work and allocator churn to
increase the chance of exposing missing stream/lifetime ordering. No live
mapping is intentionally overwritten, and no out-of-range slot is supplied.
There is no synchronization between layer submissions; setup and final-result
inspection are synchronized. CPU sources stay alive throughout the transfer.

The entire destination, including untouched sentinel rows, is compared with
known CPU data after each transfer. Exact equality is appropriate here: this
is a data copy, not model floating-point computation. A mismatch is printed per
layer with its count and maximum absolute difference.

The script prints its result directory, per-case log paths, and final verdict.
`summary.json` contains all case reports. `setup_copies` / `layer_copies` count
materialized full mappings; `reused` counts reuse-control hits. On the current
code, the two-chunk current case makes 1 setup copy and 22 layer copies; the
reuse control makes 1 setup copy and 0 layer copies. Counters do not retain
temporary mapping tensors, which would otherwise mask a lifetime race.

- `REPRODUCED_DATA_MISMATCH`: controls passed but the current multi-chunk path
  copied incorrect data. This isolates a mapping-related failure in this test.
- `CURRENT_MULTI_ERROR`: controls passed but the current multi-chunk child
  errored. Inspect its traceback and Ascend plog to identify the failing kernel;
  a generic NPU error is not automatically the same production fault.
- `NOT_REPRODUCED`: all transfers passed this run; a timing-dependent race is
  **not** ruled out.
- `INCONCLUSIVE`: a control failed; inspect that log before attributing a cause.

For a longer run: `--repeats 100 --timeout 1800`. To omit stress work:
`--delay-matmuls 0`. `ASCEND_LAUNCH_BLOCKING=1` can hide asynchronous races; the
script logs but does not override it. A native fault can invalidate the child
context; do not run this alongside a production service on the same NPU.

This isolates D-side registered-CPU-to-NPU dense indexer transfer. It does not
reproduce multi-rank shared-memory address publication, Mooncake transfers,
model computation, or the complete PD pipeline. Passing it cannot validate
those other paths or establish the root cause of the production crash.
