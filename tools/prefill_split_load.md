# Experimental short-kernel prefill load

On `lmy_merge_prefill_layerwise_cache`, **off by default**:

```bash
export LMCACHE_ASCEND_PREFILL_SPLIT_LOAD=1
```

Read once when constructing the connector and SFA implementation. Only P-node
deferred layerwise dense MLA/DSA stores and loads use this switch. Unset it or
set it to `0` and restart to restore the existing path. Ordinary dense loads,
D-node sparse loads and Mooncake publication are unchanged.
The profile script enables this experiment by default, but preserves an explicit
`LMCACHE_ASCEND_PREFILL_SPLIT_LOAD=0` override through its environment filter.
That tool-only default does not change the production connector's default.

## Build and check

Rebuild **LMCache-Ascend**, not just its Python files:

```bash
cd /workspace/lmy/LMCache-Ascend
pip install --no-build-isolation -e .
python -u tools/check_prefill_split_load.py --probe-only
python -u tools/check_prefill_split_load.py --device 0 --tokens 131614
```

The probe creates only a stream and events; it does not load a model or allocate
a KV slab. The copy check uses registered CPU sources, both latent/index groups,
both CPU layouts, ragged CPU chunks, 16k boundaries, negative slots, int32/int64
slot maps, two destination banks and repeated event reuse. It compares to both
the original kernel and an independent CPU scatter. The queue saves modified
NPU KV to registered CPU memory before loading it back, without a host wait
between those operations; both CPU destinations and NPU reloads are checked.
Copies must be bitwise
equal (there is no floating-point arithmetic in this operation).

This is NOT a guarantee that a given NPU/CANN supports priorities. For example,
[CANN 8.5's stream API documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/appdevgapi/aclcppdevg_03_0069.html)
lists the priority argument on A2/A3 as reserved at 0. The experiment queries
the supported priority range once and refuses a range with no lower priority
than the normal compute stream (0). Missing APIs, failed stream creation, and
failed priority readback are errors, not silent ordinary-stream fallbacks.
On older runtimes without `aclrtStreamGetPriority`, the log explicitly says
`priority_readback=False`; the creation API and range query are still used.

## Scheduling and costs

After SFA releases its bank, submit the current-layer D2H save followed by
the next-layer H2D load, before `v_up_proj` and `o_proj`. Both directions use
ONE torch dispatch stream and ONE native low-priority FIFO. The original
pre-allReduce callback remains in effect when disabled. D2H retains the original
single kernel (normally only the new 4096-token compute chunk).

Each H2D layer submits one queued native command. That command launches
`ceil(load_tokens / 16384)` kernels into ONE CANN low-priority FIFO stream.
130000 tokens produce 8 kernels, 131614 produce 9. No per-slice pointer/offset
tables, tensor slices, Python dispatch, CPU completion queries, TP collectives,
or content-validation scans are added. Global token offsets and the original
chunk-size/slot metadata are passed unchanged to the existing MLA/DSA processor.

Incremental costs, explicitly:

| Item | Cost |
| --- | --- |
| Disabled | One environment read during connector construction; cached boolean/None dispatch in P preparation; no extra native resources or launches |
| Extension import | The binary includes additional kernel variants; CANN may register their code even with the switch off (no low-priority stream/events are created) |
| First enabled transfer (including first-chunk store) | One priority-range query, one stream, two reusable Ex-events, optional priority readback, one INFO log per connector |
| Per store or load | Two record + two device stream-wait submissions, one captured shared queue reference; D2H adds no kernel launches |
| Per chunk/group store preparation | One device-side wait for metadata from the original store stream, no CPU wait |
| Per fragment beyond the first | One kernel launch and its kernel/UB initialization; no extra transferred KV bytes |
| Source/metadata lifetime | Existing load-stream fences remain; no second KV buffer |
| Normal completion | Existing next-layer and terminal source-release waits; no per-fragment host wait |
| Error/teardown only | Drain outstanding work before releasing referenced memory/events |

The ready event follows the existing metadata/previous-bank-save dependencies.
The low-priority stream waits on it, executes all fragments, then records done.
The common torch dispatch stream joins done before its per-layer store/load fence.
The compute stream waits on that fence at the existing next-layer consumer site.
Thus an allocator/source-release wait on the original stream still covers every
fragment. Compute never waits for the whole FIFO (which may already contain a
future transfer waiting for compute); it waits for the corresponding layer's
completion event only. Chunk-end CPU publication and bank-reuse fences remain.
Ex-event generation semantics permit reuse without per-layer creation
or reset ([record-event documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/API/appdevgapi/aclcppdevg_03_0083.html)).

Priority is NOT preemption and NOT an idle-only promise. A running fragment can
still delay a newly ready compute kernel until it finishes. Short fragments give
the scheduler more boundaries to choose compute work; actual overlap and total
latency must be measured. SFA's dependency on its own loaded KV is unchanged.

## Profile

Compare **layerwise ON in both runs**, changing only this switch:

```bash
cd /workspace/lmy/vllm-ascend
LMCACHE_ASCEND_PREFILL_SPLIT_LOAD=0 python -u tools/layerwise_prefill_profile.py --case 100k_on --cpu-cache-gb 32
LMCACHE_ASCEND_PREFILL_SPLIT_LOAD=1 python -u tools/layerwise_prefill_profile.py --case 100k_on --cpu-cache-gb 32
```

In MindStudio look for `single_layer_paged_kv_copy_prefill_*`, multiple short
kernels on one separate stream, device-side compute/HCCL interleaving, and total
prefill time. A shorter individual kernel does not by itself prove a speedup.
Existing `--include-off` compares the whole layerwise-cache feature, not this
split-load experiment.

## CPU host tests

```bash
g++ -std=c++17 -Itests/native/prefill_stubs -Icsrc -Ithird_party/kvcache-ops \
  csrc/prefill_load_queue.cpp tests/native/test_prefill_load_queue.cpp \
  -o /tmp/test_prefill_load_queue
/tmp/test_prefill_load_queue
python -m pytest tests/standalone/test_prefill_split_load.py --noconftest
```

The native test compiles the real queue with a fake ACL boundary. It covers
16384-token ranges through INT32_MAX, rejected/ignored priority configuration,
event reuse over 101 layers, FIFO record/wait ordering, absence of normal-path
host waits, and exceptional cleanup. It does not compile or run AscendC kernels.
