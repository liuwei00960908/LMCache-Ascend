// SPDX-License-Identifier: Apache-2.0
#include "prefill_load_queue.h"
#include <dlfcn.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace {
void check_acl(aclError rc, const char *operation) {
  TORCH_CHECK(rc == ACL_SUCCESS, "[PREFILL_SPLIT_LOAD] ", operation,
              " failed: aclError=", rc);
}
} // namespace

PrefillLoadQueue::PrefillLoadQueue() {
  device_index_ = c10_npu::getCurrentNPUStream().device_index();
  // Resolve version-dependent APIs once when enabled. Old CANN must still be
  // able to import c_ops with the feature disabled.
  using PriorityRange = aclError (*)(int32_t *, int32_t *);
  using CreateEvent = aclError (*)(aclrtEvent *, uint32_t);
  using GetPriority = aclError (*)(aclrtStream, uint32_t *);
  const auto range = reinterpret_cast<PriorityRange>(
      dlsym(RTLD_DEFAULT, "aclrtDeviceGetStreamPriorityRange"));
  const auto create_event = reinterpret_cast<CreateEvent>(
      dlsym(RTLD_DEFAULT, "aclrtCreateEventExWithFlag"));
  TORCH_CHECK(range && create_event,
              "[PREFILL_SPLIT_LOAD] This CANN runtime lacks priority-range or "
              "reusable-event support. Unset LMCACHE_ASCEND_PREFILL_SPLIT_LOAD "
              "to use the original load path.");
  int32_t least = 0, greatest = 0;
  check_acl(range(&least, &greatest), "query stream priority range");
  TORCH_CHECK(least > 0 && least > greatest,
              "[PREFILL_SPLIT_LOAD] Lower priority than the compute stream "
              "(priority=0) is not supported by this device/runtime: least=",
              least, ", greatest=", greatest,
              ". Unset LMCACHE_ASCEND_PREFILL_SPLIT_LOAD; ordinary-priority "
              "execution is NOT a low-priority experiment.");
  priority_ = least;
  try {
    check_acl(aclrtCreateStreamWithConfig(
                  &stream_, priority_, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC),
              "create low-priority stream");
    const auto get_priority = reinterpret_cast<GetPriority>(
        dlsym(RTLD_DEFAULT, "aclrtStreamGetPriority"));
    if (get_priority) {
      uint32_t actual = 0;
      check_acl(get_priority(stream_, &actual), "query created stream priority");
      TORCH_CHECK(actual == static_cast<uint32_t>(priority_),
                  "[PREFILL_SPLIT_LOAD] Stream priority was not applied: "
                  "requested=", priority_, ", actual=", actual);
      priority_verified_ = true;
    }
    check_acl(create_event(&ready_, ACL_EVENT_SYNC), "create ready event");
    check_acl(create_event(&done_, ACL_EVENT_SYNC), "create completion event");
  } catch (...) {
    release();
    throw;
  }
}

PrefillLoadQueue::~PrefillLoadQueue() { release(); }

void PrefillLoadQueue::release() noexcept {
  try {
    const c10_npu::NPUGuard guard(device_index_);
    // Teardown only, never a per-layer/per-fragment host synchronization.
    // Pending OpCommand handlers retain shared ownership of this queue.
    if (stream_) {
      aclrtSynchronizeStream(stream_);
    }
    if (producer_) {
      aclrtSynchronizeStream(producer_);
    }
    if (done_) {
      aclrtDestroyEvent(done_);
      done_ = nullptr;
    }
    if (ready_) {
      aclrtDestroyEvent(ready_);
      ready_ = nullptr;
    }
    if (stream_) {
      aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
  } catch (...) {
    // The process/runtime may already be shutting down; destructors cannot throw.
  }
}

void PrefillLoadQueue::submit(const kvcache_ops::PrefillLoadArgs &args,
                             uint32_t cores, aclrtStream producer) {
  producer_ = producer;
  check_acl(aclrtRecordEvent(ready_, producer), "record producer readiness");
  check_acl(aclrtStreamWaitEvent(stream_, ready_), "wait for metadata/bank");
  try {
    if (args.store) {
      kvcache_ops::launch_prefill_store(args, cores, stream_);
    } else {
      kvcache_ops::launch_prefill_load(args, cores, stream_);
    }
  } catch (...) {
    // Exceptional failures must not leave raw CPU/NPU pointers in use after
    // the connector releases its sources. Normal transfers never synchronize here.
    aclrtSynchronizeStream(stream_);
    throw;
  }
  const aclError recorded = aclrtRecordEvent(done_, stream_);
  const aclError waited = recorded == ACL_SUCCESS
                              ? aclrtStreamWaitEvent(producer, done_) : recorded;
  if (recorded != ACL_SUCCESS || waited != ACL_SUCCESS) {
    aclrtSynchronizeStream(stream_);
    check_acl(recorded, "record transfer completion");
    check_acl(waited, "join low-priority stream");
  }
  // Existing per-layer store/load events follow this wait on the SAME torch
  // dispatch stream, covering all fragments and host-source/destination leases.
}
