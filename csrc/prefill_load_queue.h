// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "prefill_load/prefill_load.h"
#include <acl/acl.h>
#include <cstdint>

// One FIFO stream per P connector, serialized by its torch load-stream queue.
// Ex-events capture record generations: no per-layer event allocation/polling.
class PrefillLoadQueue {
public:
  PrefillLoadQueue();
  ~PrefillLoadQueue();
  PrefillLoadQueue(const PrefillLoadQueue &) = delete;
  PrefillLoadQueue &operator=(const PrefillLoadQueue &) = delete;
  void submit(const kvcache_ops::PrefillLoadArgs &args, uint32_t cores,
              aclrtStream producer);
  int priority() const { return priority_; }
  int device_index() const { return device_index_; }
  bool priority_verified() const { return priority_verified_; }

private:
  void release() noexcept;
  aclrtStream stream_ = nullptr;
  aclrtStream producer_ = nullptr;
  aclrtEvent ready_ = nullptr;
  aclrtEvent done_ = nullptr;
  int device_index_ = -1;
  int priority_ = 0;
  bool priority_verified_ = false;
};
