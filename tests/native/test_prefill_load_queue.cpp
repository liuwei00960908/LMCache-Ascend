// SPDX-License-Identifier: Apache-2.0
// Compile the REAL host queue against a fake ACL boundary (no NPU required).
#include "prefill_load_queue.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::string> calls;
int least = 7, priority_readback = 7, sync_count = 0, event_count = 0;
bool missing_api = false, missing_readback = false, launch_error = false;
bool completion_error = false, join_error = false;
auto low = reinterpret_cast<void *>(101);
auto producer = reinterpret_cast<void *>(102);
aclError range(int32_t *l, int32_t *g) { *l = least; *g = 0; return 0; }
aclError get_priority(aclrtStream, uint32_t *p) { *p = priority_readback; return 0; }
aclError create_event(aclrtEvent *e, uint32_t flag) {
  assert(flag == ACL_EVENT_SYNC);
  *e = reinterpret_cast<void *>(static_cast<uintptr_t>(++event_count));
  return 0;
}
template <typename F> void expect_error(F f, const char *text) {
  try { f(); assert(false); }
  catch (const std::runtime_error &e) { assert(std::strstr(e.what(), text)); }
}
}

void *dlsym(void *, const char *name) {
  if (missing_api) return nullptr;
  if (!std::strcmp(name, "aclrtDeviceGetStreamPriorityRange"))
    return reinterpret_cast<void *>(range);
  if (!std::strcmp(name, "aclrtCreateEventExWithFlag"))
    return reinterpret_cast<void *>(create_event);
  return missing_readback ? nullptr : reinterpret_cast<void *>(get_priority);
}
aclError aclrtCreateStreamWithConfig(aclrtStream *s, uint32_t p, uint32_t) {
  assert(p == 7); *s = low; return 0;
}
aclError aclrtRecordEvent(aclrtEvent, aclrtStream s) {
  calls.push_back(s == producer ? "ready" : "done");
  return completion_error && s == low ? 98 : 0;
}
aclError aclrtStreamWaitEvent(aclrtStream s, aclrtEvent) {
  calls.push_back(s == low ? "wait_ready" : "join");
  return join_error && s == producer ? 99 : 0;
}
aclError aclrtSynchronizeStream(aclrtStream) { ++sync_count; return 0; }
aclError aclrtDestroyStream(aclrtStream) { return 0; }
aclError aclrtDestroyEvent(aclrtEvent) { return 0; }

namespace kvcache_ops {
void launch_prefill_store(const PrefillLoadArgs &a, uint32_t, void *s) {
  assert(s == low && a.store);
  calls.push_back("store:" + std::to_string(a.numTokens));
}
void launch_prefill_load(const PrefillLoadArgs &a, uint32_t, void *s) {
  assert(s == low && !a.store);
  for_each_prefill_load_range(a.numTokens, [&](int32_t start, int32_t count) {
    calls.push_back(std::to_string(start) + ":" + std::to_string(count));
  });
  if (launch_error) throw std::runtime_error("launch failed");
}
}

int main() {
  using namespace kvcache_ops;
  // Compile and exercise the production range helper, including integer bounds.
  for (int32_t tokens : {0, 1, 16383, 16384, 16385, 32768, 32769, 130000,
                         131614, std::numeric_limits<int32_t>::max()}) {
    int64_t cursor = 0, launches = 0;
    for_each_prefill_load_range(tokens, [&](int32_t start, int32_t count) {
      assert(start == cursor && count > 0 && count <= 16384);
      cursor += count; ++launches;
    });
    assert(cursor == tokens && launches == (static_cast<int64_t>(tokens) + 16383) / 16384);
  }
  missing_api = true;
  expect_error([] { PrefillLoadQueue q; }, "lacks priority-range");
  missing_api = false;
  least = 0;
  expect_error([] { PrefillLoadQueue q; }, "not supported");
  least = 7;
  priority_readback = 0;
  expect_error([] { PrefillLoadQueue q; }, "not applied");
  priority_readback = 7;
  missing_readback = true;
  { PrefillLoadQueue q; assert(!q.priority_verified()); }
  missing_readback = false;
  {
    PrefillLoadQueue q;
    assert(q.priority() == 7 && q.priority_verified() && q.device_index() == 0);
    const int events = event_count;
    PrefillLoadArgs a{};
    a.numTokens = 32769;
    calls.clear(); sync_count = 0;
    for (int layer = 0; layer < 101; ++layer) q.submit(a, 48, producer);
    const std::vector<std::string> expected{
        "ready", "wait_ready", "0:16384", "16384:16384", "32768:1", "done", "join"};
    assert(calls.size() == 101 * expected.size());
    for (int layer = 0; layer < 101; ++layer)
      assert(std::equal(expected.begin(), expected.end(), calls.begin() + layer * expected.size()));
    assert(event_count == events && sync_count == 0);
    calls.clear();
    PrefillLoadArgs store_args{};
    store_args.numTokens = 4096;
    store_args.store = true;
    for (int layer = 0; layer < 79; ++layer) {
      // Latent and index stores, then both reloads, on ONE dispatch stream.
      q.submit(store_args, 48, producer);
      q.submit(store_args, 48, producer);
      q.submit(a, 48, producer);
      q.submit(a, 48, producer);
    }
    const std::vector<std::string> store_expected{
        "ready", "wait_ready", "store:4096", "done", "join"};
    auto cursor = calls.begin();
    for (int layer = 0; layer < 79; ++layer) {
      for (int group = 0; group < 2; ++group) {
        assert(std::equal(store_expected.begin(), store_expected.end(), cursor));
        cursor += store_expected.size();
      }
      for (int group = 0; group < 2; ++group) {
        assert(std::equal(expected.begin(), expected.end(), cursor));
        cursor += expected.size();
      }
    }
    assert(cursor == calls.end() && event_count == events && sync_count == 0);
    launch_error = true;
    expect_error([&] { q.submit(a, 48, producer); }, "launch failed");
    assert(sync_count == 1);
    launch_error = false; completion_error = true;
    expect_error([&] { q.submit(a, 48, producer); }, "record transfer completion");
    assert(sync_count == 2);
    completion_error = false; join_error = true;
    expect_error([&] { q.submit(a, 48, producer); }, "join low-priority stream");
    assert(sync_count == 3);
    join_error = false;
  }
  std::cout << "PASS: ranges, capability checks, FIFO/event ordering, no hot-path sync, error drain\n";
}
