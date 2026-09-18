// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernels/types.h"
#include <cstdint>

namespace kvcache_ops {

constexpr int32_t PREFILL_LOAD_TOKENS_PER_KERNEL = 16384;

// All pointers address the ORIGINAL, unsliced metadata. tokenStart in the
// device kernel is an absolute index into that metadata, not a chunk offset.
struct PrefillLoadArgs {
  AscendType type;
  AscendType slotType;
  KVCacheFormat format;
  uint8_t *chunkPtrs;
  uint8_t *chunkOffsets;
  uint8_t *chunkSizes;
  uint8_t *key;
  uint8_t *value;
  uint8_t *index;
  uint8_t *slots;
  int64_t keyBytes;
  int64_t valueBytes;
  int64_t indexBytes;
  int64_t kDims;
  int64_t vDims;
  int64_t indexDims;
  int32_t maxTokensPerLoop;
  int32_t numTokens;
  int32_t numChunks;
  int32_t fixedChunkSize;
  int32_t totalTokens;
  int32_t blockSize;
  bool interleaved;
  bool store = false;
};

// Host-only submission helper; shared with the CPU boundary tests. The loop
// submits everything without querying completion or allocating slice metadata.
template <typename Launch>
inline void for_each_prefill_load_range(int32_t tokens, Launch launch) {
  for (int32_t start = 0; start < tokens;) {
    const int32_t remaining = tokens - start;
    const int32_t count = remaining < PREFILL_LOAD_TOKENS_PER_KERNEL
                              ? remaining : PREFILL_LOAD_TOKENS_PER_KERNEL;
    launch(start, count);
    start += count;
  }
}

void launch_prefill_load(const PrefillLoadArgs &args, uint32_t cores,
                         void *stream);
// D2H keeps the original single-kernel implementation and metadata layout.
void launch_prefill_store(const PrefillLoadArgs &args, uint32_t cores,
                          void *stream);
} // namespace kvcache_ops
