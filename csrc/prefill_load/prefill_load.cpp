// SPDX-License-Identifier: Apache-2.0
#include "prefill_load.h"
#include "kernels/single_layer/single_layer_mem_kernels_v2.h"
#include <stdexcept>

// Reuse the production MLA/DSA copy implementation. Only the assigned token
// interval changes. Keep six GM arguments (the index address is a scalar), as
// in the original dense multi-chunk kernel.
#define DEFINE_PREFILL_LOAD(TYPE, SLOT, FORMAT)                                \
  extern "C" __global__ __aicore__ void                                        \
  single_layer_paged_kv_copy_prefill_##TYPE##_##SLOT##_##FORMAT(                 \
      __gm__ uint8_t *chunkPtrs, __gm__ uint8_t *chunkOffsets,                  \
      __gm__ uint8_t *chunkSizes, __gm__ uint8_t *key,                          \
      __gm__ uint8_t *value, __gm__ uint8_t *slots, uint64_t indexAddr,          \
      int64_t keyBytes, int64_t valueBytes, int64_t indexBytes,                 \
      int64_t kDims, int64_t vDims, int64_t indexDims,                          \
      int32_t maxTokensPerLoop, int32_t numTokens, int32_t numChunks,           \
      int32_t fixedChunkSize, int32_t totalTokens, int32_t blockSize,           \
      bool interleaved, int32_t tokenStart, int32_t tokenCount) {              \
    AscendC::TPipe pipe;                                                      \
    SingleLayerPagedKVCopyProcessor<                                          \
        TYPE, SLOT, MLADSAPolicy<TYPE, kvcache_ops::KVCacheFormat::FORMAT>> op{}; \
    op.GetPolicy().Init(key, value,                                           \
                        reinterpret_cast<__gm__ uint8_t *>(indexAddr),        \
                        keyBytes, valueBytes, indexBytes, kDims, vDims,       \
                        indexDims, 0, blockSize, interleaved);                \
    op.InitMlaDsaCommon(chunkPtrs, 0, maxTokensPerLoop, numTokens,              \
                        blockSize, false, &pipe);                            \
    const int32_t cores = AscendC::GetBlockNum();                              \
    const int32_t perCore = (tokenCount + cores - 1) / cores;                  \
    const int32_t start = tokenStart + AscendC::GetBlockIdx() * perCore;       \
    const int32_t end = min(tokenStart + tokenCount, start + perCore);        \
    for (int32_t token = start; token < end; token += maxTokensPerLoop) {     \
      op.processDenseMultiChunk(slots, chunkPtrs, chunkOffsets, chunkSizes,   \
                                token, min(maxTokensPerLoop, end - token),  \
                                numChunks, fixedChunkSize, totalTokens);     \
    }                                                                        \
  }

#define DEFINE_PREFILL_TYPE(TYPE)                 \
  DEFINE_PREFILL_LOAD(TYPE, int32_t, MLA_KV)       \
  DEFINE_PREFILL_LOAD(TYPE, int64_t, MLA_KV)       \
  DEFINE_PREFILL_LOAD(TYPE, int32_t, DSA_KV)       \
  DEFINE_PREFILL_LOAD(TYPE, int64_t, DSA_KV)

DEFINE_PREFILL_TYPE(half)
DEFINE_PREFILL_TYPE(int8_t)
#if ASCEND_AICORE_ARCH >= 220
DEFINE_PREFILL_TYPE(bfloat16_t)
#endif

namespace kvcache_ops {

#define LAUNCH_PREFILL(TYPE, SLOT, FORMAT)                                     \
  for_each_prefill_load_range(a.numTokens, [&](int32_t start, int32_t count) { \
    const uint32_t blockDim = cores < static_cast<uint32_t>(count)             \
                                  ? cores : static_cast<uint32_t>(count);    \
    single_layer_paged_kv_copy_prefill_##TYPE##_##SLOT##_##FORMAT               \
        <<<blockDim, nullptr, stream>>>(                                      \
            a.chunkPtrs, a.chunkOffsets, a.chunkSizes, a.key, a.value,        \
            a.slots, reinterpret_cast<uint64_t>(a.index), a.keyBytes,         \
            a.valueBytes, a.indexBytes, a.kDims, a.vDims, a.indexDims,         \
            a.maxTokensPerLoop, a.numTokens, a.numChunks, a.fixedChunkSize,   \
            a.totalTokens, a.blockSize, a.interleaved, start, count);         \
  });                                                                        \
  return

#define DISPATCH_PREFILL_SLOT(TYPE, SLOT)           \
  if (a.format == KVCacheFormat::DSA_KV) {          \
    LAUNCH_PREFILL(TYPE, SLOT, DSA_KV);             \
  } else {                                        \
    LAUNCH_PREFILL(TYPE, SLOT, MLA_KV);             \
  }

#define DISPATCH_PREFILL_TYPE(TYPE)                \
  if (a.slotType == AscendType::INT32) {            \
    DISPATCH_PREFILL_SLOT(TYPE, int32_t);          \
  } else if (a.slotType == AscendType::INT64) {     \
    DISPATCH_PREFILL_SLOT(TYPE, int64_t);          \
  }                                              \
  break

void launch_prefill_load(const PrefillLoadArgs &a, uint32_t cores,
                         void *stream) {
  // Dispatch once per layer, NOT once per short kernel.
  switch (a.type) {
  case AscendType::FP16:
    DISPATCH_PREFILL_TYPE(half);
  case AscendType::INT8:
    DISPATCH_PREFILL_TYPE(int8_t);
#if ASCEND_AICORE_ARCH >= 220
  case AscendType::BF16:
    DISPATCH_PREFILL_TYPE(bfloat16_t);
#endif
  default:
    break;
  }
  throw std::runtime_error("Unsupported dtype in split prefill load");
}
} // namespace kvcache_ops
