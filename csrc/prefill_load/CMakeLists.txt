# Build the original and prefill kernels in ONE AscendC library/scope. Including
# ascendc_with_def.cmake in a second sibling directory recreates host_intf_pub
# on CANN 8.5 and also loses directory-scoped AscendC configuration variables.
# Keep the vendored sources and ABI unchanged.
set(KVCACHE_OPS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/kvcache-ops")
file(GLOB KERNEL_FILES
  "${KVCACHE_OPS_SOURCE_DIR}/kernels/*.cpp"
  "${KVCACHE_OPS_SOURCE_DIR}/kernels/cachegen/*.cpp"
  "${KVCACHE_OPS_SOURCE_DIR}/kernels/fused_rope/*.cpp"
  "${KVCACHE_OPS_SOURCE_DIR}/kernels/single_layer/*.cpp"
  "${KVCACHE_OPS_SOURCE_DIR}/kernels/multi_layer/*.cpp"
)
list(APPEND KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/prefill_load.cpp")
include("${KVCACHE_OPS_SOURCE_DIR}/ascendc_with_def.cmake")
if(DEFINED ASCEND_AICORE_ARCH)
  ascendc_library_with_def(cache_kernels SHARED ${KERNEL_FILES})
  ascendc_compile_definitions(cache_kernels PRIVATE
    -DASCEND_AICORE_ARCH=${ASCEND_AICORE_ARCH})
else()
  ascendc_library(cache_kernels SHARED ${KERNEL_FILES})
endif()
ascendc_include_directories(cache_kernels PRIVATE "${KVCACHE_OPS_SOURCE_DIR}")
target_include_directories(cache_kernels PUBLIC "${KVCACHE_OPS_SOURCE_DIR}")
