# Keep the extension in LMCache-Ascend; do not change the vendored kernel ABI.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(${PROJECT_SOURCE_DIR}/third_party/kvcache-ops/ascendc_with_def.cmake)
if(DEFINED ASCEND_AICORE_ARCH)
  ascendc_library_with_def(prefill_load_kernels STATIC prefill_load.cpp)
  ascendc_compile_definitions(prefill_load_kernels PRIVATE
    -DASCEND_AICORE_ARCH=${ASCEND_AICORE_ARCH})
else()
  ascendc_library(prefill_load_kernels STATIC prefill_load.cpp)
endif()
ascendc_include_directories(prefill_load_kernels PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/kvcache-ops)
