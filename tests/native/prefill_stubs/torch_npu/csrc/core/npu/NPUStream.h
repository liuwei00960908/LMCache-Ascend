// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace c10_npu {
struct TestStream { int device_index() const { return 0; } };
inline TestStream getCurrentNPUStream() { return {}; }
}
