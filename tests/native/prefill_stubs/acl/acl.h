// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
using aclError = int;
using aclrtStream = void *;
using aclrtEvent = void *;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_STREAM_FAST_LAUNCH = 1;
constexpr int ACL_STREAM_FAST_SYNC = 2;
constexpr int ACL_EVENT_SYNC = 1;
aclError aclrtCreateStreamWithConfig(aclrtStream *, uint32_t, uint32_t);
aclError aclrtRecordEvent(aclrtEvent, aclrtStream);
aclError aclrtStreamWaitEvent(aclrtStream, aclrtEvent);
aclError aclrtSynchronizeStream(aclrtStream);
aclError aclrtDestroyStream(aclrtStream);
aclError aclrtDestroyEvent(aclrtEvent);
