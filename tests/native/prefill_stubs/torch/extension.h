// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sstream>
#include <stdexcept>
template <typename... T> void test_torch_check(bool ok, T... message) {
  if (!ok) {
    std::ostringstream text;
    (text << ... << message);
    throw std::runtime_error(text.str());
  }
}
#define TORCH_CHECK(...) test_torch_check(__VA_ARGS__)
