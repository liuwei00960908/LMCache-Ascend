# SPDX-License-Identifier: Apache-2.0
"""Configure the actual kernel build graph against a non-idempotent fake SDK."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("PREFILL_TEST_CMAKE") or shutil.which("cmake")


@pytest.mark.skipif(not CMAKE, reason="CMake is required for build-graph tests")
@pytest.mark.parametrize("mindspore", [False, True])
@pytest.mark.parametrize("arch", [None, "220"])
def test_one_ascendc_initialization_with_all_original_kernels(
    tmp_path, mindspore, arch
):
    env = os.environ.copy()
    env.pop("USE_MINDSPORE", None)
    if mindspore:
        env["USE_MINDSPORE"] = "1"
    command = [
        CMAKE,
        "-S",
        str(ROOT / "tests/native/cmake_prefill_build"),
        "-B",
        str(tmp_path / "build"),
        f"-DPREFILL_REPO_ROOT={ROOT.as_posix()}",
    ]
    if arch is not None:
        command.append(f"-DASCEND_AICORE_ARCH={arch}")
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "PREFILL_BUILD_GRAPH_PASSED" in output
