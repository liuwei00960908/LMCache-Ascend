# SPDX-License-Identifier: Apache-2.0
"""Guard the CANN launch-generator declaration format; no device compile here."""

from pathlib import Path
import re

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "csrc/prefill_load/prefill_load.cpp"
POINTER_NAMES = ("chunkPtrs", "chunkOffsets", "chunkSizes", "key", "value", "slots")


def kernel_parameters():
    source = SOURCE.read_text(encoding="utf-8").replace("\\\n", " ")
    match = re.search(
        r"single_layer_paged_kv_copy_prefill_.*?\((.*?)\)\s*\{", source, re.S
    )
    assert match is not None
    return [" ".join(param.split()) for param in match.group(1).split(",")]


@pytest.mark.parametrize("position,name", list(enumerate(POINTER_NAMES)))
def test_gm_pointer_declarations_match_existing_kernel_style(position, name):
    # CANN 8.5 emitted '*chunkPtrs' as the forwarded argument for the previous
    # 'uint8_t *chunkPtrs' spelling. Keep '*' on the type, as in vendor kernels,
    # so the final whitespace-delimited token is the bare parameter name.
    declaration = kernel_parameters()[position]
    assert declaration == f"__gm__ uint8_t* {name}"
    assert declaration.rsplit(None, 1)[1] == name


def test_index_address_and_split_range_abi_remain_unchanged():
    parameters = kernel_parameters()
    assert parameters[6:] == [
        "uint64_t indexAddr",
        "int64_t keyBytes",
        "int64_t valueBytes",
        "int64_t indexBytes",
        "int64_t kDims",
        "int64_t vDims",
        "int64_t indexDims",
        "int32_t maxTokensPerLoop",
        "int32_t numTokens",
        "int32_t numChunks",
        "int32_t fixedChunkSize",
        "int32_t totalTokens",
        "int32_t blockSize",
        "bool interleaved",
        "int32_t tokenStart",
        "int32_t tokenCount",
    ]
