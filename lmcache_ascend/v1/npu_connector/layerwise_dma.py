"""CPU-only address planning for P-node layerwise KV DMA.

The plan is computed once per request/bank. A segment never crosses an
LMCache chunk, a physical DSA bundle, or a gap in the vLLM slot map. Latent
nope/rope planes are bound separately when a layer's pointers are known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DmaSegment:
    chunk: int
    slot: int
    chunk_token: int
    tokens: int


def plan_bundle_copies(
    slots: Sequence[int], chunk_sizes: Sequence[int], bundle_tokens: int
) -> tuple[DmaSegment, ...]:
    if bundle_tokens <= 0 or not chunk_sizes or any(n <= 0 for n in chunk_sizes):
        raise ValueError("DMA needs positive bundle and chunk token counts")
    if sum(chunk_sizes) != len(slots):
        raise ValueError("DMA chunk sizes must cover the slot map exactly")
    segments: list[DmaSegment] = []
    offset = 0
    for chunk, chunk_size in enumerate(chunk_sizes):
        local = 0
        while local < chunk_size:
            slot = int(slots[offset + local])
            if slot < 0:
                raise ValueError("DMA cannot represent an invalid KV slot")
            count = 1
            limit = min(chunk_size - local, bundle_tokens - slot % bundle_tokens)
            while count < limit and int(slots[offset + local + count]) == slot + count:
                count += 1
            segments.append(DmaSegment(chunk, slot, local, count))
            local += count
        offset += chunk_size
    return tuple(segments)


def bind_copy_addresses(
    plan: Sequence[DmaSegment],
    host_ptrs: Sequence[int],
    npu_ptrs: Sequence[int],
    chunk_sizes: Sequence[int],
    plane_widths: Sequence[int],
    element_bytes: int,
    *,
    device_to_host: bool,
    host_chunk_tokens: Sequence[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Return (destination, source, bytes) for aclrtMemcpyAsync.

    The LMCache MLA/DSA CPU chunk is plane-major (all nope, then all rope).
    The NPU cache exposes the planes as separate contiguous tensors.
    """
    if len(host_ptrs) != len(chunk_sizes) or len(npu_ptrs) != len(plane_widths):
        raise ValueError("DMA pointer count differs from the prepared layout")
    if host_chunk_tokens is None:
        host_chunk_tokens = chunk_sizes
    if len(host_chunk_tokens) != len(chunk_sizes) or any(
        physical < copied for physical, copied in zip(
            host_chunk_tokens, chunk_sizes, strict=True
        )
    ):
        raise ValueError("DMA host chunk is shorter than its copied token range")
    if element_bytes <= 0 or any(width <= 0 for width in plane_widths):
        raise ValueError("DMA element and plane widths must be positive")
    copies: list[tuple[int, int, int]] = []
    for segment in plan:
        chunk_tokens = host_chunk_tokens[segment.chunk]
        host_base = host_ptrs[segment.chunk]
        host_plane_offset = 0
        for npu_base, width in zip(npu_ptrs, plane_widths, strict=True):
            byte_count = segment.tokens * width * element_bytes
            host = (
                host_base + host_plane_offset
                + segment.chunk_token * width * element_bytes
            )
            npu = npu_base + segment.slot * width * element_bytes
            copies.append(
                (host, npu, byte_count)
                if device_to_host else (npu, host, byte_count)
            )
            host_plane_offset += chunk_tokens * width * element_bytes
    return copies
