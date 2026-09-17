# SPDX-License-Identifier: Apache-2.0
"""Keep allocator pages alive while RemoteFill reads completed CPU staging.

A tensor reference alone does not prevent LMCache's allocator from recycling
its storage. This lease owns MemoryObj references, not another payload copy.
"""

from collections import deque
from concurrent.futures import Future, wait
from threading import Lock
from time import monotonic
from typing import Any


class LayerwiseCPUFillLease:
    """Retain completed CPU pages until a known native terminal outcome.

    The caller must supply the real D2H completion events separately. Creating
    a lease proves ownership only, not completion of the producer's writes.
    """

    def __init__(self, pages: tuple[Any, ...]) -> None:
        self._lock = Lock()
        self._pages: list[Any] = []
        try:
            for page in {id(page): page for page in pages}.values():
                if not page.is_valid() or page.raw_data.device.type != "cpu":
                    raise ValueError("Layerwise RemoteFill needs valid CPU pages")
                page.ref_count_up()
                self._pages.append(page)
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        """Release once, only after all native readers are known to be done."""
        with self._lock:
            pages, self._pages = self._pages, []
        for page in pages:
            page.ref_count_down()


class LayerwisePutQueue:
    """Bound asynchronous CPU-page puts without a barrier every model forward.

    Single producer (the model thread). Backends own the DMA source references.
    Byte/count limits apply to submitted group batches, not model layers.
    A single oversized batch is allowed only in an otherwise empty queue.
    Failures remain sticky: an early LocalCPU hit is never proof of persistence.
    """

    def __init__(self, max_bytes: int, max_batches: int, timeout: float) -> None:
        if max_bytes <= 0 or max_batches <= 0 or timeout < 0:
            raise ValueError(
                "Layerwise put limits must be positive, timeout nonnegative"
            )
        self.max_bytes = max_bytes
        self.max_batches = max_batches
        self.timeout = timeout
        self.pending: deque[tuple[int, tuple[Future, ...]]] = deque()
        self.pending_bytes = 0
        self.error: BaseException | None = None

    def fail(self, error: BaseException) -> None:
        """Latch synchronous submission failures as well as future failures."""
        if self.error is None:
            self.error = error

    def poll(self) -> None:
        """Reap completed puts without waiting; propagate any observed error."""
        if self.error is not None:
            raise self.error
        # Reap out-of-order completions too: one slow DMA must not keep fully
        # completed later batches charged against admission capacity.
        try:
            remaining = deque()
            remaining_bytes = 0
            for size, futures in self.pending:
                done = True
                for future in futures:
                    if future.done():
                        future.result()
                    else:
                        done = False
                if not done:
                    remaining.append((size, futures))
                    remaining_bytes += size
            self.pending = remaining
            self.pending_bytes = remaining_bytes
        except BaseException as error:
            self.error = error
            raise

    def reserve(self, size: int) -> None:
        """Apply backpressure only at a batch boundary when limits are reached."""
        self.poll()
        deadline = monotonic() + self.timeout
        while self.pending and (
            len(self.pending) >= self.max_batches
            or self.pending_bytes + size > self.max_bytes
        ):
            self._wait_first(deadline)

    def add(self, size: int, futures: list[Future]) -> None:
        """Track a submitted put; its caller must first reserve this capacity."""
        if futures:
            self.pending.append((size, tuple(futures)))
            self.pending_bytes += size

    def drain(self) -> None:
        """Fence all pending puts before handoff, abort or allocator teardown."""
        self.poll()
        deadline = monotonic() + self.timeout
        while self.pending:
            self._wait_first(deadline)
        self.poll()

    def _wait_first(self, deadline: float) -> None:
        _, pending = wait(self.pending[0][1], timeout=max(0.0, deadline - monotonic()))
        if pending:
            self.error = TimeoutError("Layerwise CPU remote puts did not complete")
            raise self.error
        self.poll()
