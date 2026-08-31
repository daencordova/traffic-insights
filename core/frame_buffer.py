"""Optimized circular buffer for frame storage with low overhead.

This module provides an efficient circular buffer for storing video frames
with associated metadata, optimized for use in computer vision pipelines.

Features:
    - Preallocated memory to avoid reallocations
    - Support for selective dropping
    - Performance statistics
    - Thread-safe operations
    - Immediate frame release for memory optimization
    - Batch retrieval for efficient processing
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
import threading
import time

import numpy as np

from core.constants import BUFFER_USAGE_FULL, BUFFER_USAGE_OVERFLOW


class BufferStatus(Enum):
    """Possible states of the circular buffer.

    Attributes:
        EMPTY: Buffer is empty with no frames.
        PARTIAL: Buffer has some frames (0-70% capacity).
        FULL: Buffer is nearly full (70-90% capacity).
        OVERFLOW: Buffer at risk of overflow (>90% capacity).
        DRAINING: Buffer is being drained.
    """

    EMPTY = auto()
    PARTIAL = auto()
    FULL = auto()
    OVERFLOW = auto()
    DRAINING = auto()


class FrameMetadata:
    """Metadata associated with a stored frame.

    This class stores timing and processing information for each frame
    to enable performance monitoring and debugging.

    Attributes:
        timestamp: Capture timestamp of the frame (time.time()).
        frame_number: Sequential frame number.
        source_fps: FPS of the source origin.
        capture_time_ms: Capture time in milliseconds.
        processing_time_ms: Processing time in milliseconds.
        dropped: Indicates if the frame was dropped.

    Example:
        >>> metadata = FrameMetadata(
        ...     timestamp=time.time(), frame_number=42, source_fps=30.0, capture_time_ms=16.67
        ... )
    """

    __slots__ = (
        "timestamp",
        "frame_number",
        "source_fps",
        "capture_time_ms",
        "processing_time_ms",
        "dropped",
    )

    def __init__(
        self,
        timestamp: float,
        frame_number: int,
        source_fps: float,
        capture_time_ms: float,
        processing_time_ms: float = 0.0,
        dropped: bool = False,
    ):
        """Initializes frame metadata.

        Args:
            timestamp: Capture timestamp (time.time()).
            frame_number: Sequential frame number.
            source_fps: FPS of the source origin.
            capture_time_ms: Capture time in milliseconds.
            processing_time_ms: Processing time in milliseconds (default: 0.0).
            dropped: Indicates if the frame was dropped (default: False).
        """
        self.timestamp = timestamp or time.time()
        self.frame_number = frame_number
        self.source_fps = source_fps
        self.capture_time_ms = capture_time_ms
        self.processing_time_ms = processing_time_ms
        self.dropped = dropped


class FrameBuffer:
    """Optimized circular buffer with preallocated memory for frames.

    This buffer is designed to minimize memory allocations and provide
    fast access to frames in a real-time video processing pipeline.

    Features:
        - Preallocated memory to avoid reallocations
        - Support for selective dropping ('oldest' or 'newest')
        - Performance statistics
        - Thread-safe operations with locking
        - Immediate frame release for memory optimization
        - Batch retrieval for efficient processing
        - Watermark tracking for monitoring

    Attributes:
        max_size: Maximum buffer size.
        drop_policy: Drop policy when full ('oldest' or 'newest').
        dtype: Data type of frames (default: np.uint8).
        count: Current number of frames in the buffer.
        status: Current buffer status.

    Example:
        >>> buffer = FrameBuffer(max_size=30, frame_shape=(480, 640, 3))
        >>>
        >>> # Add frames
        >>> buffer.put(frame, metadata)
        >>>
        >>> # Retrieve frames
        >>> frame, meta = buffer.get()
        >>>
        >>> # Get batch
        >>> batch = buffer.get_batch(batch_size=10)
    """

    __slots__ = (
        "max_size",
        "drop_policy",
        "dtype",
        "_preallocated",
        "_buffer",
        "_metadata",
        "_lock",
        "_total_frames_received",
        "_total_frames_dropped",
        "_total_frames_processed",
        "_buffer_overflow_count",
        "_head",
        "_tail",
        "_count",
        "_status",
        "_last_watermark_time",
        "_watermark_history",
        "_memory_freed",
        "_total_memory_allocated",
    )

    def __init__(
        self,
        max_size: int = 30,
        frame_shape: tuple[int, int, int] | None = None,
        dtype: np.dtype = np.uint8,
        drop_policy: str = "oldest",
    ):
        """Initializes the circular buffer.

        Args:
            max_size: Maximum buffer size (must be > 0).
            frame_shape: Predefined shape for preallocating memory.
                If provided, memory is preallocated for all frames.
            dtype: Data type of frames.
            drop_policy: Drop policy when full.
                'oldest': Discard the oldest frame.
                'newest': Discard the newest frame.

        Raises:
            ValueError: If max_size is <= 0.
            ValueError: If drop_policy is invalid.

        Example:
            >>> # Preallocated buffer for performance
            >>> buffer = FrameBuffer(max_size=30, frame_shape=(480, 640, 3), drop_policy="oldest")
            >>>
            >>> # Dynamic buffer for flexibility
            >>> buffer = FrameBuffer(max_size=30)
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be greater than 0: {max_size}")

        if drop_policy not in ["oldest", "newest"]:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        self.max_size = max_size
        self.drop_policy = drop_policy
        self.dtype = dtype

        self._preallocated = frame_shape is not None
        if self._preallocated:
            self._buffer = np.zeros((max_size, *frame_shape), dtype=dtype)
        else:
            self._buffer: deque[np.ndarray] = deque(maxlen=max_size)

        self._metadata: deque[FrameMetadata] = deque(maxlen=max_size)
        self._lock = threading.RLock()

        self._total_frames_received = 0
        self._total_frames_dropped = 0
        self._total_frames_processed = 0
        self._buffer_overflow_count = 0

        self._head = 0
        self._tail = 0
        self._count = 0

        self._status = BufferStatus.EMPTY

        self._last_watermark_time = time.time()
        self._watermark_history: deque[float] = deque(maxlen=60)

        self._memory_freed = 0
        self._total_memory_allocated = 0

    def put(self, frame: np.ndarray, metadata: FrameMetadata | None = None) -> bool:
        """Inserts a frame into the buffer (optimized).

        This method adds a frame to the buffer. If the buffer is full and
        drop_policy is 'oldest', the oldest frame is discarded. If drop_policy
        is 'newest', the new frame is discarded.

        Args:
            frame: Frame to insert (numpy array).
            metadata: Metadata associated with the frame (optional).

        Returns:
            bool: True if the frame was inserted, False if dropped.

        Example:
            >>> success = buffer.put(frame, FrameMetadata(...))
            >>> if not success:
            ...     print("Frame was dropped due to overflow")
        """
        if frame is None or frame.size == 0:
            return False

        with self._lock:
            self._total_frames_received += 1

            if self._is_full():
                self._handle_overflow()
                self._total_frames_dropped += 1
                return False

            if self._preallocated:
                np.copyto(self._buffer[self._tail], frame)
            else:
                self._buffer.append(frame.copy())

            if metadata is None:
                metadata = FrameMetadata(
                    timestamp=time.time(),
                    frame_number=self._total_frames_received,
                    source_fps=0.0,
                    capture_time_ms=0.0,
                )
            self._metadata.append(metadata)

            self._count += 1
            self._tail = (self._tail + 1) % self.max_size

            self._update_status()

            return True

    def get(
        self, block: bool = True, timeout: float = 0.1
    ) -> tuple[np.ndarray, FrameMetadata] | None:
        """Gets the next frame from the buffer and releases memory immediately.

        This method retrieves and removes the oldest frame from the buffer.
        In preallocated mode, the frame memory is zeroed to free it.

        Args:
            block: Whether to block waiting for a frame.
            timeout: Timeout in seconds if block=True.

        Returns:
            Optional[Tuple[np.ndarray, FrameMetadata]]: Tuple (frame, metadata)
                or None if no frames are available.

        Raises:
            TimeoutError: If timeout expires and no frames are available.

        Example:
            >>> result = buffer.get(block=True, timeout=0.5)
            >>> if result:
            ...     frame, metadata = result
            ...     process_frame(frame)
        """
        start_time = time.time()

        while True:
            with self._lock:
                if self._count > 0:
                    if self._preallocated:
                        frame = self._buffer[self._head].copy()
                        self._buffer[self._head].fill(0)
                    else:
                        frame = self._buffer.popleft()

                    metadata = self._metadata.popleft()

                    self._count -= 1
                    self._head = (self._head + 1) % self.max_size
                    self._total_frames_processed += 1

                    self._update_status()

                    return frame, metadata

                if not block or time.time() - start_time >= timeout:
                    return None

            time.sleep(0.001)

    def get_batch(self, batch_size: int, timeout: float = 0.05) -> list:
        """Gets a batch of frames releasing memory immediately.

        This method retrieves multiple frames at once for efficient processing.

        Args:
            batch_size: Maximum batch size.
            timeout: Timeout for waiting for frames in seconds.

        Returns:
            list: List of tuples (frame, metadata) with up to batch_size items.

        Example:
            >>> batch = buffer.get_batch(batch_size=10, timeout=0.1)
            >>> for frame, metadata in batch:
            ...     process_frame(frame)
        """
        batch = []
        start_time = time.time()

        while len(batch) < batch_size and (time.time() - start_time) < timeout:
            result = self.get(block=False)
            if result is None:
                time.sleep(0.001)
                continue
            batch.append(result)

        return batch

    def peek(self) -> tuple[np.ndarray, FrameMetadata] | None:
        """Peek at the next frame without removing it from the buffer.

        Returns:
            Optional[Tuple[np.ndarray, FrameMetadata]]: Tuple (frame, metadata)
                or None if no frames are available.

        Example:
            >>> frame, meta = buffer.peek()
            >>> if frame is not None:
            ...     # Preview without consuming
            ...     display_preview(frame)
        """
        with self._lock:
            if self._count == 0:
                return None

            if self._preallocated:
                frame = self._buffer[self._head].copy()
            else:
                frame = self._buffer[0].copy()

            metadata = self._metadata[0]
            return frame, metadata

    def clear(self) -> int:
        """Clears the buffer and returns the number of frames removed.

        Returns:
            int: Number of frames removed from the buffer.

        Example:
            >>> removed = buffer.clear()
            >>> print(f"Removed {removed} frames from buffer")
        """
        with self._lock:
            count = self._count
            if self._preallocated:
                for i in range(self.max_size):
                    self._buffer[i].fill(0)
            else:
                self._buffer.clear()
            self._metadata.clear()
            self._head = 0
            self._tail = 0
            self._count = 0
            self._update_status()
            return count

    def _is_full(self) -> bool:
        """Checks if the buffer is full."""
        return self._count >= self.max_size

    def _handle_overflow(self):
        """Handles overflow according to the configured policy.

        If drop_policy is 'oldest', removes the oldest frame.
        If drop_policy is 'newest', does nothing (drops the new one).
        """
        if self.drop_policy == "oldest":
            if self._preallocated:
                old_frame = self._buffer[self._head]
                self._memory_freed += old_frame.nbytes
                old_frame.fill(0)
                self._head = (self._head + 1) % self.max_size
            else:
                old_frame = self._buffer.popleft()
                if old_frame is not None:
                    self._memory_freed += old_frame.nbytes
            self._metadata.popleft()
            self._count -= 1

    def _update_status(self):
        """Updates the buffer status based on its occupancy."""
        ratio = self._count / self.max_size if self.max_size > 0 else 0

        if self._count == 0:
            self._status = BufferStatus.EMPTY
        elif ratio >= BUFFER_USAGE_OVERFLOW:
            self._status = BufferStatus.OVERFLOW
        elif ratio >= BUFFER_USAGE_FULL:
            self._status = BufferStatus.FULL
        else:
            self._status = BufferStatus.PARTIAL

        current_time = time.time()
        if current_time - self._last_watermark_time >= 1.0:
            self._watermark_history.append(ratio)
            self._last_watermark_time = current_time

    def get_stats(self) -> dict:
        """Gets detailed statistics for the buffer.

        Returns:
            dict: Dictionary with buffer statistics including:
                - size: Current size
                - max_size: Maximum size
                - capacity_ratio: Usage ratio
                - status: Current status
                - total_frames_received: Total frames received
                - total_frames_dropped: Total frames dropped
                - total_frames_processed: Total frames processed
                - drop_rate: Drop rate
                - overflow_count: Number of overflows
                - avg_watermark: Average occupancy level
                - preallocated: Whether preallocated memory is used
                - drop_policy: Drop policy
                - memory_freed_mb: Memory freed in MB
                - total_memory_allocated_mb: Total memory allocated in MB

        Example:
            >>> stats = buffer.get_stats()
            >>> print(f"Drop rate: {stats['drop_rate']:.2%}")
            >>> print(f"Average occupancy: {stats['avg_watermark']:.2%}")
        """
        with self._lock:
            return {
                "size": self._count,
                "max_size": self.max_size,
                "capacity_ratio": self._count / self.max_size if self.max_size > 0 else 0,
                "status": self._status.name,
                "total_frames_received": self._total_frames_received,
                "total_frames_dropped": self._total_frames_dropped,
                "total_frames_processed": self._total_frames_processed,
                "drop_rate": self._total_frames_dropped / max(1, self._total_frames_received),
                "overflow_count": self._buffer_overflow_count,
                "avg_watermark": sum(self._watermark_history)
                / max(1, len(self._watermark_history)),
                "preallocated": self._preallocated,
                "drop_policy": self.drop_policy,
                "memory_freed_mb": self._memory_freed / (1024 * 1024),
                "total_memory_allocated_mb": self._total_memory_allocated / (1024 * 1024),
            }

    @property
    def count(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return self._count

    @property
    def is_empty(self) -> bool:
        """Indicates whether the buffer is empty."""
        with self._lock:
            return self._count == 0

    @property
    def is_full(self) -> bool:
        """Indicates whether the buffer is full."""
        with self._lock:
            return self._count >= self.max_size

    @property
    def status(self) -> BufferStatus:
        """Current status of the buffer."""
        with self._lock:
            return self._status

    def __len__(self) -> int:
        """Returns the number of frames in the buffer."""
        with self._lock:
            return self._count

    def __bool__(self) -> bool:
        """Indicates whether the buffer has frames."""
        return self.count > 0
