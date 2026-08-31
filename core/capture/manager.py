"""Video capture manager with automatic reconnection and flow control.

Extracted from AsyncPipeline to improve code structure and separation of concerns.
This module handles video capture with robust error handling and adaptive flow control.
"""

from collections.abc import Callable
import threading
import time

import numpy as np

from core.capture.reconnector import Reconnector
from core.constants import (
    BUFFER_DROP_THRESHOLD,
    BUFFER_RECOVERY_THRESHOLD,
    BUFFER_SKIP_CONSECUTIVE_LIMIT,
    BUFFER_SKIP_MAX,
    BUFFER_USAGE_RECOVERY_BOUNDARY,
    CAPTURE_DEFAULT_INTERVAL_CPU,
    CAPTURE_DEFAULT_INTERVAL_GPU,
    CAPTURE_MAX_CONSECUTIVE_ERRORS,
    CAPTURE_MAX_FPS_CPU,
    CAPTURE_MIN_FPS_CPU,
    CAPTURE_TARGET_FPS_CPU,
    CAPTURE_TARGET_FPS_GPU,
    DEFAULT_SLEEP_SHORT,
    HEALTH_ISSUES_TRIM_SIZE,
    MAX_HEALTH_ISSUES,
    MIN_FRAME_DIMENSION,
    SLEEP_PAUSE_CHECK,
)
from core.frame_buffer import FrameBuffer, FrameMetadata
from core.validators import validate_frame
from utils.logger import LoggerMixin


class CaptureManager(LoggerMixin):
    """Manages video frame capture from a video source.

    This class handles all aspects of video capture including:
        - Connection and automatic reconnection
        - Flow control with circular buffer management
        - FPS monitoring and adaptive capture rate
        - Pause/resume management
        - Error handling and health monitoring

    The capture manager works in conjunction with a FrameBuffer to
    decouple capture from processing, allowing for smooth video pipelines.

    Attributes:
        config: System configuration.
        buffer: Circular buffer for frames.
        stop_event: Event to stop capture.
        pause_event: Event to pause capture.
        fps_target: Target FPS.
        is_cpu_mode: Whether in CPU mode (affects performance settings).

    Example:
        >>> from core.frame_buffer import FrameBuffer
        >>>
        >>> buffer = FrameBuffer(max_size=30, frame_shape=(480, 640, 3))
        >>> stop_event = threading.Event()
        >>> pause_event = threading.Event()
        >>>
        >>> manager = CaptureManager(
        ...     config=config,
        ...     buffer=buffer,
        ...     stop_event=stop_event,
        ...     pause_event=pause_event,
        ...     is_cpu_mode=True,
        ... )
        >>>
        >>> # Start capture in a separate thread
        >>> import threading
        >>> thread = threading.Thread(target=manager.run, args=("0",))
        >>> thread.start()
        >>>
        >>> # Get frames from buffer
        >>> frame, metadata = buffer.get()
        >>>
        >>> # Stop capture
        >>> manager.stop()
    """

    def __init__(
        self,
        config,
        buffer: FrameBuffer,
        stop_event: threading.Event,
        pause_event: threading.Event,
        *,
        is_cpu_mode: bool = False,
        capture_interval: float | None = None,
    ) -> None:
        """Initializes the capture manager.

        Args:
            config: System configuration object.
            buffer: Circular buffer for storing frames.
            stop_event: Event to signal capture stop.
            pause_event: Event to signal capture pause.
            is_cpu_mode: Whether running in CPU mode (lower performance settings).
            capture_interval: Custom capture interval in seconds (optional).

        Example:
            >>> manager = CaptureManager(
            ...     config=config,
            ...     buffer=buffer,
            ...     stop_event=stop_event,
            ...     pause_event=pause_event,
            ...     is_cpu_mode=False,
            ...     capture_interval=0.033,  # ~30 FPS
            ... )
        """
        self.config = config
        self.buffer = buffer
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.is_cpu_mode = is_cpu_mode

        if capture_interval is None:
            capture_interval = (
                CAPTURE_DEFAULT_INTERVAL_CPU if is_cpu_mode else CAPTURE_DEFAULT_INTERVAL_GPU
            )

        self._capture_interval = capture_interval
        self._last_capture_time = time.time()
        self._frame_count = 0
        self._dropped_count = 0
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

        self._reconnector = Reconnector(
            max_attempts=config.camera.reconnect_attempts, delay=config.camera.reconnect_delay
        )

        self._flow_control_enabled = True
        self._frame_skip_counter = 0
        self._max_frame_skip = BUFFER_SKIP_MAX
        self._consecutive_skips = 0

        self._min_capture_fps = CAPTURE_MIN_FPS_CPU
        self._max_capture_fps = CAPTURE_MAX_FPS_CPU if is_cpu_mode else CAPTURE_TARGET_FPS_GPU
        self._capture_fps_target = CAPTURE_TARGET_FPS_CPU if is_cpu_mode else CAPTURE_TARGET_FPS_GPU

        self._on_frame_dropped: Callable[[int], None] | None = None
        self._on_frame_captured: Callable[[int], None] | None = None

        self._max_consecutive_errors = CAPTURE_MAX_CONSECUTIVE_ERRORS
        self._health_issues: list[str] = []

        self.logger.info(
            "CaptureManager initialized",
            is_cpu_mode=is_cpu_mode,
            capture_fps_target=self._capture_fps_target,
            capture_interval=self._capture_interval,
        )

    def run(self, source: str | None = None) -> None:
        """Main capture loop.

        This method runs the continuous capture loop, handling connection,
        frame reading, flow control, and error recovery.

        Args:
            source: Video source (camera index, file path, or URL).
                If None, uses the source from configuration.

        Example:
            >>> manager.run("0")  # Capture from camera 0
            >>> manager.run("video.mp4")  # Capture from file
        """
        source = source or self.config.camera.source
        self.logger.info(f"Starting capture from: {source}")

        cap = None
        consecutive_errors = 0

        while not self.stop_event.is_set():
            try:
                if not self._should_capture_frame():
                    time.sleep(DEFAULT_SLEEP_SHORT)
                    continue

                if self.pause_event.is_set():
                    time.sleep(SLEEP_PAUSE_CHECK)
                    continue

                cap = self._ensure_connection(cap, source)
                if cap is None:
                    consecutive_errors += 1
                    if consecutive_errors > self._max_consecutive_errors:
                        self._add_health_issue("Persistent connection failure to source")
                        consecutive_errors = 0
                    continue

                consecutive_errors = 0

                ret, frame = cap.read()
                if not ret or frame is None:
                    self.logger.warning("Error reading frame, reconnecting...")
                    cap.release()
                    cap = None
                    continue

                if not validate_frame(
                    frame, min_width=MIN_FRAME_DIMENSION, min_height=MIN_FRAME_DIMENSION
                ):
                    self.logger.debug("Invalid frame, skipping...")
                    continue

                if not self._apply_flow_control():
                    continue

                self._store_frame(frame)

            except Exception as e:
                self.logger.error(f"Capture error: {e}", exc_info=True)
                if cap:
                    cap.release()
                    cap = None
                time.sleep(self._reconnector.delay)

        if cap:
            cap.release()

        self.logger.info("Capture loop terminated")

    def _should_capture_frame(self) -> bool:
        """Checks if it's time to capture a frame.

        Returns:
            bool: True if a frame should be captured.
        """
        current_time = time.time()
        if current_time - self._last_capture_time < self._capture_interval:
            return False
        self._last_capture_time = current_time
        return True

    def _ensure_connection(self, cap, source: str):
        """Ensures the connection is active.

        Args:
            cap: Current capture object or None.
            source: Video source.

        Returns:
            Active capture object or None.
        """
        if cap is not None and cap.isOpened():
            return cap

        self.logger.debug("Connecting to source...")
        cap = self._reconnector.connect(source, self.config.camera)

        if cap is None:
            self.logger.warning("Could not connect, retrying...")
            time.sleep(self._reconnector.delay)
            return None

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            self.logger.warning("Test frame failed, reconnecting...")
            cap.release()
            return None

        return cap

    def _apply_flow_control(self) -> bool:
        """Applies flow control based on buffer state.

        This method checks buffer usage and decides whether to
        accept or skip frames based on current buffer load.

        Returns:
            bool: True if the frame should be processed.

        Example:
            >>> if not self._apply_flow_control():
            ...     # Frame will be skipped
            ...     continue
        """
        if not self._flow_control_enabled:
            return True

        buffer_usage = self.buffer.count / self.buffer.max_size if self.buffer.max_size > 0 else 0

        if buffer_usage > BUFFER_DROP_THRESHOLD:
            return self._handle_buffer_overflow(buffer_usage)

        if buffer_usage < BUFFER_RECOVERY_THRESHOLD:
            self._handle_buffer_recovery()
            return True

        if buffer_usage < BUFFER_USAGE_RECOVERY_BOUNDARY:
            self._consecutive_skips = max(0, self._consecutive_skips - 1)

        return True

    def _handle_buffer_overflow(self, buffer_usage: float) -> bool:
        """Handles buffer overflow condition.

        Args:
            buffer_usage: Current buffer usage (0-1).

        Returns:
            bool: True if the frame should be processed.
        """
        self._frame_skip_counter += 1

        if self._frame_skip_counter < self._max_frame_skip:
            self._dropped_count += 1
            self._consecutive_skips += 1

            if self._consecutive_skips > BUFFER_SKIP_CONSECUTIVE_LIMIT:
                self._add_health_issue(f"Critical buffer: {buffer_usage * 100:.1f}%")
                if self.is_cpu_mode:
                    self._reduce_capture_fps()

            if self._on_frame_dropped:
                self._on_frame_dropped(self._frame_count)
            return False

        self._frame_skip_counter = 0
        self._consecutive_skips = max(0, self._consecutive_skips - 2)
        return True

    def _handle_buffer_recovery(self) -> None:
        """Handles buffer recovery from overflow."""
        self._frame_skip_counter = 0
        self._consecutive_skips = max(0, self._consecutive_skips - 2)

        if self._capture_fps_target < self._max_capture_fps:
            self._capture_fps_target = min(self._max_capture_fps, self._capture_fps_target + 0.5)
            self._capture_interval = 1.0 / self._capture_fps_target

    def _reduce_capture_fps(self) -> None:
        """Reduces capture FPS to relieve buffer pressure."""
        self._capture_fps_target = max(self._min_capture_fps, self._capture_fps_target * 0.9)
        self._capture_interval = 1.0 / self._capture_fps_target

    def _store_frame(self, frame: np.ndarray) -> None:
        """Stores a frame in the buffer.

        Args:
            frame: Frame to store.
        """
        metadata = FrameMetadata(
            timestamp=time.time(),
            frame_number=self._frame_count,
            source_fps=self._current_fps,
            capture_time_ms=0.0,
        )

        if not self.buffer.put(frame, metadata):
            self._dropped_count += 1
            if self._on_frame_dropped:
                self._on_frame_dropped(self._frame_count)
            self.logger.debug(f"Frame {self._frame_count} dropped (buffer full)")
            return

        self._frame_count += 1
        if self._on_frame_captured:
            self._on_frame_captured(self._frame_count)

        self._update_fps()

    def _update_fps(self) -> None:
        """Updates the FPS counter."""
        self._fps_counter += 1
        if time.time() - self._fps_timer >= 1.0:
            self._current_fps = self._fps_counter
            self._fps_counter = 0
            self._fps_timer = time.time()

    def _add_health_issue(self, issue: str) -> None:
        """Records a system health issue.

        Args:
            issue: Description of the health issue.
        """
        timestamp = time.strftime("%H:%M:%S")
        self._health_issues.append(f"[{timestamp}] {issue}")

        if len(self._health_issues) > MAX_HEALTH_ISSUES:
            self._health_issues = self._health_issues[-HEALTH_ISSUES_TRIM_SIZE:]

        self.logger.warning(f"[{timestamp}] {issue}")

    @property
    def frame_count(self) -> int:
        """Number of frames captured."""
        return self._frame_count

    @property
    def dropped_count(self) -> int:
        """Number of frames dropped."""
        return self._dropped_count

    @property
    def current_fps(self) -> float:
        """Current capture FPS."""
        return self._current_fps

    def set_on_frame_dropped(self, callback: Callable[[int], None]) -> None:
        """Sets callback for frame dropped events.

        Args:
            callback: Function called when a frame is dropped.
                Receives the frame count as argument.

        Example:
            >>> def on_dropped(frame_id):
            ...     print(f"Frame {frame_id} was dropped")
            >>> manager.set_on_frame_dropped(on_dropped)
        """
        self._on_frame_dropped = callback

    def set_on_frame_captured(self, callback: Callable[[int], None]) -> None:
        """Sets callback for frame captured events.

        Args:
            callback: Function called when a frame is captured.
                Receives the frame count as argument.

        Example:
            >>> def on_captured(frame_id):
            ...     print(f"Frame {frame_id} captured")
            >>> manager.set_on_frame_captured(on_captured)
        """
        self._on_frame_captured = callback

    def get_stats(self) -> dict:
        """Gets capture statistics.

        Returns:
            dict: Statistics including:
                - frames_captured: Total frames captured
                - frames_dropped: Total frames dropped
                - current_fps: Current capture FPS
                - capture_fps_target: Target capture FPS
                - capture_interval: Capture interval in seconds
                - is_paused: Whether capture is paused
                - is_running: Whether capture is running
                - health_issues: Recent health issues (last 5)

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"FPS: {stats['current_fps']:.1f}")
            >>> print(f"Dropped: {stats['frames_dropped']}")
        """
        return {
            "frames_captured": self._frame_count,
            "frames_dropped": self._dropped_count,
            "current_fps": self._current_fps,
            "capture_fps_target": self._capture_fps_target,
            "capture_interval": self._capture_interval,
            "is_paused": self.pause_event.is_set(),
            "is_running": not self.stop_event.is_set(),
            "health_issues": self._health_issues[-5:],
        }

    def stop(self) -> None:
        """Stops the capture process.

        This method signals the capture loop to stop and releases
        all resources.

        Example:
            >>> manager.stop()
            >>> # Wait for capture thread to finish
            >>> thread.join()
        """
        self.stop_event.set()
        self.logger.info("Capture stopped")

    def pause(self) -> None:
        """Pauses the capture process.

        This method temporarily pauses frame capture while keeping
        resources allocated.

        Example:
            >>> manager.pause()
            >>> # Capture is paused
        """
        self.pause_event.set()
        self.logger.info("Capture paused")

    def resume(self) -> None:
        """Resumes the capture process.

        This method resumes frame capture after a pause.

        Example:
            >>> manager.resume()
            >>> # Capture resumed
        """
        self.pause_event.clear()
        self.logger.info("Capture resumed")
