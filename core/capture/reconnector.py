"""Reconnection handler for video capture.

Provides automatic reconnection logic with retries and delays
for robust video source management.
"""

import time

import cv2

from utils.logger import LoggerMixin


class Reconnector(LoggerMixin):
    """Manages reconnection to video sources.

    This class handles automatic reconnection attempts to video sources
    with configurable retry logic, delays, and camera configuration.

    Features:
        - Configurable retry attempts
        - Delay between attempts
        - Camera parameter configuration
        - Connection status logging
        - Attempt tracking

    Attributes:
        max_attempts: Maximum number of reconnection attempts.
        delay: Delay between attempts in seconds.

    Example:
        >>> reconnector = Reconnector(max_attempts=5, delay=1.0)
        >>>
        >>> # Connect to camera
        >>> cap = reconnector.connect("0", config=camera_config)
        >>> if cap:
        ...     ret, frame = cap.read()
        ...     cap.release()
        >>>
        >>> # Reset attempt counter
        >>> reconnector.reset()
        >>>
        >>> # Get statistics
        >>> stats = reconnector.get_stats()
        >>> print(f"Current attempts: {stats['current_attempts']}")
    """

    def __init__(self, max_attempts: int = 5, delay: float = 1.0):
        """Initializes the reconnector.

        Args:
            max_attempts: Maximum number of reconnection attempts.
            delay: Delay between attempts in seconds.

        Example:
            >>> # Aggressive reconnection with short delays
            >>> reconnector = Reconnector(max_attempts=10, delay=0.5)
            >>>
            >>> # Conservative reconnection with longer delays
            >>> reconnector = Reconnector(max_attempts=3, delay=2.0)
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self._attempts = 0

        self.logger.info("Reconnector initialized", max_attempts=max_attempts, delay=delay)

    def connect(self, source: str | int, config=None) -> cv2.VideoCapture | None:
        """Attempts to connect to the source with retries.

        This method tries to open the video source and applies configuration
        parameters if successful. It retries up to max_attempts times.

        Args:
            source: Video source (camera index as int/string or file path).
            config: Camera configuration object (optional).

        Returns:
            Optional[cv2.VideoCapture]: Connected capture object or None.

        Example:
            >>> # Connect to camera 0 with configuration
            >>> cap = reconnector.connect(0, config=camera_config)
            >>>
            >>> # Connect to video file
            >>> cap = reconnector.connect("video.mp4")
            >>>
            >>> if cap:
            ...     process_video(cap)
            ...     cap.release()
        """
        self._attempts = 0

        while self._attempts < self.max_attempts:
            try:
                self._attempts += 1

                if isinstance(source, str) and source.isdigit():
                    cap = cv2.VideoCapture(int(source))
                else:
                    cap = cv2.VideoCapture(source)

                if cap.isOpened():
                    self._configure_capture(cap, config)
                    self.logger.info(
                        "Connected successfully", source=source, attempts=self._attempts
                    )
                    self._attempts = 0
                    return cap

                self.logger.warning(
                    "Connection attempt failed",
                    source=source,
                    attempt=self._attempts,
                    max_attempts=self.max_attempts,
                )

                if self._attempts < self.max_attempts:
                    time.sleep(self.delay)

            except Exception as e:
                self.logger.warning(
                    "Connection error", source=source, attempt=self._attempts, error=str(e)
                )
                if self._attempts < self.max_attempts:
                    time.sleep(self.delay)

        self.logger.error(
            "Could not connect after multiple attempts",
            source=source,
            attempts=self.max_attempts,
        )
        return None

    def _configure_capture(self, cap: cv2.VideoCapture, config) -> None:
        """Configures the capture with desired parameters.

        This method applies camera configuration including resolution,
        buffer size, and codec settings.

        Args:
            cap: Capture object to configure.
            config: Camera configuration object.

        Note:
            The configuration object should have width, height attributes
            and may have other camera-specific settings.
        """
        if config is None:
            return

        if hasattr(config, "width") and config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)

        if hasattr(config, "height") and config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def reset(self) -> None:
        """Resets the attempt counter.

        This method resets the internal attempt counter to zero,
        useful for starting a fresh connection sequence.

        Example:
            >>> reconnector.reset()
            >>> # Now connect with fresh attempt counter
            >>> cap = reconnector.connect("0")
        """
        self._attempts = 0
        self.logger.debug("Reconnector reset")

    def get_stats(self) -> dict:
        """Gets statistics for the reconnector.

        Returns:
            dict: Statistics including:
                - max_attempts: Maximum connection attempts
                - delay: Delay between attempts
                - current_attempts: Current attempt count

        Example:
            >>> stats = reconnector.get_stats()
            >>> print(f"Max attempts: {stats['max_attempts']}")
            >>> print(f"Current attempts: {stats['current_attempts']}")
        """
        return {
            "max_attempts": self.max_attempts,
            "delay": self.delay,
            "current_attempts": self._attempts,
        }
