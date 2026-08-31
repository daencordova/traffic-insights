"""Centralized system enums.

This module contains all enumerations used in the system,
organized by application domain.

Example:
    >>> from core.enums import TrackStatus, DeviceType, CongestionLevel
    >>>
    >>> # Track status
    >>> status = TrackStatus.CONFIRMED
    >>> if status.is_active():
    ...     print("Track is active")
    >>>
    >>> # Device selection
    >>> device = DeviceType.AUTO
    >>> if device.is_gpu():
    ...     print("Using GPU")
    >>>
    >>> # Congestion level from occupancy
    >>> level = CongestionLevel.from_occupancy(0.75)
    >>> print(f"Traffic level: {level.value}")
"""

from enum import Enum, auto

from core.constants import CONGESTION_HIGH, CONGESTION_LOW, CONGESTION_MEDIUM, LOG_LEVELS


class TrackStatus(Enum):
    """Possible states of a track.

    Attributes:
        TENTATIVE: Initial state, track is being confirmed.
        CONFIRMED: Track is confirmed and reliable.
        LOST: Track was lost but may be recovered.
        DEAD: Track is dead and should be removed.

    Example:
        >>> status = TrackStatus.CONFIRMED
        >>> if status.is_active():
        ...     print("Track is active")
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DEAD = "dead"

    @classmethod
    def active_statuses(cls) -> list["TrackStatus"]:
        """Returns active statuses.

        Returns:
            list[TrackStatus]: List of active statuses.

        Example:
            >>> active = TrackStatus.active_statuses()
            >>> print([s.value for s in active])
            ['tentative', 'confirmed', 'lost']
        """
        return [cls.TENTATIVE, cls.CONFIRMED, cls.LOST]

    @classmethod
    def terminal_statuses(cls) -> list["TrackStatus"]:
        """Returns terminal statuses.

        Returns:
            list[TrackStatus]: List of terminal statuses.

        Example:
            >>> terminal = TrackStatus.terminal_statuses()
            >>> print([s.value for s in terminal])
            ['dead']
        """
        return [cls.DEAD]

    def is_active(self) -> bool:
        """Checks if the status is active.

        Returns:
            bool: True if status is active.

        Example:
            >>> TrackStatus.CONFIRMED.is_active()
            True
            >>> TrackStatus.DEAD.is_active()
            False
        """
        return self in self.active_statuses()

    def is_terminal(self) -> bool:
        """Checks if the status is terminal.

        Returns:
            bool: True if status is terminal.

        Example:
            >>> TrackStatus.DEAD.is_terminal()
            True
            >>> TrackStatus.CONFIRMED.is_terminal()
            False
        """
        return self in self.terminal_statuses()


class DetectionStatus(Enum):
    """Status of a detection.

    Attributes:
        VALID: Detection is valid.
        INVALID: Detection is invalid.
        LOW_CONFIDENCE: Detection has low confidence.
        SMALL_AREA: Detection has small area.
        OUTSIDE_ROI: Detection is outside ROI.
        DUPLICATE: Detection is a duplicate.

    Example:
        >>> status = DetectionStatus.VALID
        >>> if status.is_valid():
        ...     print("Detection is valid")
    """

    VALID = auto()
    INVALID = auto()
    LOW_CONFIDENCE = auto()
    SMALL_AREA = auto()
    OUTSIDE_ROI = auto()
    DUPLICATE = auto()

    def is_valid(self) -> bool:
        """Checks if the detection is valid.

        Returns:
            bool: True if detection is valid.
        """
        return self == DetectionStatus.VALID


class DeviceType(Enum):
    """Supported device types.

    Attributes:
        CPU: CPU device.
        CUDA: NVIDIA GPU (CUDA).
        MPS: Apple Metal Performance Shaders.
        AUTO: Auto-select best available device.

    Example:
        >>> device = DeviceType.AUTO
        >>> if device.is_gpu():
        ...     print("GPU device selected")
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

    @classmethod
    def gpu_devices(cls) -> list["DeviceType"]:
        """Returns GPU devices.

        Returns:
            list[DeviceType]: List of GPU device types.

        Example:
            >>> gpu_devices = DeviceType.gpu_devices()
            >>> print([d.value for d in gpu_devices])
            ['cuda', 'mps']
        """
        return [cls.CUDA, cls.MPS]

    def is_gpu(self) -> bool:
        """Checks if it's a GPU device.

        Returns:
            bool: True if device is GPU.

        Example:
            >>> DeviceType.CUDA.is_gpu()
            True
            >>> DeviceType.CPU.is_gpu()
            False
        """
        return self in self.gpu_devices()


class TrackerType(Enum):
    """Available tracker types.

    Attributes:
        CENTROID: Centroid-based tracker (simple).
        DEEPSORT: DeepSORT tracker (deep learning).
        HYBRID: Hybrid tracker combining multiple methods.

    Example:
        >>> tracker = TrackerType.HYBRID
        >>> print(f"Using {tracker.value} tracker")
    """

    CENTROID = "centroid"
    DEEPSORT = "deepsort"
    HYBRID = "hybrid"


class MotionModel(Enum):
    """Motion models for Kalman filter.

    Attributes:
        CONSTANT_VELOCITY: Constant velocity model.
        CONSTANT_ACCELERATION: Constant acceleration model.

    Example:
        >>> model = MotionModel.CONSTANT_VELOCITY
        >>> print(f"Using {model.value} model")
    """

    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"

    @classmethod
    def default(cls) -> "MotionModel":
        """Returns the default motion model.

        Returns:
            MotionModel: Default motion model.

        Example:
            >>> default = MotionModel.default()
            >>> print(default.value)
            'constant_velocity'
        """
        return cls.CONSTANT_VELOCITY


class ExportFormat(Enum):
    """Supported export formats.

    Attributes:
        JSON: JSON format.
        CSV: CSV format.
        BOTH: Both JSON and CSV.

    Example:
        >>> format = ExportFormat.JSON
        >>> print(format.value)  # 'json'
    """

    JSON = "json"
    CSV = "csv"
    BOTH = "both"

    @classmethod
    def values(cls) -> list[str]:
        """Returns values as strings.

        Returns:
            list[str]: List of format values.

        Example:
            >>> formats = ExportFormat.values()
            >>> print(formats)  # ['json', 'csv', 'both']
        """
        return [e.value for e in cls]


class ImageFormat(Enum):
    """Supported image formats.

    Attributes:
        JPG: JPEG format.
        PNG: PNG format.
        BMP: BMP format.
        TIFF: TIFF format.

    Example:
        >>> format = ImageFormat.PNG
        >>> print(f"Saving as {format.value}")
    """

    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"

    @classmethod
    def values(cls) -> list[str]:
        """Returns values as strings.

        Returns:
            list[str]: List of format values.

        Example:
            >>> formats = ImageFormat.values()
            >>> print(formats)  # ['jpg', 'png', 'bmp', 'tiff']
        """
        return [e.value for e in cls]


class CongestionLevel(Enum):
    """Traffic congestion levels.

    Attributes:
        LOW: Low congestion.
        MEDIUM: Medium congestion.
        HIGH: High congestion.
        CRITICAL: Critical congestion.
        UNKNOWN: Unknown congestion level.

    Example:
        >>> level = CongestionLevel.from_occupancy(0.75)
        >>> print(f"Congestion: {level.value}")  # 'high'
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

    @classmethod
    def from_occupancy(cls, occupancy: float) -> "CongestionLevel":
        """Gets congestion level from occupancy.

        Args:
            occupancy: Occupancy percentage (0-1).

        Returns:
            CongestionLevel: Corresponding congestion level.

        Example:
            >>> level = CongestionLevel.from_occupancy(0.3)
            >>> print(level.value)  # 'low'
            >>>
            >>> level = CongestionLevel.from_occupancy(0.85)
            >>> print(level.value)  # 'critical'
        """
        if occupancy < CONGESTION_LOW:
            return cls.LOW
        if occupancy < CONGESTION_MEDIUM:
            return cls.MEDIUM
        if occupancy < CONGESTION_HIGH:
            return cls.HIGH
        return cls.CRITICAL


class DashboardPosition(Enum):
    """Dashboard positions on screen.

    Attributes:
        TOP_LEFT: Top-left corner.
        TOP_RIGHT: Top-right corner.
        BOTTOM_LEFT: Bottom-left corner.
        BOTTOM_RIGHT: Bottom-right corner.

    Example:
        >>> position = DashboardPosition.TOP_LEFT
        >>> print(f"Dashboard at {position.value}")
    """

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"

    @classmethod
    def default(cls) -> "DashboardPosition":
        """Returns the default dashboard position.

        Returns:
            DashboardPosition: Default position.

        Example:
            >>> default = DashboardPosition.default()
            >>> print(default.value)  # 'top-left'
        """
        return cls.TOP_LEFT


class LogLevel(Enum):
    """Logging levels.

    Attributes:
        DEBUG: Debug level.
        INFO: Info level.
        WARNING: Warning level.
        ERROR: Error level.
        CRITICAL: Critical level.

    Example:
        >>> level = LogLevel.INFO
        >>> print(f"Log level: {level.value}")
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_int(self) -> int:
        """Converts to integer logging value.

        Returns:
            int: Logging level integer value.

        Example:
            >>> LogLevel.INFO.to_int()
            20
            >>> LogLevel.ERROR.to_int()
            40
        """
        return LOG_LEVELS.get(self.value, 20)


class LaneType(Enum):
    """Lane types.

    Attributes:
        STANDARD: Standard lane.
        BUS: Bus lane.
        BIKE: Bicycle lane.
        PEDESTRIAN: Pedestrian lane.
        EMERGENCY: Emergency lane.
        TURNING: Turning lane.

    Example:
        >>> lane = LaneType.STANDARD
        >>> print(f"Lane type: {lane.value}")
    """

    STANDARD = "standard"
    BUS = "bus"
    BIKE = "bike"
    PEDESTRIAN = "pedestrian"
    EMERGENCY = "emergency"
    TURNING = "turning"


class LaneDirection(Enum):
    """Lane directions.

    Attributes:
        UP: Up direction.
        DOWN: Down direction.
        LEFT: Left direction.
        RIGHT: Right direction.
        BIDIRECTIONAL: Bidirectional.

    Example:
        >>> direction = LaneDirection.DOWN
        >>> print(f"Lane direction: {direction.value}")
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    BIDIRECTIONAL = "bidirectional"

    @classmethod
    def default(cls) -> "LaneDirection":
        """Returns the default lane direction.

        Returns:
            LaneDirection: Default direction.

        Example:
            >>> default = LaneDirection.default()
            >>> print(default.value)  # 'down'
        """
        return cls.DOWN


class ValidationResult(Enum):
    """Validation results.

    Attributes:
        PASS: Validation passed.
        FAIL: Validation failed.
        WARNING: Validation warning.
        ERROR: Validation error.

    Example:
        >>> result = ValidationResult.PASS
        >>> if result.is_success():
        ...     print("Validation succeeded")
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"

    def is_success(self) -> bool:
        """Checks if the result is successful.

        Returns:
            bool: True if result is PASS or WARNING.

        Example:
            >>> ValidationResult.PASS.is_success()
            True
            >>> ValidationResult.FAIL.is_success()
            False
        """
        return self in [ValidationResult.PASS, ValidationResult.WARNING]
