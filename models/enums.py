"""
Enums centralizados del sistema
"""

from enum import Enum, auto
from typing import List


class TrackStatus(Enum):
    """Estados posibles de un track"""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DEAD = "dead"

    @classmethod
    def active_statuses(cls) -> List["TrackStatus"]:
        """Retorna los estados activos"""
        return [cls.TENTATIVE, cls.CONFIRMED, cls.LOST]

    @classmethod
    def terminal_statuses(cls) -> List["TrackStatus"]:
        """Retorna los estados terminales"""
        return [cls.DEAD]

    def is_active(self) -> bool:
        """Verifica si el estado es activo"""
        return self in self.active_statuses()

    def is_terminal(self) -> bool:
        """Verifica si el estado es terminal"""
        return self in self.terminal_statuses()


class DetectionStatus(Enum):
    """Estado de una detección"""
    VALID = auto()
    INVALID = auto()
    LOW_CONFIDENCE = auto()
    SMALL_AREA = auto()
    OUTSIDE_ROI = auto()
    DUPLICATE = auto()

    def is_valid(self) -> bool:
        """Verifica si la detección es válida"""
        return self == DetectionStatus.VALID


class DeviceType(Enum):
    """Tipos de dispositivos soportados"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

    @classmethod
    def gpu_devices(cls) -> List["DeviceType"]:
        """Retorna los dispositivos GPU"""
        return [cls.CUDA, cls.MPS]

    def is_gpu(self) -> bool:
        """Verifica si es un dispositivo GPU"""
        return self in self.gpu_devices()


class TrackerType(Enum):
    """Tipos de tracker disponibles"""
    CENTROID = "centroid"
    DEEPSORT = "deepsort"
    HYBRID = "hybrid"


class MotionModel(Enum):
    """Modelos de movimiento para Kalman"""
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"

    @classmethod
    def default(cls) -> "MotionModel":
        """Retorna el modelo por defecto"""
        return cls.CONSTANT_VELOCITY


class ExportFormat(Enum):
    """Formatos de exportación soportados"""
    JSON = "json"
    CSV = "csv"
    BOTH = "both"

    @classmethod
    def values(cls) -> List[str]:
        """Retorna los valores como strings"""
        return [e.value for e in cls]


class ImageFormat(Enum):
    """Formatos de imagen soportados"""
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"

    @classmethod
    def values(cls) -> List[str]:
        """Retorna los valores como strings"""
        return [e.value for e in cls]


class CongestionLevel(Enum):
    """Niveles de congestión de tráfico"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

    @classmethod
    def from_occupancy(cls, occupancy: float) -> "CongestionLevel":
        """Obtiene nivel de congestión desde ocupación"""
        from core.constants import CONGESTION_LOW, CONGESTION_MEDIUM, CONGESTION_HIGH

        if occupancy < CONGESTION_LOW:
            return cls.LOW
        elif occupancy < CONGESTION_MEDIUM:
            return cls.MEDIUM
        elif occupancy < CONGESTION_HIGH:
            return cls.HIGH
        else:
            return cls.CRITICAL


class DashboardPosition(Enum):
    """Posiciones del dashboard en pantalla"""
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"

    @classmethod
    def default(cls) -> "DashboardPosition":
        """Retorna la posición por defecto"""
        return cls.TOP_LEFT


class LogLevel(Enum):
    """Niveles de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_int(self) -> int:
        """Convierte a valor entero de logging"""
        from core.constants import LOG_LEVELS
        return LOG_LEVELS.get(self.value, 20)


class LaneType(Enum):
    """Tipos de carriles"""
    STANDARD = "standard"
    BUS = "bus"
    BIKE = "bike"
    PEDESTRIAN = "pedestrian"
    EMERGENCY = "emergency"
    TURNING = "turning"


class LaneDirection(Enum):
    """Direcciones de carriles"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    BIDIRECTIONAL = "bidirectional"

    @classmethod
    def default(cls) -> "LaneDirection":
        """Retorna la dirección por defecto"""
        return cls.DOWN


class ValidationResult(Enum):
    """Resultados de validación"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"

    def is_success(self) -> bool:
        """Verifica si el resultado es exitoso"""
        return self in [ValidationResult.PASS, ValidationResult.WARNING]
