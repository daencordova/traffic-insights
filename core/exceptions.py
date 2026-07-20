"""
Excepciones personalizadas para el sistema de seguimiento de tráfico.

Esta jerarquía permite un manejo granular de errores y facilita
la recuperación automática en diferentes escenarios.
"""

from typing import Optional, Dict, Any


class VehicleCountingError(Exception):
    """
    Excepción base para todo el sistema de seguimiento de tráfico.

    Todas las excepciones personalizadas heredan de esta clase,
    permitiendo capturar cualquier error del dominio.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Detalles: {self.details}"
        return self.message


class ConfigurationError(VehicleCountingError):
    """Error relacionado con la configuración del sistema."""
    pass


class ValidationError(VehicleCountingError):
    """Error en la validación de datos o parámetros."""
    pass


class ModelLoadError(VehicleCountingError):
    """Error al cargar un modelo de machine learning."""
    pass


class DetectionError(VehicleCountingError):
    """Error en el sistema de detección de objetos."""
    pass


class InferenceError(DetectionError):
    """Error durante la inferencia del modelo."""
    pass


class TrackingError(VehicleCountingError):
    """Error en el sistema de seguimiento (tracking)."""
    pass


class MatchingError(TrackingError):
    """Error en el proceso de matching entre detecciones y tracks."""
    pass


class ReIdentificationError(TrackingError):
    """Error en el sistema de re-identificación."""
    pass


class PipelineError(VehicleCountingError):
    """Error en el pipeline de procesamiento."""
    pass


class FrameProcessingError(PipelineError):
    """Error al procesar un frame de video."""
    pass


class CaptureError(PipelineError):
    """Error en la captura de video."""
    pass


class ResourceError(VehicleCountingError):
    """Error al gestionar recursos (memoria, archivos, etc.)."""
    pass


class CacheError(ResourceError):
    """Error en el sistema de caché."""
    pass


class MemoryError(ResourceError):
    """Error relacionado con memoria insuficiente."""
    pass


class IOError(VehicleCountingError):
    """Error de entrada/salida general."""
    pass


class FileNotFoundError(IOError):
    """Archivo no encontrado."""
    pass


class CameraError(IOError):
    """Error relacionado con la cámara o fuente de video."""
    pass


class CountingError(VehicleCountingError):
    """Error en el sistema de conteo."""
    pass


class ConnectionError(VehicleCountingError):
    pass


class TimeoutError(VehicleCountingError):
    pass
