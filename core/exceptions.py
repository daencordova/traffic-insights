"""
Excepciones personalizadas para el sistema de seguimiento de trafico.

Esta jerarquía de excepciones permite un manejo más granular de errores
y facilita la depuración del sistema.
"""

from typing import Optional


class VehicleCountingError(Exception):
    """
    Excepción base para todo el sistema de seguimiento de trafico.

    Todas las excepciones personalizadas heredan de esta clase.
    """
    def __init__(self, message: str, details: Optional[dict] = None):
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


class ModelLoadError(VehicleCountingError):
    """Error al cargar un modelo de machine learning."""
    pass


class FrameProcessingError(VehicleCountingError):
    """Error al procesar un frame de video."""
    pass


class TrackingError(VehicleCountingError):
    """Error en el sistema de tracking."""
    pass


class DetectionError(VehicleCountingError):
    """Error en el sistema de detección."""
    pass


class CountingError(VehicleCountingError):
    """Error en el sistema de conteo."""
    pass


class CameraError(VehicleCountingError):
    """Error relacionado con la cámara o fuente de video."""
    pass


class ResourceError(VehicleCountingError):
    """Error al gestionar recursos (memoria, archivos, etc.)."""
    pass


class PipelineError(VehicleCountingError):
    """Error en el pipeline de procesamiento."""
    pass


class CacheError(VehicleCountingError):
    """Error en el sistema de caché."""
    pass


class FeatureExtractionError(VehicleCountingError):
    """Error al extraer features de una imagen."""
    pass


class MatchingError(VehicleCountingError):
    """Error en el proceso de matching entre detecciones y tracks."""
    pass


class ReIdentificationError(VehicleCountingError):
    """Error en el sistema de re-identificación."""
    pass


class ValidationError(VehicleCountingError):
    """Error en la validación de datos o parámetros."""
    pass
