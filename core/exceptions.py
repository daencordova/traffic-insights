"""
Excepciones personalizadas para el sistema de seguimiento de tráfico.

Esta jerarquía permite un manejo granular de errores y facilita
la recuperación automática en diferentes escenarios.

La estructura sigue un patrón de herencia donde todas las excepciones
del dominio heredan de VehicleCountingError, permitiendo capturar
cualquier error del sistema de forma consistente.
"""

from __future__ import annotations

from typing import Optional, Dict, Any


class VehicleCountingError(Exception):
    """
    Excepción base para todo el sistema de seguimiento de tráfico.

    Todas las excepciones personalizadas heredan de esta clase,
    permitiendo capturar cualquier error del dominio.

    Attributes:
        message: Mensaje descriptivo del error.
        details: Diccionario con información adicional contextual.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Inicializa la excepción base.

        Args:
            message: Mensaje descriptivo del error.
            details: Diccionario con información adicional para depuración.
                Puede incluir: 'component', 'frame_number', 'source', etc.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        """Retorna una representación legible del error."""
        if self.details:
            return f"{self.message} | Detalles: {self.details}"
        return self.message


class ConfigurationError(VehicleCountingError):
    """
    Error relacionado con la configuración del sistema.

    Se lanza cuando hay problemas con:
    - Archivos de configuración mal formados
    - Parámetros inválidos
    - Validaciones de configuración fallidas
    """
    pass


class ValidationError(VehicleCountingError):
    """
    Error en la validación de datos o parámetros.

    Se lanza cuando:
    - Datos de entrada no cumplen especificaciones
    - Parámetros están fuera de rango
    - Estructuras de datos son inválidas
    """
    pass


class ModelLoadError(VehicleCountingError):
    """
    Error al cargar un modelo de machine learning.

    Se lanza cuando:
    - El archivo del modelo no existe
    - El modelo no es compatible
    - Hay problemas de memoria al cargar
    """
    pass


class DetectionError(VehicleCountingError):
    """
    Error en el sistema de detección de objetos.

    Se lanza cuando:
    - La inferencia del modelo falla
    - Los resultados son inválidos
    - Hay problemas de GPU/memoria
    """
    pass


class InferenceError(DetectionError):
    """
    Error durante la inferencia del modelo.

    Específico para fallos en el motor de inferencia
    (ONNX, PyTorch, etc.).
    """
    pass


class TrackingError(VehicleCountingError):
    """
    Error en el sistema de seguimiento (tracking).

    Se lanza cuando:
    - El matching entre detecciones y tracks falla
    - El filtro de Kalman tiene problemas numéricos
    - Hay inconsistencias en los estados de tracks
    """
    pass


class MatchingError(TrackingError):
    """
    Error en el proceso de matching entre detecciones y tracks.

    Específico para fallos en:
    - Cálculo de matrices de coste
    - Asignación óptima (Hungarian algorithm)
    - Matching jerárquico
    """
    pass


class ReIdentificationError(TrackingError):
    """
    Error en el sistema de re-identificación.

    Se lanza cuando:
    - La extracción de features falla
    - El caché de features tiene problemas
    - La comparación de features es inválida
    """
    pass


class PipelineError(VehicleCountingError):
    """
    Error en el pipeline de procesamiento.

    Error genérico para problemas en el flujo principal.
    """
    pass


class FrameProcessingError(PipelineError):
    """
    Error al procesar un frame de video.

    Se lanza cuando:
    - Un frame no se puede procesar correctamente
    - El formato del frame es inválido
    - Hay errores en transformaciones de imagen
    """
    pass


class CaptureError(PipelineError):
    """
    Error en la captura de video.

    Se lanza cuando:
    - La cámara no responde
    - La fuente de video es inaccesible
    - Hay problemas de lectura de frames
    """
    pass


class ResourceError(VehicleCountingError):
    """
    Error al gestionar recursos (memoria, archivos, etc.).

    Error genérico para problemas de recursos del sistema.
    """
    pass


class CacheError(ResourceError):
    """
    Error en el sistema de caché.

    Se lanza cuando:
    - La caché está corrupta
    - Hay problemas de memoria en la caché
    - La política de evicción falla
    """
    pass


class MemoryError(ResourceError):
    """
    Error relacionado con memoria insuficiente.

    Se lanza cuando:
    - No hay suficiente RAM
    - La memoria swap está agotada
    - Hay un memory leak detectado
    """
    pass


class IOError(VehicleCountingError):
    """
    Error de entrada/salida general.

    Se lanza para problemas de I/O que no encajan en categorías específicas.
    """
    pass


class FileNotFoundError(IOError):
    """
    Archivo no encontrado.

    Se lanza cuando un archivo requerido no existe.
    """
    pass


class CameraError(IOError):
    """
    Error relacionado con la cámara o fuente de video.

    Se lanza cuando:
    - La cámara no se puede abrir
    - Los parámetros de la cámara no se pueden configurar
    - La cámara se desconecta inesperadamente
    """
    pass


class CountingError(VehicleCountingError):
    """
    Error en el sistema de conteo.

    Se lanza cuando:
    - Las líneas de conteo no son válidas
    - Hay problemas en la detección de cruces
    - Las estadísticas no se pueden actualizar
    """
    pass


class ConnectionError(VehicleCountingError):
    """
    Error de conexión a servicios externos.

    Se lanza para problemas de red o conexiones a servicios remotos.
    """
    pass


class TimeoutError(VehicleCountingError):
    """
    Error de timeout en operaciones.

    Se lanza cuando una operación excede su tiempo límite.
    """
    pass
