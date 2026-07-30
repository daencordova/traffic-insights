"""Fábrica para crear detectores de objetos.

Proporciona una interfaz unificada para crear diferentes tipos de detectores
según la configuración y disponibilidad del hardware.
"""


from core.detector.base import YOLODetector
from core.detector.config import DetectorConfig
from core.detector.optimized import OptimizedYOLODetector
from utils.logger import LoggerMixin


class DetectorFactory(LoggerMixin):
    """Fábrica de detectores de objetos.

    Crea detectores según la configuración y disponibilidad.
    Prioriza versiones optimizadas cuando están disponibles.

    Example:
        >>> detector = DetectorFactory.create_best_available()
        >>> detections = detector.detect(frame)
    """

    @staticmethod
    def create(
        config: DetectorConfig | None = None,
        force_optimized: bool = False,
        force_standard: bool = False,
    ) -> YOLODetector:
        """Crea un detector de objetos.

        Args:
            config: Configuración del detector (opcional).
                Si es None, usa la configuración global.
            force_optimized: Forzar versión optimizada para CPU.
                Lanza excepción si no está disponible.
            force_standard: Forzar versión estándar (PyTorch).
                Ignora cualquier optimización disponible.

        Returns:
            YOLODetector: Detector creado.

        Note:
            El orden de prioridad es:
            1. force_standard -> YOLODetector
            2. force_optimized -> OptimizedYOLODetector
            3. config.use_optimized -> OptimizedYOLODetector (fallback a YOLODetector)
            4. Default -> YOLODetector
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        if force_standard:
            return YOLODetector(config)

        if force_optimized:
            try:
                return OptimizedYOLODetector(config)
            except Exception as e:
                logger = LoggerMixin().logger
                logger.warning(f"Error creando detector optimizado: {e}")
                return YOLODetector(config)

        if config.use_optimized:
            try:
                return OptimizedYOLODetector(config)
            except Exception as e:
                logger = LoggerMixin().logger
                logger.warning(f"Detector optimizado no disponible: {e}")
                return YOLODetector(config)

        return YOLODetector(config)

    @staticmethod
    def create_optimized(config: DetectorConfig | None = None) -> OptimizedYOLODetector:
        """Crea un detector optimizado para CPU.

        Args:
            config: Configuración del detector (opcional).

        Returns:
            OptimizedYOLODetector: Detector optimizado con ONNX y Numba.

        Raises:
            RuntimeError: Si no se puede crear el detector optimizado.

        Example:
            >>> detector = DetectorFactory.create_optimized()
            >>> # Usa ONNX Runtime y Numba para máxima velocidad en CPU
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        try:
            return OptimizedYOLODetector(config)
        except Exception as e:
            raise RuntimeError(f"No se pudo crear detector optimizado: {e}")

    @staticmethod
    def create_standard(config: DetectorConfig | None = None) -> YOLODetector:
        """Crea un detector estándar.

        Args:
            config: Configuración del detector (opcional).

        Returns:
            YOLODetector: Detector estándar con PyTorch.

        Example:
            >>> detector = DetectorFactory.create_standard()
            >>> # Usa PyTorch para inferencia
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        return YOLODetector(config)

    @staticmethod
    def create_best_available(config: DetectorConfig | None = None) -> YOLODetector:
        """Crea el mejor detector disponible según el hardware.

        Args:
            config: Configuración del detector (opcional).

        Returns:
            YOLODetector: Mejor detector disponible.

        Note:
            Prioriza:
            1. OptimizedYOLODetector (si está disponible)
            2. YOLODetector (fallback)

        Example:
            >>> detector = DetectorFactory.create_best_available()
            >>> # Usa el detector más rápido disponible en el hardware actual
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        try:
            detector = OptimizedYOLODetector(config)
            logger = LoggerMixin().logger
            logger.info("✅ Detector optimizado creado")
            return detector
        except Exception as e:
            logger = LoggerMixin().logger
            logger.warning(f"Detector optimizado no disponible: {e}")

        logger = LoggerMixin().logger
        logger.info("📦 Usando detector estándar")
        return YOLODetector(config)
