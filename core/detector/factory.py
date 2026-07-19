"""
Fábrica para crear detectores de objetos.

Proporciona una interfaz unificada para crear diferentes tipos de detectores.
"""

from typing import Optional

from core.detector.base import YOLODetector
from core.detector.optimized import OptimizedYOLODetector
from core.detector.config import DetectorConfig
from utils.logger import LoggerMixin


class DetectorFactory(LoggerMixin):
    """
    Fábrica de detectores de objetos.

    Crea detectores según la configuración y disponibilidad.
    """

    @staticmethod
    def create(
        config: Optional[DetectorConfig] = None,
        force_optimized: bool = False,
        force_standard: bool = False
    ) -> YOLODetector:
        """
        Crea un detector de objetos.

        Args:
            config: Configuración del detector (opcional)
            force_optimized: Forzar versión optimizada para CPU
            force_standard: Forzar versión estándar

        Returns:
            YOLODetector: Detector creado
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
    def create_optimized(
        config: Optional[DetectorConfig] = None
    ) -> OptimizedYOLODetector:
        """
        Crea un detector optimizado para CPU.

        Args:
            config: Configuración del detector (opcional)

        Returns:
            OptimizedYOLODetector: Detector optimizado

        Raises:
            RuntimeError: Si no se puede crear el detector optimizado
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        try:
            return OptimizedYOLODetector(config)
        except Exception as e:
            raise RuntimeError(f"No se pudo crear detector optimizado: {e}")

    @staticmethod
    def create_standard(
        config: Optional[DetectorConfig] = None
    ) -> YOLODetector:
        """
        Crea un detector estándar.

        Args:
            config: Configuración del detector (opcional)

        Returns:
            YOLODetector: Detector estándar
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        return YOLODetector(config)

    @staticmethod
    def create_best_available(
        config: Optional[DetectorConfig] = None
    ) -> YOLODetector:
        """
        Crea el mejor detector disponible según el hardware.

        Args:
            config: Configuración del detector (opcional)

        Returns:
            YOLODetector: Mejor detector disponible
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
