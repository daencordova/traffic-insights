"""Factory for creating object detectors.

Provides a unified interface for creating different types of detectors
based on configuration and hardware availability.

This module abstracts the detector creation process, allowing the system
to choose the optimal detector implementation based on the current
hardware and configuration settings.
"""

from core.detector.base import YOLODetector
from core.detector.config import DetectorConfig
from core.detector.optimized import OptimizedYOLODetector
from utils.logger import LoggerMixin


class DetectorFactory(LoggerMixin):
    """Factory for object detectors.

    Creates detectors based on configuration and availability.
    Prioritizes optimized versions when available.

    Features:
        - Automatic detector selection based on hardware
        - Support for optimized CPU detectors (ONNX + Numba)
        - Support for standard PyTorch detectors
        - Configurable creation with fallback options

    Example:
        >>> # Create the best available detector
        >>> detector = DetectorFactory.create_best_available()
        >>> detections = detector.detect(frame)
        >>>
        >>> # Force optimized detector
        >>> detector = DetectorFactory.create_optimized()
        >>>
        >>> # Force standard detector
        >>> detector = DetectorFactory.create_standard()
        >>>
        >>> # Create with custom configuration
        >>> config = DetectorConfig(confidence_threshold=0.5)
        >>> detector = DetectorFactory.create(config, force_optimized=True)
    """

    @staticmethod
    def create(
        config: DetectorConfig | None = None,
        force_optimized: bool = False,
        force_standard: bool = False,
    ) -> YOLODetector:
        """Creates an object detector.

        This method provides flexible detector creation with priority rules
        for choosing between optimized and standard implementations.

        Args:
            config: Detector configuration (optional).
                If None, uses global configuration.
            force_optimized: Force optimized CPU version.
                Falls back to standard if not available.
            force_standard: Force standard version (PyTorch).
                Ignores any available optimizations.

        Returns:
            YOLODetector: Created detector instance.

        Note:
            Priority order:
            1. force_standard -> YOLODetector
            2. force_optimized -> OptimizedYOLODetector
            3. config.use_optimized -> OptimizedYOLODetector (fallback to YOLODetector)
            4. Default -> YOLODetector

        Example:
            >>> # Create with automatic selection
            >>> detector = DetectorFactory.create()
            >>>
            >>> # Force optimized version
            >>> detector = DetectorFactory.create(force_optimized=True)
            >>>
            >>> # Force standard version
            >>> detector = DetectorFactory.create(force_standard=True)
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
                logger.warning(f"Error creating optimized detector: {e}")
                return YOLODetector(config)

        if config.use_optimized:
            try:
                return OptimizedYOLODetector(config)
            except Exception as e:
                logger = LoggerMixin().logger
                logger.warning(f"Optimized detector not available: {e}")
                return YOLODetector(config)

        return YOLODetector(config)

    @staticmethod
    def create_optimized(config: DetectorConfig | None = None) -> OptimizedYOLODetector:
        """Creates a CPU-optimized detector.

        This method creates a detector using ONNX Runtime and Numba
        optimizations for maximum performance on CPU.

        Args:
            config: Detector configuration (optional).
                If None, uses global configuration.

        Returns:
            OptimizedYOLODetector: Optimized detector with ONNX and Numba.

        Raises:
            RuntimeError: If the optimized detector cannot be created.

        Example:
            >>> # Create optimized detector
            >>> detector = DetectorFactory.create_optimized()
            >>> # Uses ONNX Runtime and Numba for maximum CPU speed
            >>>
            >>> # With custom configuration
            >>> config = DetectorConfig(imgsz=640)
            >>> detector = DetectorFactory.create_optimized(config)
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        try:
            return OptimizedYOLODetector(config)
        except Exception as e:
            raise RuntimeError(f"Could not create optimized detector: {e}") from e

    @staticmethod
    def create_standard(config: DetectorConfig | None = None) -> YOLODetector:
        """Creates a standard detector.

        This method creates a detector using PyTorch for inference,
        which is the most compatible option.

        Args:
            config: Detector configuration (optional).
                If None, uses global configuration.

        Returns:
            YOLODetector: Standard detector with PyTorch.

        Example:
            >>> # Create standard detector
            >>> detector = DetectorFactory.create_standard()
            >>> # Uses PyTorch for inference
            >>>
            >>> # With custom configuration
            >>> config = DetectorConfig(model_path="yolov8m.pt")
            >>> detector = DetectorFactory.create_standard(config)
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        return YOLODetector(config)

    @staticmethod
    def create_best_available(config: DetectorConfig | None = None) -> YOLODetector:
        """Creates the best available detector based on hardware.

        This method automatically selects the optimal detector implementation
        for the current hardware configuration.

        Args:
            config: Detector configuration (optional).
                If None, uses global configuration.

        Returns:
            YOLODetector: Best available detector.

        Note:
            Priority order:
            1. OptimizedYOLODetector (if available)
            2. YOLODetector (fallback)

        Example:
            >>> # Create best detector for current hardware
            >>> detector = DetectorFactory.create_best_available()
            >>> # Uses the fastest available detector on current hardware
            >>>
            >>> # Check which detector was created
            >>> if isinstance(detector, OptimizedYOLODetector):
            ...     print("Using optimized detector")
            ... else:
            ...     print("Using standard detector")
        """
        if config is None:
            config = DetectorConfig.from_global_config()

        try:
            detector = OptimizedYOLODetector(config)
            logger = LoggerMixin().logger
            logger.info("Optimized detector created")
            return detector
        except Exception as e:
            logger = LoggerMixin().logger
            logger.warning(f"Optimized detector not available: {e}")

        logger = LoggerMixin().logger
        logger.info("Using standard detector")
        return YOLODetector(config)
