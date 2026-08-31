"""Specific configuration for the object detector.

This module provides a configuration dataclass for the object detector
with validation and integration with the global system configuration.
"""

from dataclasses import dataclass, field

from config.manager import config


@dataclass
class DetectorConfig:
    """Configuration for the object detector.

    This class manages all configuration parameters for the object
    detection system, including model settings, thresholds, and
    inference options.

    Attributes:
        model_path: Path to the YOLO model file.
        confidence_threshold: Confidence threshold (0-1) for detections.
        iou_threshold: IoU threshold for NMS (0-1).
        vehicle_classes: List of class IDs to detect.
        device: Inference device ('cpu', 'cuda', 'mps', 'auto').
        use_half_precision: Whether to use half precision (FP16).
        use_onnx: Whether to use ONNX model format.
        imgsz: Image size for inference (must be multiple of 32).
        max_det: Maximum number of detections per image.
        use_optimized: Whether to use CPU-optimized version.

    Example:
        >>> # Create configuration with custom values
        >>> config = DetectorConfig(
        ...     model_path="yolov8m.pt",
        ...     confidence_threshold=0.5,
        ...     imgsz=640,
        ...     max_det=50,
        ...     use_optimized=True,
        ... )
        >>>
        >>> # Validate configuration
        >>> config.validate()
        True
        >>>
        >>> # Create from global configuration
        >>> config = DetectorConfig.from_global_config()
        >>> print(f"Model: {config.model_path}")
        >>> print(f"Device: {config.device}")
    """

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    vehicle_classes: list[int] = field(default_factory=lambda: [2, 3, 5, 7])
    device: str = "auto"
    use_half_precision: bool = False
    use_onnx: bool = False
    imgsz: int = 320
    max_det: int = 10
    use_optimized: bool = True

    def validate(self) -> bool:
        """Validates the configuration parameters.

        This method checks that all configuration values are within
        valid ranges and meet system requirements.

        Returns:
            bool: True if configuration is valid.

        Raises:
            ValueError: If any parameter is out of range or invalid.

        Example:
            >>> try:
            ...     config.validate()
            ...     print("Configuration is valid")
            ... except ValueError as e:
            ...     print(f"Invalid configuration: {e}")
        """
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"Confidence threshold must be between 0 and 1: {self.confidence_threshold}"
            )

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"IoU threshold must be between 0 and 1: {self.iou_threshold}")

        if self.imgsz not in [320, 416, 512, 640, 768, 832, 1024]:
            raise ValueError(f"imgsz must be a multiple of 32: {self.imgsz}")

        return True

    @classmethod
    def from_global_config(cls) -> "DetectorConfig":
        """Creates detector configuration from the global system configuration.

        This factory method extracts detector-specific settings from the
        global configuration object, providing a convenient way to create
        detector configuration that matches system-wide settings.

        Returns:
            DetectorConfig: Detector configuration based on global settings.

        Example:
            >>> # Use global configuration
            >>> detector_config = DetectorConfig.from_global_config()
            >>> if detector_config.use_optimized:
            ...     detector = OptimizedYOLODetector(detector_config)
            ... else:
            ...     detector = YOLODetector(detector_config)
        """
        return cls(
            model_path=config.model.model_path,
            confidence_threshold=config.model.confidence_threshold,
            iou_threshold=config.model.iou_threshold,
            vehicle_classes=config.model.vehicle_classes,
            device=config.model.device,
            use_half_precision=config.model.use_half_precision,
            use_onnx=config.model.use_onnx,
            imgsz=config.model.imgsz,
            max_det=config.model.max_det,
            use_optimized=getattr(config.optimization, "use_optimized_detector", True),
        )
