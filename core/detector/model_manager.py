"""Model manager for YOLO detector.

Handles loading, management, and switching between different model
formats (PyTorch and ONNX).

This module provides a unified interface for managing YOLO models
in different formats, allowing seamless switching between PyTorch
and ONNX Runtime for inference.
"""

import os
from pathlib import Path

from ultralytics import YOLO

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from utils.logger import LoggerMixin


class ModelLoadError(Exception):
    """Exception raised when a model fails to load.

    This exception is raised when there are issues loading a model,
    such as file not found, format incompatibility, or other loading errors.

    Example:
        >>> try:
        ...     manager.load_pytorch()
        ... except ModelLoadError as e:
        ...     print(f"Failed to load model: {e}")
    """


class ModelManager(LoggerMixin):
    """Model manager for YOLO.

    This class manages YOLO models in PyTorch and ONNX formats,
    handling loading, switching, and configuration.

    Responsibilities:
        - Load PyTorch models (.pt)
        - Load ONNX models (.onnx)
        - Switch between model formats
        - Verify model availability
        - Configure device (CPU/GPU)
        - Update model parameters

    Attributes:
        model_path: Path to the model file.
        device: Device for inference ('cpu', 'cuda', 'mps').
        use_half_precision: Whether to use FP16 precision.
        imgsz: Image size for inference.
        vehicle_classes: Classes to detect.

    Example:
        >>> manager = ModelManager(
        ...     model_path="yolov8n.pt", device="auto", imgsz=640, vehicle_classes=[2, 3, 5, 7]
        ... )
        >>>
        >>> # Load PyTorch model
        >>> if manager.load_pytorch():
        ...     print("PyTorch model loaded")
        >>>
        >>> # Load ONNX model
        >>> if manager.load_onnx():
        ...     print("ONNX model loaded")
        >>>
        >>> # Get model information
        >>> info = manager.get_model_info()
        >>> print(f"PyTorch: {info['pytorch_available']}")
        >>> print(f"ONNX: {info['onnx_available']}")
        >>>
        >>> # Update thresholds
        >>> manager.set_confidence_threshold(0.5)
        >>> manager.set_iou_threshold(0.45)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_half_precision: bool = False,
        imgsz: int = 320,
        vehicle_classes: list = None,
    ):
        """Initializes the model manager.

        Args:
            model_path: Path to the model file (.pt or .onnx).
            device: Device for inference ('cpu', 'cuda', 'mps', 'auto').
            use_half_precision: Whether to use FP16 precision.
            imgsz: Image size for inference (must be multiple of 32).
            vehicle_classes: List of class IDs to detect.

        Example:
            >>> manager = ModelManager(
            ...     model_path="yolov8m.pt", device="cuda", use_half_precision=True
            ... )
        """
        if hasattr(device, "value"):
            device = device.value

        if device not in ["cpu", "cuda", "mps"]:
            device = "cpu"

        self.model_path = model_path
        self.device = device
        self.use_half_precision = use_half_precision
        self.imgsz = imgsz
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]

        self._pytorch_model: YOLO | None = None
        self._onnx_session: ort.InferenceSession | None = None
        self._onnx_available = False
        self._pytorch_available = False

        self._input_name: str | None = None
        self._output_names: list | None = None

        self.logger.info(
            "ModelManager initialized", model_path=model_path, device=device, imgsz=imgsz
        )

    def load_pytorch(self) -> bool:
        """Loads the PyTorch model.

        This method loads a YOLO model in PyTorch format (.pt) and
        configures it with the current device and parameters.

        Returns:
            bool: True if the model loaded successfully.

        Example:
            >>> if manager.load_pytorch():
            ...     print("PyTorch model loaded successfully")
            ... else:
            ...     print("Failed to load PyTorch model")
        """
        model_path = Path(self.model_path)
        if not model_path.exists():
            self.logger.error(f"Model not found: {self.model_path}")
            return False

        try:
            self.logger.info("Loading PyTorch model...")
            self._pytorch_model = YOLO(self.model_path)

            if self.device != "cpu":
                try:
                    self._pytorch_model.to(self.device)
                    self.logger.debug(f"Model moved to {self.device}")
                except Exception as e:
                    self.logger.warning(f"Could not move to {self.device}, using CPU: {e}")
                    self.device = "cpu"

            self._pytorch_model.conf = 0.35
            self._pytorch_model.iou = 0.45
            self._pytorch_model.classes = self.vehicle_classes

            self._pytorch_available = True
            self.logger.info("PyTorch model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading PyTorch: {e}")
            self._pytorch_available = False
            return False

    def load_onnx(self, onnx_path: str | None = None) -> bool:
        """Loads the ONNX model.

        This method loads a YOLO model in ONNX format using ONNX Runtime.

        Args:
            onnx_path: Path to the ONNX file (optional).
                If None, uses the .onnx version of the model_path.

        Returns:
            bool: True if the model loaded successfully.

        Example:
            >>> # Load from default path (model_path.pt -> model_path.onnx)
            >>> if manager.load_onnx():
            ...     print("ONNX model loaded")
            >>>
            >>> # Load from specific path
            >>> if manager.load_onnx("models/yolov8n.onnx"):
            ...     print("ONNX model loaded from custom path")
        """
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX Runtime not available")
            return False

        if onnx_path is None:
            onnx_path = self.model_path.replace(".pt", ".onnx")

        if not os.path.exists(onnx_path):
            self.logger.warning(f"ONNX file not found: {onnx_path}")
            return False

        try:
            self.logger.info("Loading ONNX model...")

            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

            self._onnx_session = ort.InferenceSession(
                onnx_path, providers=providers, sess_options=sess_options
            )

            self._input_name = self._onnx_session.get_inputs()[0].name
            self._output_names = [o.name for o in self._onnx_session.get_outputs()]

            self._onnx_available = True
            self.logger.info(
                "ONNX model loaded successfully", providers=self._onnx_session.get_providers()
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading ONNX: {e}")
            self._onnx_available = False
            return False

    def get_pytorch_model(self) -> YOLO | None:
        """Gets the PyTorch model.

        Returns:
            YOLO | None: The PyTorch model or None if not loaded.
        """
        return self._pytorch_model

    def get_onnx_session(self) -> ort.InferenceSession | None:
        """Gets the ONNX session.

        Returns:
            ort.InferenceSession | None: The ONNX session or None if not loaded.
        """
        return self._onnx_session

    def get_onnx_input_name(self) -> str | None:
        """Gets the ONNX input name.

        Returns:
            str | None: The input name or None if not loaded.
        """
        return self._input_name

    def get_onnx_output_names(self) -> list | None:
        """Gets the ONNX output names.

        Returns:
            list | None: The output names or None if not loaded.
        """
        return self._output_names

    @property
    def has_pytorch(self) -> bool:
        """Whether the PyTorch model is loaded."""
        return self._pytorch_available

    @property
    def has_onnx(self) -> bool:
        """Whether the ONNX model is loaded."""
        return self._onnx_available

    @property
    def is_onnx_available_globally(self) -> bool:
        """Whether ONNX Runtime is available globally."""
        return ONNX_AVAILABLE

    def set_confidence_threshold(self, threshold: float) -> None:
        """Updates the confidence threshold in the models.

        Args:
            threshold: Confidence threshold (0-1).

        Example:
            >>> manager.set_confidence_threshold(0.6)
            >>> # All loaded models will use the new threshold
        """
        if self._pytorch_model:
            self._pytorch_model.conf = threshold

    def set_iou_threshold(self, threshold: float) -> None:
        """Updates the IoU threshold in the models.

        Args:
            threshold: IoU threshold (0-1).

        Example:
            >>> manager.set_iou_threshold(0.5)
            >>> # All loaded models will use the new threshold
        """
        if self._pytorch_model:
            self._pytorch_model.iou = threshold

    def set_classes(self, classes: list) -> None:
        """Updates the classes to detect.

        Args:
            classes: List of class IDs to detect.

        Example:
            >>> manager.set_classes([2, 3, 5, 7])
            >>> # Only cars, motorcycles, buses, and trucks
        """
        if self._pytorch_model:
            self._pytorch_model.classes = classes

    def get_model_info(self) -> dict:
        """Gets information about the loaded models.

        Returns:
            dict: Model information including:
                - pytorch_available: Whether PyTorch model is loaded
                - onnx_available: Whether ONNX model is loaded
                - device: Current device
                - imgsz: Image size
                - model_path: Model file path
                - half_precision: Whether FP16 is enabled

        Example:
            >>> info = manager.get_model_info()
            >>> print(f"PyTorch: {info['pytorch_available']}")
            >>> print(f"ONNX: {info['onnx_available']}")
            >>> print(f"Device: {info['device']}")
        """
        return {
            "pytorch_available": self._pytorch_available,
            "onnx_available": self._onnx_available,
            "device": self.device,
            "imgsz": self.imgsz,
            "model_path": self.model_path,
            "half_precision": self.use_half_precision,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up resources.

        This method releases the PyTorch model and ONNX session
        when exiting the context.
        """
        self._pytorch_model = None
        self._onnx_session = None
