"""Inference engine for YOLO.

Handles inference with different backends (PyTorch and ONNX)
and provides a unified interface for object detection.

This module abstracts away backend-specific details, allowing the
system to switch between PyTorch and ONNX Runtime seamlessly.
"""

from abc import ABC, abstractmethod

import numpy as np
from ultralytics import YOLO

from core.constants import IMAGE_CHANNELS_RGB

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from utils.logger import LoggerMixin


class InferenceEngine(ABC):
    """Abstract interface for inference engines.

    This interface defines the contract that all inference engines
    must implement, providing a unified way to perform inference
    across different backends.

    Methods:
        infer: Perform inference on a frame.
        warmup: Warm up the engine to reduce initial latency.
        is_available: Check if the engine is available.

    Example:
        >>> engine = PyTorchInferenceEngine(model, imgsz=640)
        >>> engine.warmup()
        >>> results = engine.infer(frame)
        >>> if engine.is_available:
        ...     print("Engine is ready")
    """

    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Performs inference on a frame.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR format.

        Returns:
            np.ndarray: Inference results.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Warms up the engine to reduce initial latency.

        This method performs dummy inference to initialize the engine
        and load models into memory, reducing the first inference time.
        """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the engine is available.

        Returns:
            bool: True if the engine is ready for inference.
        """


class PyTorchInferenceEngine(InferenceEngine, LoggerMixin):
    """PyTorch inference engine.

    This engine uses PyTorch (via Ultralytics YOLO) for inference,
    supporting CUDA, MPS, and CPU devices.

    Attributes:
        model: PyTorch YOLO model.
        imgsz: Image size for inference.
        vehicle_classes: Classes to detect.
        device: Device for inference ('cpu', 'cuda', 'mps').
        max_det: Maximum detections per image.

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolov8n.pt")
        >>> engine = PyTorchInferenceEngine(model=model, imgsz=640, device="cuda", max_det=100)
        >>> engine.warmup()
        >>> results = engine.infer(frame)
    """

    def __init__(
        self,
        model: YOLO,
        imgsz: int = 320,
        vehicle_classes: list | None = None,
        device: str = "cpu",
        max_det: int = 100,
    ):
        """Initializes the PyTorch inference engine.

        Args:
            model: PyTorch YOLO model.
            imgsz: Image size for inference (must be multiple of 32).
            vehicle_classes: List of class IDs to detect.
            device: Device for inference ('cpu', 'cuda', 'mps').
            max_det: Maximum number of detections per image.
        """
        if hasattr(device, "value"):
            device = device.value

        if device not in ["cpu", "cuda", "mps"]:
            device = "cpu"

        self.model = model
        self.imgsz = imgsz
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]
        self.device = device
        self.max_det = max_det

        self._warmed_up = False

        self.logger.info("PyTorchInferenceEngine initialized", imgsz=imgsz, device=device)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Performs inference using PyTorch.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR format.

        Returns:
            np.ndarray: Inference results from YOLO.

        Note:
            Returns None if inference fails.
        """
        if self.model is None:
            return np.array([])

        try:
            results = self.model(
                frame,
                classes=self.vehicle_classes,
                verbose=False,
                augment=False,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_det,
            )
            return results[0] if results else None

        except Exception as e:
            self.logger.error(f"PyTorch inference error: {e}")
            return None

    def warmup(self) -> None:
        """Warms up the PyTorch model.

        Performs dummy inference to initialize CUDA/MPS kernels and
        load the model into memory.
        """
        if self._warmed_up or self.model is None:
            return

        self.logger.info("Warming up PyTorch...")
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(3):
                _ = self.infer(dummy)
            self._warmed_up = True
            self.logger.info("PyTorch warmed up")
        except Exception as e:
            self.logger.warning(f"PyTorch warmup error: {e}")

    @property
    def is_available(self) -> bool:
        """Checks if the PyTorch engine is available.

        Returns:
            bool: True if the model is loaded.
        """
        return self.model is not None


class ONNXInferenceEngine(InferenceEngine, LoggerMixin):
    """ONNX Runtime inference engine.

    This engine uses ONNX Runtime for inference, providing better
    CPU performance and cross-platform compatibility.

    Attributes:
        session: ONNX Runtime inference session.
        input_name: Name of the input node.
        output_names: Names of the output nodes.
        imgsz: Image size for inference.

    Example:
        >>> import onnxruntime as ort
        >>> session = ort.InferenceSession("model.onnx")
        >>> engine = ONNXInferenceEngine(
        ...     session=session, input_name="images", output_names=["outputs"], imgsz=640
        ... )
        >>> engine.warmup()
        >>> results = engine.infer(frame)
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        input_name: str,
        output_names: list,
        imgsz: int = 320,
    ):
        """Initializes the ONNX inference engine.

        Args:
            session: ONNX Runtime inference session.
            input_name: Name of the input node.
            output_names: Names of the output nodes.
            imgsz: Image size for inference.
        """
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        self.imgsz = imgsz

        self._warmed_up = False

        self.logger.info(
            "ONNXInferenceEngine initialized",
            imgsz=imgsz,
            providers=session.get_providers() if session else [],
        )

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Performs inference using ONNX Runtime.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR format.

        Returns:
            np.ndarray: Inference results.

        Note:
            The frame is automatically preprocessed (resized, normalized,
            and transposed) to match ONNX input requirements.
        """
        if self.session is None:
            return np.array([])

        try:
            if len(frame.shape) == IMAGE_CHANNELS_RGB:
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frame = np.expand_dims(frame, axis=0)

            inputs = {self.input_name: frame}
            outputs = self.session.run(self.output_names, inputs)

            return outputs[0] if outputs else np.array([])

        except Exception as e:
            self.logger.error(f"ONNX inference error: {e}")
            return np.array([])

    def warmup(self) -> None:
        """Warms up ONNX Runtime.

        Performs dummy inference to initialize the session and load
        the model into memory.
        """
        if self._warmed_up or self.session is None:
            return

        self.logger.info("Warming up ONNX...")
        try:
            dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
            for _ in range(3):
                inputs = {self.input_name: dummy}
                self.session.run(self.output_names, inputs)
            self._warmed_up = True
            self.logger.info("ONNX warmed up")
        except Exception as e:
            self.logger.warning(f"ONNX warmup error: {e}")

    @property
    def is_available(self) -> bool:
        """Checks if the ONNX engine is available.

        Returns:
            bool: True if the session is loaded.
        """
        return self.session is not None


class InferenceEngineFactory:
    """Factory for inference engines.

    This factory provides methods to create different inference
    engine types with consistent configuration.

    Example:
        >>> # Create PyTorch engine
        >>> engine = InferenceEngineFactory.create_pytorch(model=model, imgsz=640, device="cuda")
        >>>
        >>> # Create ONNX engine
        >>> engine = InferenceEngineFactory.create_onnx(
        ...     session=session, input_name="images", output_names=["outputs"], imgsz=640
        ... )
    """

    @staticmethod
    def create_pytorch(
        model: YOLO,
        imgsz: int = 320,
        vehicle_classes: list = None,
        device: str = "cpu",
        max_det: int = 100,
    ) -> PyTorchInferenceEngine:
        """Creates a PyTorch inference engine.

        Args:
            model: PyTorch YOLO model.
            imgsz: Image size for inference.
            vehicle_classes: Classes to detect.
            device: Device for inference.
            max_det: Maximum detections per image.

        Returns:
            PyTorchInferenceEngine: PyTorch inference engine instance.

        Example:
            >>> engine = InferenceEngineFactory.create_pytorch(
            ...     model=model, imgsz=640, device="cuda"
            ... )
        """
        return PyTorchInferenceEngine(
            model=model,
            imgsz=imgsz,
            vehicle_classes=vehicle_classes,
            device=device,
            max_det=max_det,
        )

    @staticmethod
    def create_onnx(
        session: ort.InferenceSession,
        input_name: str,
        output_names: list,
        imgsz: int = 320,
    ) -> ONNXInferenceEngine:
        """Creates an ONNX inference engine.

        Args:
            session: ONNX Runtime inference session.
            input_name: Name of the input node.
            output_names: Names of the output nodes.
            imgsz: Image size for inference.

        Returns:
            ONNXInferenceEngine: ONNX inference engine instance.

        Example:
            >>> engine = InferenceEngineFactory.create_onnx(
            ...     session=session, input_name="images", output_names=["outputs"], imgsz=640
            ... )
        """
        return ONNXInferenceEngine(
            session=session,
            input_name=input_name,
            output_names=output_names,
            imgsz=imgsz,
        )
