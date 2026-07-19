"""
Motor de inferencia para YOLO.

Maneja la inferencia con diferentes backends (PyTorch y ONNX)
y proporciona una interfaz unificada.
"""

from abc import ABC, abstractmethod

import numpy as np
from ultralytics import YOLO

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from utils.logger import LoggerMixin


class InferenceEngine(ABC):
    """Interfaz abstracta para motores de inferencia."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Realiza inferencia en un frame."""
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Calienta el motor para reducir latencia inicial."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el motor está disponible."""
        pass


class PyTorchInferenceEngine(InferenceEngine, LoggerMixin):
    """
    Motor de inferencia con PyTorch.

    Attributes:
        model: Modelo YOLO de PyTorch
        imgsz: Tamaño de imagen para inferencia
        vehicle_classes: Clases a detectar
    """

    def __init__(
        self,
        model: YOLO,
        imgsz: int = 320,
        vehicle_classes: list = None,
        device: str = "cpu",
        max_det: int = 100,
    ):
        self.model = model
        self.imgsz = imgsz
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]
        self.device = device
        self.max_det = max_det

        self._warmed_up = False

        self.logger.info(
            "PyTorchInferenceEngine inicializado",
            imgsz=imgsz,
            device=device
        )

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Realiza inferencia con PyTorch.

        Args:
            frame: Imagen a procesar

        Returns:
            np.ndarray: Resultados de la inferencia
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
            self.logger.error(f"Error en inferencia PyTorch: {e}")
            return None

    def warmup(self) -> None:
        """Calienta el modelo PyTorch."""
        if self._warmed_up or self.model is None:
            return

        self.logger.info("🔥 Calentando PyTorch...")
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(3):
                _ = self.infer(dummy)
            self._warmed_up = True
            self.logger.info("✅ PyTorch calentado")
        except Exception as e:
            self.logger.warning(f"Error en warmup PyTorch: {e}")

    @property
    def is_available(self) -> bool:
        return self.model is not None


class ONNXInferenceEngine(InferenceEngine, LoggerMixin):
    """
    Motor de inferencia con ONNX Runtime.

    Attributes:
        session: Sesión de ONNX Runtime
        input_name: Nombre del input
        output_names: Nombres de los outputs
        imgsz: Tamaño de imagen para inferencia
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        input_name: str,
        output_names: list,
        imgsz: int = 320,
    ):
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        self.imgsz = imgsz

        self._warmed_up = False

        self.logger.info(
            "ONNXInferenceEngine inicializado",
            imgsz=imgsz,
            providers=session.get_providers() if session else []
        )

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Realiza inferencia con ONNX Runtime.

        Args:
            frame: Imagen a procesar

        Returns:
            np.ndarray: Resultados de la inferencia
        """
        if self.session is None:
            return np.array([])

        try:
            if len(frame.shape) == 3:
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frame = np.expand_dims(frame, axis=0)

            inputs = {self.input_name: frame}
            outputs = self.session.run(self.output_names, inputs)

            return outputs[0] if outputs else np.array([])

        except Exception as e:
            self.logger.error(f"Error en inferencia ONNX: {e}")
            return np.array([])

    def warmup(self) -> None:
        """Calienta ONNX Runtime."""
        if self._warmed_up or self.session is None:
            return

        self.logger.info("🔥 Calentando ONNX...")
        try:
            dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
            for _ in range(3):
                inputs = {self.input_name: dummy}
                self.session.run(self.output_names, inputs)
            self._warmed_up = True
            self.logger.info("✅ ONNX calentado")
        except Exception as e:
            self.logger.warning(f"Error en warmup ONNX: {e}")

    @property
    def is_available(self) -> bool:
        return self.session is not None


class InferenceEngineFactory:
    """Fábrica de motores de inferencia."""

    @staticmethod
    def create_pytorch(
        model: YOLO,
        imgsz: int = 320,
        vehicle_classes: list = None,
        device: str = "cpu",
        max_det: int = 100,
    ) -> PyTorchInferenceEngine:
        """Crea un motor PyTorch."""
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
        """Crea un motor ONNX."""
        return ONNXInferenceEngine(
            session=session,
            input_name=input_name,
            output_names=output_names,
            imgsz=imgsz,
        )
