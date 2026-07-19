"""
Gestor de modelos para el detector YOLO.

Maneja la carga, gestión y cambio entre diferentes formatos de modelo
(PyTorch y ONNX).
"""

import os
from typing import Optional

from ultralytics import YOLO

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from utils.logger import LoggerMixin


class ModelLoadError(Exception):
    """Error al cargar un modelo."""
    pass


class ModelManager(LoggerMixin):
    """
    Gestor de modelos para YOLO.

    Responsabilidades:
    - Cargar modelos PyTorch (.pt)
    - Cargar modelos ONNX (.onnx)
    - Cambiar entre formatos de modelo
    - Verificar disponibilidad del modelo
    - Configurar dispositivo (CPU/GPU)

    Attributes:
        model_path: Ruta al modelo
        device: Dispositivo para inferencia
        use_half_precision: Usar FP16
        imgsz: Tamaño de imagen para inferencia
        vehicle_classes: Clases a detectar
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_half_precision: bool = False,
        imgsz: int = 320,
        vehicle_classes: list = None,
    ):
        self.model_path = model_path
        self.device = device
        self.use_half_precision = use_half_precision
        self.imgsz = imgsz
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]

        self._pytorch_model: Optional[YOLO] = None
        self._onnx_session: Optional[ort.InferenceSession] = None
        self._onnx_available = False
        self._pytorch_available = False

        self._input_name: Optional[str] = None
        self._output_names: Optional[list] = None

        self.logger.info(
            "ModelManager inicializado",
            model_path=model_path,
            device=device,
            imgsz=imgsz
        )

    def load_pytorch(self) -> bool:
        """
        Carga el modelo PyTorch.

        Returns:
            bool: True si se cargó correctamente
        """
        if not os.path.exists(self.model_path):
            self.logger.error(f"Modelo no encontrado: {self.model_path}")
            return False

        try:
            self.logger.info("Cargando modelo PyTorch...")
            self._pytorch_model = YOLO(self.model_path)

            if self.device != "cpu":
                try:
                    self._pytorch_model.to(self.device)
                    self.logger.debug(f"Modelo movido a {self.device}")
                except Exception as e:
                    self.logger.warning(
                        f"No se pudo mover a {self.device}, usando CPU: {e}"
                    )
                    self.device = "cpu"

            self._pytorch_model.conf = 0.35
            self._pytorch_model.iou = 0.45
            self._pytorch_model.classes = self.vehicle_classes

            self._pytorch_available = True
            self.logger.info("✅ Modelo PyTorch cargado correctamente")
            return True

        except Exception as e:
            self.logger.error(f"Error cargando PyTorch: {e}")
            self._pytorch_available = False
            return False

    def load_onnx(self, onnx_path: Optional[str] = None) -> bool:
        """
        Carga el modelo ONNX.

        Args:
            onnx_path: Ruta al archivo ONNX (opcional)

        Returns:
            bool: True si se cargó correctamente
        """
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX Runtime no disponible")
            return False

        if onnx_path is None:
            onnx_path = self.model_path.replace(".pt", ".onnx")

        if not os.path.exists(onnx_path):
            self.logger.warning(f"Archivo ONNX no encontrado: {onnx_path}")
            return False

        try:
            self.logger.info("Cargando modelo ONNX...")

            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')

            self._onnx_session = ort.InferenceSession(
                onnx_path,
                providers=providers,
                sess_options=sess_options
            )

            self._input_name = self._onnx_session.get_inputs()[0].name
            self._output_names = [o.name for o in self._onnx_session.get_outputs()]

            self._onnx_available = True
            self.logger.info(
                "✅ Modelo ONNX cargado correctamente",
                providers=self._onnx_session.get_providers()
            )
            return True

        except Exception as e:
            self.logger.error(f"Error cargando ONNX: {e}")
            self._onnx_available = False
            return False

    def get_pytorch_model(self) -> Optional[YOLO]:
        """Obtiene el modelo PyTorch."""
        return self._pytorch_model

    def get_onnx_session(self) -> Optional[ort.InferenceSession]:
        """Obtiene la sesión ONNX."""
        return self._onnx_session

    def get_onnx_input_name(self) -> Optional[str]:
        """Obtiene el nombre del input ONNX."""
        return self._input_name

    def get_onnx_output_names(self) -> Optional[list]:
        """Obtiene los nombres de los outputs ONNX."""
        return self._output_names

    @property
    def has_pytorch(self) -> bool:
        return self._pytorch_available

    @property
    def has_onnx(self) -> bool:
        return self._onnx_available

    @property
    def is_onnx_available_globally(self) -> bool:
        return ONNX_AVAILABLE

    def set_confidence_threshold(self, threshold: float) -> None:
        """Actualiza el umbral de confianza en los modelos."""
        if self._pytorch_model:
            self._pytorch_model.conf = threshold

    def set_iou_threshold(self, threshold: float) -> None:
        """Actualiza el umbral de IoU en los modelos."""
        if self._pytorch_model:
            self._pytorch_model.iou = threshold

    def set_classes(self, classes: list) -> None:
        """Actualiza las clases a detectar."""
        if self._pytorch_model:
            self._pytorch_model.classes = classes

    def get_model_info(self) -> dict:
        """Obtiene información del modelo."""
        return {
            "pytorch_available": self._pytorch_available,
            "onnx_available": self._onnx_available,
            "device": self.device,
            "imgsz": self.imgsz,
            "model_path": self.model_path,
            "half_precision": self.use_half_precision,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Limpia recursos."""
        self._pytorch_model = None
        self._onnx_session = None
