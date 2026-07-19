"""
Exportador de modelos a ONNX.

Maneja la exportación de modelos YOLO a formato ONNX para
inferencia optimizada en CPU.
"""

import os
from typing import Optional

from ultralytics import YOLO

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from utils.logger import LoggerMixin


class ModelExporter(LoggerMixin):
    """
    Exportador de modelos a ONNX.

    Responsabilidades:
    - Exportar modelos PyTorch a ONNX
    - Optimizar modelos ONNX
    - Verificar exportación exitosa

    Attributes:
        model_path: Ruta al modelo PyTorch
        imgsz: Tamaño de imagen para exportación
        opset: Versión de opset ONNX
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 320,
        opset: int = 12,
        simplify: bool = True,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.opset = opset
        self.simplify = simplify

        self._exported_path: Optional[str] = None
        self._export_success = False

        self.logger.info(
            "ModelExporter inicializado",
            model_path=model_path,
            imgsz=imgsz,
            opset=opset
        )

    def export(self, force: bool = False) -> Optional[str]:
        """
        Exporta el modelo a ONNX.

        Args:
            force: Forzar exportación aunque el archivo exista

        Returns:
            Optional[str]: Ruta al archivo ONNX o None
        """
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX Runtime no disponible para exportación")
            return None

        onnx_path = self.model_path.replace(".pt", ".onnx")

        if os.path.exists(onnx_path) and not force:
            self.logger.info(f"Archivo ONNX ya existe: {onnx_path}")
            self._exported_path = onnx_path
            self._export_success = True
            return onnx_path

        try:
            self.logger.info(f"Exportando modelo a ONNX: {onnx_path}")

            model = YOLO(self.model_path)

            model.export(
                format="onnx",
                imgsz=self.imgsz,
                optimize=True,
                opset=self.opset,
                simplify=self.simplify,
                dynamic=False,
                verbose=False,
            )

            if os.path.exists(onnx_path):
                self._exported_path = onnx_path
                self._export_success = True
                self.logger.info("✅ Modelo exportado a ONNX correctamente")
                return onnx_path
            else:
                self.logger.error("Exportación falló - archivo no creado")
                return None

        except Exception as e:
            self.logger.error(f"Error exportando a ONNX: {e}")
            self._export_success = False
            return None

    def verify_export(self, onnx_path: Optional[str] = None) -> bool:
        """
        Verifica que el archivo ONNX sea válido.

        Args:
            onnx_path: Ruta al archivo ONNX (opcional)

        Returns:
            bool: True si el archivo es válido
        """
        if onnx_path is None:
            onnx_path = self._exported_path

        if onnx_path is None or not os.path.exists(onnx_path):
            return False

        try:
            import onnx

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            self.logger.info("✅ Modelo ONNX verificado correctamente")
            return True

        except ImportError:
            self.logger.warning("No se pudo verificar ONNX (onnx no instalado)")
            return True

        except Exception as e:
            self.logger.error(f"Error verificando ONNX: {e}")
            return False

    @property
    def exported_path(self) -> Optional[str]:
        return self._exported_path

    @property
    def success(self) -> bool:
        return self._export_success
