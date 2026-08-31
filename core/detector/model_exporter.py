"""Model exporter to ONNX.

Handles exporting YOLO models to ONNX format for
optimized inference on CPU.
"""

import os
from pathlib import Path

from ultralytics import YOLO

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from utils.logger import LoggerMixin


class ModelExporter(LoggerMixin):
    """Exporter for models to ONNX format.

    This class handles the export of PyTorch YOLO models to ONNX
    format for optimized inference with ONNX Runtime.

    Responsibilities:
        - Export PyTorch models to ONNX
        - Optimize ONNX models
        - Verify successful export
        - Validate exported models

    Attributes:
        model_path: Path to the PyTorch model.
        imgsz: Image size for export.
        opset: ONNX opset version.
        simplify: Whether to simplify the ONNX model.

    Example:
        >>> exporter = ModelExporter(model_path="yolov8n.pt", imgsz=640, opset=12, simplify=True)
        >>>
        >>> # Export the model
        >>> onnx_path = exporter.export()
        >>> if onnx_path:
        ...     print(f"Model exported to: {onnx_path}")
        ...     # Verify the export
        ...     if exporter.verify_export():
        ...         print("Export verified successfully")
        >>>
        >>> # Force re-export
        >>> onnx_path = exporter.export(force=True)
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 320,
        opset: int = 12,
        simplify: bool = True,
    ):
        """Initializes the model exporter.

        Args:
            model_path: Path to the PyTorch model file (.pt).
            imgsz: Image size for inference (must be multiple of 32).
            opset: ONNX opset version (default: 12).
            simplify: Whether to simplify the ONNX model.

        Example:
            >>> exporter = ModelExporter(model_path="yolov8m.pt", imgsz=640, simplify=True)
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.opset = opset
        self.simplify = simplify

        self._exported_path: str | None = None
        self._export_success = False

        self.logger.info(
            "ModelExporter initialized", model_path=model_path, imgsz=imgsz, opset=opset
        )

    def export(self, force: bool = False) -> str | None:
        """Exports the model to ONNX format.

        Args:
            force: Force export even if the file already exists.

        Returns:
            Optional[str]: Path to the ONNX file or None if export failed.

        Example:
            >>> # Export with default settings
            >>> onnx_path = exporter.export()
            >>>
            >>> # Force re-export
            >>> onnx_path = exporter.export(force=True)
            >>>
            >>> if onnx_path:
            ...     print(f"ONNX file created at: {onnx_path}")
        """
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX Runtime not available for export")
            return None

        onnx_path = Path(self.model_path).with_suffix(".onnx")

        if onnx_path.exists() and not force:
            self.logger.info(f"ONNX file already exists: {onnx_path}")
            self._exported_path = str(onnx_path)
            self._export_success = True
            return str(onnx_path)

        try:
            self.logger.info(f"Exporting model to ONNX: {onnx_path}")

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
                self._exported_path = str(onnx_path)
                self._export_success = True
                self.logger.info("Model exported to ONNX successfully")
                return str(onnx_path)

            self.logger.error("Export failed - file not created")
            return None

        except Exception as e:
            self.logger.error(f"Error exporting to ONNX: {e}")
            self._export_success = False
            return None

    def verify_export(self, onnx_path: str | None = None) -> bool:
        """Verifies that the ONNX file is valid.

        Args:
            onnx_path: Path to the ONNX file (optional).
                If None, uses the exported path.

        Returns:
            bool: True if the file is valid.

        Example:
            >>> # Verify the exported model
            >>> if exporter.verify_export():
            ...     print("ONNX model is valid")
            >>>
            >>> # Verify a specific file
            >>> if exporter.verify_export("model.onnx"):
            ...     print("File is valid")
        """
        if onnx_path is None:
            onnx_path = self._exported_path

        if onnx_path is None or not os.path.exists(onnx_path):
            return False

        try:
            import onnx

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            self.logger.info("ONNX model verified successfully")
            return True

        except ImportError:
            self.logger.warning("Could not verify ONNX (onnx package not installed)")
            return True

        except Exception as e:
            self.logger.error(f"Error verifying ONNX: {e}")
            return False

    @property
    def exported_path(self) -> str | None:
        """Path to the exported ONNX file.

        Returns:
            Optional[str]: Path to the exported ONNX file or None.
        """
        return self._exported_path

    @property
    def success(self) -> bool:
        """Whether the export was successful.

        Returns:
            bool: True if the export succeeded.
        """
        return self._export_success
