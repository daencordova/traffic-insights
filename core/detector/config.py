"""
Configuración específica para el detector de objetos.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DetectorConfig:
    """
    Configuración para el detector de objetos.

    Attributes:
        model_path: Ruta al modelo YOLO
        confidence_threshold: Umbral de confianza (0-1)
        iou_threshold: Umbral de IoU para NMS (0-1)
        vehicle_classes: Lista de IDs de clases a detectar
        device: Dispositivo para inferencia ('cpu', 'cuda', 'mps')
        use_half_precision: Si usar half precision (FP16)
        use_onnx: Si usar modelo ONNX
        imgsz: Tamaño de imagen para inferencia
        max_det: Número máximo de detecciones por imagen
        use_optimized: Si usar versión optimizada para CPU
    """
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    device: str = "auto"
    use_half_precision: bool = False
    use_onnx: bool = False
    imgsz: int = 320
    max_det: int = 10
    use_optimized: bool = True

    def validate(self) -> bool:
        """Valida la configuración."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold debe estar entre 0 y 1: {self.confidence_threshold}")

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"IoU threshold debe estar entre 0 y 1: {self.iou_threshold}")

        if self.imgsz not in [320, 416, 512, 640, 768, 832, 1024]:
            raise ValueError(f"imgsz debe ser múltiplo de 32: {self.imgsz}")

        return True

    @classmethod
    def from_global_config(cls) -> "DetectorConfig":
        """Crea configuración desde la configuración global."""
        from config.manager import config

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
            use_optimized=getattr(config.optimization, "use_optimized_detector", True)
        )
