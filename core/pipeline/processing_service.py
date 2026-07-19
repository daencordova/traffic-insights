"""
Servicio de procesamiento de frames.

Maneja la detección, tracking y conteo de vehículos.
"""

import time
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass

import numpy as np

from core.detector import YOLODetector, OptimizedYOLODetector
from core.tracker import AdvancedTracker
from core.counter import VehicleCounter
from core.frame_buffer import FrameMetadata
from utils.logger import LoggerMixin


@dataclass
class ProcessingResult:
    """Resultado del procesamiento de un frame."""
    frame_number: int
    detections: List[dict]
    tracks: dict
    stats: dict
    processed_frame: np.ndarray
    processing_time_ms: float
    capture_time_ms: float
    timestamp: float

    @property
    def total_time_ms(self) -> float:
        return self.capture_time_ms + self.processing_time_ms


class ProcessingService(LoggerMixin):
    """
    Servicio de procesamiento de frames.

    Responsabilidades:
    - Detectar objetos en el frame
    - Actualizar el tracker
    - Actualizar el contador
    - Aplicar optimizaciones (batch processing)

    Attributes:
        detector: Detector de objetos
        tracker: Tracker de objetos
        counter: Contador de vehículos
        enable_batch: Habilitar procesamiento por lotes
        batch_size: Tamaño del lote
        on_frame_processed: Callback opcional al procesar un frame
    """

    def __init__(
        self,
        config=None,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[AdvancedTracker] = None,
        counter: Optional[VehicleCounter] = None,
        enable_batch: bool = False,
        batch_size: int = 4,
        on_frame_processed: Optional[Callable[[ProcessingResult], None]] = None,
    ):
        self.config = config
        self.tracker = tracker or AdvancedTracker()
        self.counter = counter or VehicleCounter()
        self.enable_batch = enable_batch
        self.batch_size = batch_size
        self.on_frame_processed = on_frame_processed

        self.detector = self._init_detector(detector)

        self._processing_time_ms = 0.0
        self._processed_count = 0

        self.logger.info(
            "ProcessingService inicializado",
            batch_enabled=enable_batch,
            batch_size=batch_size,
            detector_type=type(self.detector).__name__
        )

    def _init_detector(self, detector: Optional[YOLODetector]) -> YOLODetector:
        """Inicializa el detector."""
        if detector is not None:
            return detector

        use_optimized = getattr(
            self.config.optimization, "use_optimized_detector", True
        )

        if use_optimized:
            try:
                self.logger.info("✅ Detector optimizado activado")
                return OptimizedYOLODetector()
            except Exception as e:
                self.logger.warning(
                    f"Detector optimizado no disponible: {e}. Usando estándar."
                )

        return YOLODetector()

    def process_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> Optional[ProcessingResult]:
        """
        Procesa un único frame.

        Args:
            frame: Frame a procesar
            metadata: Metadatos del frame

        Returns:
            Optional[ProcessingResult]: Resultado del procesamiento
        """
        if frame is None or frame.size == 0:
            return None

        capture_start = time.perf_counter()
        process_start = time.perf_counter()

        try:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)
            stats = self.counter.process(tracks, frame)
        except Exception as e:
            self.logger.error(f"Error procesando frame: {e}")
            return None

        process_time = (time.perf_counter() - process_start) * 1000
        capture_time = (time.perf_counter() - capture_start) * 1000

        self._processing_time_ms = process_time
        self._processed_count += 1

        result = ProcessingResult(
            frame_number=metadata.frame_number,
            detections=detections,
            tracks=tracks,
            stats=stats,
            processed_frame=frame.copy(),
            processing_time_ms=process_time,
            capture_time_ms=capture_time,
            timestamp=metadata.timestamp
        )

        if self.on_frame_processed:
            self.on_frame_processed(result)

        return result

    def process_batch(self, batch: List[Tuple[np.ndarray, FrameMetadata]]) -> List[ProcessingResult]:
        """
        Procesa un lote de frames usando batch inference.

        Args:
            batch: Lista de tuplas (frame, metadata)

        Returns:
            List[ProcessingResult]: Resultados del procesamiento
        """
        if not batch:
            return []

        results = []

        try:
            frames = [frame for frame, _ in batch]
            metadatas = [metadata for _, metadata in batch]

            if hasattr(self.detector, 'detect_batch'):
                batch_detections = self.detector.detect_batch(frames)
            else:
                batch_detections = [self.detector.detect(frame) for frame in frames]

            for i, (frame, metadata, detections) in enumerate(
                zip(frames, metadatas, batch_detections)
            ):
                try:
                    process_start = time.perf_counter()

                    tracks = self.tracker.update(detections, frame)
                    stats = self.counter.process(tracks, frame)

                    process_time = (time.perf_counter() - process_start) * 1000

                    result = ProcessingResult(
                        frame_number=metadata.frame_number,
                        detections=detections,
                        tracks=tracks,
                        stats=stats,
                        processed_frame=frame.copy(),
                        processing_time_ms=process_time,
                        capture_time_ms=0.0,
                        timestamp=metadata.timestamp
                    )

                    results.append(result)

                    if self.on_frame_processed:
                        self.on_frame_processed(result)

                except Exception as e:
                    self.logger.error(f"Error procesando frame en lote: {e}")
                    continue

            return results

        except Exception as e:
            self.logger.error(f"Error en batch processing: {e}")
            for frame, metadata in batch:
                result = self.process_frame(frame, metadata)
                if result:
                    results.append(result)
            return results

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        return {
            "processed_count": self._processed_count,
            "avg_processing_time_ms": self._processing_time_ms,
            "detector": self.detector.get_performance_stats(),
            "tracker": self.tracker.get_stats(),
            "counter": self.counter.get_stats(),
        }

    def reset(self) -> None:
        """Reinicia los componentes."""
        self.tracker.reset()
        self.counter.reset()
        self._processed_count = 0
        self._processing_time_ms = 0.0
