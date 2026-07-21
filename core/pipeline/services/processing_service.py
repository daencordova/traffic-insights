"""
Servicio de procesamiento de frames.

Responsable de:
- Detección de objetos en el frame
- Tracking de objetos
- Conteo de vehículos
- Procesamiento por lotes (batch processing)
"""

import time
import threading
from typing import Optional, Callable, List, Tuple
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
    __slots__ = (
        'frame_number', 'detections', 'tracks', 'stats',
        'processed_frame', 'processing_time_ms', 'capture_time_ms', 'timestamp'
    )

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
    Servicio especializado en procesamiento de frames.
    """

    def __init__(
        self,
        config,
        detector=None,
        tracker=None,
        counter=None,
        enable_batch: bool = False,
        batch_size: int = 4,
        on_frame_processed: Optional[Callable] = None,
    ):
        self.config = config
        self.detector = self._init_detector(detector)
        self.tracker = tracker or AdvancedTracker()
        self.counter = counter or VehicleCounter()
        self.enable_batch = enable_batch
        self.batch_size = batch_size
        self.on_frame_processed = on_frame_processed

        self._frame_queue: List[Tuple[np.ndarray, FrameMetadata]] = []
        self._queue_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False

        self._processed_count = 0
        self._processing_time_ms = 0.0
        self._last_process_time = 0.0

        self.logger.info(
            "ProcessingService inicializado",
            batch_enabled=enable_batch,
            batch_size=batch_size,
            detector_type=type(self.detector).__name__
        )

    def _init_detector(self, detector):
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

    def start(self) -> None:
        """Inicia el servicio de procesamiento."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop,
            name="ProcessingService",
            daemon=True
        )
        self._thread.start()
        self.logger.info("Servicio de procesamiento iniciado")

    def stop(self) -> None:
        """Detiene el servicio de procesamiento."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self.logger.info("Servicio de procesamiento detenido")

    def pause(self) -> None:
        """Pausa el procesamiento."""
        self._paused = True
        self.logger.debug("Procesamiento pausado")

    def resume(self) -> None:
        """Reanuda el procesamiento."""
        self._paused = False
        self.logger.debug("Procesamiento reanudado")

    def enqueue_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        """Encola un frame para procesamiento."""
        if not self._running or self._paused:
            return

        with self._queue_lock:
            self._frame_queue.append((frame.copy(), metadata))

            max_queue = self.batch_size * 3 if self.enable_batch else 10
            if len(self._frame_queue) > max_queue:
                self._frame_queue.pop(0)

    def _process_loop(self) -> None:
        """Bucle principal de procesamiento."""
        self.logger.info("Bucle de procesamiento iniciado")

        while self._running:
            try:
                if self._paused:
                    time.sleep(0.01)
                    continue

                if not self._frame_queue:
                    time.sleep(0.001)
                    continue

                if self.enable_batch:
                    self._process_batch()
                else:
                    self._process_single()

            except Exception as e:
                self.logger.error(f"Error en procesamiento: {e}", exc_info=True)
                time.sleep(0.01)

        self.logger.info("Bucle de procesamiento terminado")

    def _process_single(self) -> None:
        """Procesa un solo frame."""
        with self._queue_lock:
            if not self._frame_queue:
                return
            frame, metadata = self._frame_queue.pop(0)

        result = self.process_frame(frame, metadata)
        if result and self.on_frame_processed:
            self.on_frame_processed(result)

    def _process_batch(self) -> None:
        """Procesa un lote de frames."""
        with self._queue_lock:
            batch_size = min(self.batch_size, len(self._frame_queue))
            if batch_size == 0:
                return

            batch = []
            for _ in range(batch_size):
                if self._frame_queue:
                    batch.append(self._frame_queue.pop(0))

        results = self.process_batch(batch)

        for result in results:
            if result and self.on_frame_processed:
                self.on_frame_processed(result)

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

        start_time = time.perf_counter()

        try:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)
            stats = self.counter.process(tracks, frame)
        except Exception as e:
            self.logger.error(f"Error procesando frame: {e}")
            return None

        process_time = (time.perf_counter() - start_time) * 1000

        self._processed_count += 1
        self._processing_time_ms = process_time
        self._last_process_time = time.time()

        return ProcessingResult(
            frame_number=metadata.frame_number,
            detections=detections,
            tracks=tracks,
            stats=stats,
            processed_frame=frame.copy(),
            processing_time_ms=process_time,
            capture_time_ms=0.0,
            timestamp=metadata.timestamp
        )

    def process_batch(self, batch: List[Tuple[np.ndarray, FrameMetadata]]) -> List[ProcessingResult]:
        """
        Procesa un lote de frames.

        Args:
            batch: Lista de tuplas (frame, metadata)

        Returns:
            List[ProcessingResult]: Resultados del procesamiento
        """
        if not batch:
            return []

        results = []

        try:
            if hasattr(self.detector, 'detect_batch'):
                frames = [frame for frame, _ in batch]
                metadatas = [metadata for _, metadata in batch]
                batch_detections = self.detector.detect_batch(frames)

                for i, (frame, metadata, detections) in enumerate(
                    zip(frames, metadatas, batch_detections)
                ):
                    try:
                        start_time = time.perf_counter()

                        tracks = self.tracker.update(detections, frame)
                        stats = self.counter.process(tracks, frame)

                        process_time = (time.perf_counter() - start_time) * 1000

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
                        self._processed_count += 1

                    except Exception as e:
                        self.logger.error(f"Error procesando frame en lote: {e}")
                        continue
            else:
                for frame, metadata in batch:
                    result = self.process_frame(frame, metadata)
                    if result:
                        results.append(result)

        except Exception as e:
            self.logger.error(f"Error en batch processing: {e}")
            for frame, metadata in batch:
                result = self.process_frame(frame, metadata)
                if result:
                    results.append(result)

        return results

    def reset(self) -> None:
        """Reinicia el servicio."""
        self.tracker.reset()
        self.counter.reset()
        self._processed_count = 0
        self._processing_time_ms = 0.0
        self.logger.info("Servicio de procesamiento reiniciado")

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        return {
            'processed_count': self._processed_count,
            'avg_processing_time_ms': self._processing_time_ms,
            'queue_size': len(self._frame_queue),
            'is_running': self._running,
            'is_paused': self._paused,
            'batch_enabled': self.enable_batch,
            'batch_size': self.batch_size,
            'detector_stats': self.detector.get_performance_stats(),
            'tracker_stats': self.tracker.get_stats(),
            'counter_stats': self.counter.get_stats(),
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused
