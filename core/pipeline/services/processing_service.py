"""Servicio de procesamiento de frames.

Responsable de la detección, tracking y conteo de objetos en los frames,
con soporte para procesamiento por lotes y gestión de colas.
"""

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time

import numpy as np

from core.counter import VehicleCounter
from core.detector import OptimizedYOLODetector, YOLODetector
from core.frame_buffer import FrameMetadata
from core.tracker import MultiObjectTracker
from core.types import DetectionList, MetadataDict, StatsDict, TracksDict
from utils.logger import LoggerMixin


@dataclass(slots=True)
class ProcessingResult:
    """Resultado del procesamiento de un frame.

    Attributes:
        frame_number: Número del frame procesado.
        detections: Lista de detecciones encontradas.
        tracks: Diccionario de tracks actualizados.
        stats: Estadísticas del conteo.
        processed_frame: Frame procesado con visualizaciones.
        processing_time_ms: Tiempo de procesamiento en ms.
        capture_time_ms: Tiempo de captura en ms.
        timestamp: Timestamp del frame.

    Example:
        >>> result = ProcessingResult(
        ...     frame_number=42,
        ...     detections=detections,
        ...     tracks=tracks,
        ...     stats=stats,
        ...     processed_frame=frame,
        ...     processing_time_ms=25.3,
        ...     capture_time_ms=5.2,
        ...     timestamp=time.time()
        ... )
    """

    def __init__(
        self,
        frame_number: int,
        detections: DetectionList,
        tracks: TracksDict,
        stats: StatsDict,
        processed_frame: np.ndarray,
        processing_time_ms: float,
        capture_time_ms: float,
        timestamp: float,
        metadata: MetadataDict | None = None
    ):
        self.frame_number = frame_number
        self.detections = detections
        self.tracks = tracks
        self.stats = stats
        self.processed_frame = processed_frame
        self.processing_time_ms = processing_time_ms
        self.capture_time_ms = capture_time_ms
        self.timestamp = timestamp

    @property
    def total_time_ms(self) -> float:
        """Tiempo total desde captura hasta procesamiento."""
        return self.capture_time_ms + self.processing_time_ms


class ProcessingService(LoggerMixin):
    """Servicio especializado en procesamiento de frames.

    Responsabilidades:
        - Detección de objetos en el frame
        - Tracking de objetos
        - Conteo de vehículos
        - Procesamiento por lotes (batch processing)
        - Gestión de cola de frames

    Attributes:
        config: Configuración del sistema.
        detector: Detector de objetos.
        tracker: Tracker de objetos.
        counter: Contador de vehículos.
        enable_batch: Si batch processing está activado.
        batch_size: Tamaño del lote.

    Example:
        >>> service = ProcessingService(config)
        >>> service.start()
        >>> service.enqueue_frame(frame, metadata)
        >>> result = service.process_frame(frame, metadata)
        >>> service.stop()
    """

    def __init__(
        self,
        config,
        detector=None,
        tracker=None,
        counter=None,
        enable_batch: bool = False,
        batch_size: int = 4,
        on_frame_processed: Callable | None = None,
    ):
        self.config = config
        self.detector = self._init_detector(detector)
        self.tracker = tracker or MultiObjectTracker()
        self.counter = counter or VehicleCounter()
        self.enable_batch = enable_batch
        self.batch_size = batch_size
        self.on_frame_processed = on_frame_processed

        self._frame_queue: list[tuple[np.ndarray, FrameMetadata]] = []
        self._queue_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False

        self._processed_count = 0
        self._processing_time_ms = 0.0
        self._last_process_time = 0.0

        self.logger.info(
            "ProcessingService inicializado",
            batch_enabled=enable_batch,
            batch_size=batch_size,
            detector_type=type(self.detector).__name__,
        )

    def _init_detector(self, detector):
        """Inicializa el detector."""
        if detector is not None:
            return detector

        use_optimized = getattr(self.config.optimization, "use_optimized_detector", True)

        if use_optimized:
            try:
                self.logger.info("✅ Detector optimizado activado")
                return OptimizedYOLODetector()
            except Exception as e:
                self.logger.warning(f"Detector optimizado no disponible: {e}. Usando estándar.")

        return YOLODetector()

    def start(self) -> None:
        """Inicia el servicio de procesamiento.

        Note:
            Inicia un thread dedicado para el procesamiento continuo.
            El thread procesa frames de la cola en segundo plano.
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop, name="ProcessingService", daemon=True
        )
        self._thread.start()
        self.logger.info("Servicio de procesamiento iniciado")

    def stop(self) -> None:
        """Detiene el servicio de procesamiento.

        Note:
            Espera a que el thread termine (timeout 2s).
            Limpia la cola de frames pendientes.
        """
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self.logger.info("Servicio de procesamiento detenido")

    def pause(self) -> None:
        """Pausa el procesamiento.

        Note:
            Los frames en cola no se procesan mientras está pausado.
        """
        self._paused = True
        self.logger.debug("Procesamiento pausado")

    def resume(self) -> None:
        """Reanuda el procesamiento."""
        self._paused = False
        self.logger.debug("Procesamiento reanudado")

    def enqueue_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        """Encola un frame para procesamiento.

        Args:
            frame: Frame a procesar.
            metadata: Metadatos del frame.

        Note:
            Si la cola está llena, descarta el frame más antiguo.
        """
        if not self._running or self._paused:
            return

        with self._queue_lock:
            self._frame_queue.append((frame.copy(), metadata))

            max_queue = self.batch_size * 3 if self.enable_batch else 10
            if len(self._frame_queue) > max_queue:
                self._frame_queue.pop(0)

    def _process_loop(self) -> None:
        """Bucle principal de procesamiento.

        Note:
            Se ejecuta en un thread separado.
            Procesa frames de la cola en orden FIFO.
            Soporta procesamiento individual o por lotes.
        """
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
        """Procesa un solo frame.

        Note:
            Toma un frame de la cola y lo procesa.
            El resultado se envía al callback on_frame_processed.
        """
        with self._queue_lock:
            if not self._frame_queue:
                return
            frame, metadata = self._frame_queue.pop(0)

        result = self.process_frame(frame, metadata)
        if result and self.on_frame_processed:
            self.on_frame_processed(result)

    def _process_batch(self) -> None:
        """Procesa un lote de frames.

        Note:
            Toma batch_size frames de la cola.
            Procesa todos en lote para mejor rendimiento.
        """
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

    def process_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> ProcessingResult | None:
        """Procesa un único frame.

        Args:
            frame: Frame a procesar.
            metadata: Metadatos del frame.

        Returns:
            Optional[ProcessingResult]: Resultado del procesamiento o None.

        Note:
            Realiza detección, tracking y conteo en secuencia.
            Mide y registra tiempos de procesamiento.
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
            timestamp=metadata.timestamp,
        )

    def process_batch(
        self, batch: list[tuple[np.ndarray, FrameMetadata]]
    ) -> list[ProcessingResult]:
        """Procesa un lote de frames.

        Args:
            batch: Lista de tuplas (frame, metadata).

        Returns:
            List[ProcessingResult]: Resultados del procesamiento.

        Note:
            Usa detect_batch del detector para mejor rendimiento.
            Cada frame se procesa individualmente para tracking y conteo.
        """
        if not batch:
            return []

        results = []

        try:
            if hasattr(self.detector, "detect_batch"):
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
                            timestamp=metadata.timestamp,
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
        """Reinicia el servicio.

        Note:
            Reinicia tracker y counter.
            Limpia estadísticas y cola.
        """
        self.tracker.reset()
        self.counter.reset()
        self._processed_count = 0
        self._processing_time_ms = 0.0
        self.logger.info("Servicio de procesamiento reiniciado")

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio.

        Returns:
            Dict: Estadísticas incluyendo:
                - processed_count: Frames procesados
                - avg_processing_time_ms: Tiempo promedio de procesamiento
                - queue_size: Tamaño de la cola
                - is_running: Si está en ejecución
                - is_paused: Si está pausado
                - batch_enabled: Si batch está activado
                - batch_size: Tamaño del lote
                - detector_stats: Estadísticas del detector
                - tracker_stats: Estadísticas del tracker
                - counter_stats: Estadísticas del contador

        Example:
            >>> stats = service.get_stats()
            >>> print(f"Avg time: {stats['avg_processing_time_ms']:.2f}ms")
        """
        return {
            "processed_count": self._processed_count,
            "avg_processing_time_ms": self._processing_time_ms,
            "queue_size": len(self._frame_queue),
            "is_running": self._running,
            "is_paused": self._paused,
            "batch_enabled": self.enable_batch,
            "batch_size": self.batch_size,
            "detector_stats": self.detector.get_performance_stats(),
            "tracker_stats": self.tracker.get_stats(),
            "counter_stats": self.counter.get_stats(),
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused
