"""
Pipeline asíncrono con procesamiento paralelo de frames
Optimizado para CPU con límites de recursos
"""

import time
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum, auto
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.detector import YOLODetector
from core.tracker import AdvancedTracker
from core.counter import VehicleCounter
from core.frame_buffer import CircularFrameBuffer, FrameMetadata
from utils.thread_pool import OptimizedThreadPool, TaskPriority
from utils.logger import LoggerMixin
from utils.helpers import get_memory_usage, force_garbage_collection
from core.constants import MEMORY_CHECK_INTERVAL, WINDOW_NAME

from core.pipeline.renderer import FrameRenderer
from core.pipeline.system_info import set_system_status
from core.capture import CaptureManager


class PipelineState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class ProcessingResult:
    frame_number: int
    detections: List[Dict[str, Any]]
    tracks: Dict[int, Dict[str, Any]]
    stats: Dict[str, Any]
    processed_frame: np.ndarray
    processing_time_ms: float
    capture_time_ms: float
    timestamp: float

    @property
    def total_time_ms(self) -> float:
        return self.capture_time_ms + self.processing_time_ms


class AsyncVehicleCountingPipeline(LoggerMixin):
    """
    Pipeline asíncrono de seguimiento de trafico optimizado para CPU

    Arquitectura:
    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐
    │  Capture │───▶│    Buffer    │───▶│   Process    │───▶│ Render  │
    │  Thread  │    │  (Circular)  │    │   Pool       │    │ Thread  │
    └──────────┘    └──────────────┘    └──────────────┘    └─────────┘
    """

    MAX_WORKERS_CPU = 4
    MAX_BUFFER_SIZE_CPU = 20
    MIN_CAPTURE_FPS_CPU = 8.0
    MAX_CAPTURE_FPS_CPU = 15.0

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[AdvancedTracker] = None,
        counter: Optional[VehicleCounter] = None,
        buffer_size: int = 30,
        num_workers: int = 4,
        enable_batch_processing: bool = False,
        batch_size: int = 4,
        render_callback: Optional[Callable] = None,
    ):
        """
        Inicializa el pipeline asíncrono con optimizaciones de memoria.

        Args:
            detector: Detector de objetos (si None, se crea automáticamente).
            tracker: Tracker de objetos (si None, se crea automáticamente).
            counter: Contador de vehículos (si None, se crea automáticamente).
            buffer_size: Tamaño del buffer de frames.
            num_workers: Número de workers para procesamiento paralelo.
            enable_batch_processing: Habilitar procesamiento por lotes.
            batch_size: Tamaño del lote para procesamiento por lotes.
            render_callback: Función callback para renderizado personalizado.
        """
        from config.manager import config_manager
        self.config = config_manager.config
        self.logger.info("Inicializando AsyncVehicleCountingPipeline")

        use_optimized = getattr(self.config.optimization, "use_optimized_detector", True)
        self._using_optimized_detector = False

        if detector is None and use_optimized:
            try:
                from core.detector import OptimizedYOLODetector
                self.detector = OptimizedYOLODetector()
                self._using_optimized_detector = True
                self.logger.info("✅ Detector optimizado (ONNX + Numba) activado")
            except (ImportError, Exception) as e:
                self.logger.warning(
                    f"Detector optimizado no disponible: {e}. Usando estándar."
                )
                self.detector = YOLODetector()
        else:
            self.detector = detector or YOLODetector()

        self.tracker = tracker or AdvancedTracker()
        self.counter = counter or VehicleCounter()

        self.renderer = FrameRenderer(self.config)
        self.renderer.set_pipeline_reference(self)

        is_cpu = self.detector.device == "cpu"
        self._is_cpu_mode = is_cpu

        if is_cpu:
            buffer_size = min(buffer_size, self.MAX_BUFFER_SIZE_CPU)
            num_workers = min(num_workers, self.MAX_WORKERS_CPU)
            enable_batch_processing = True
            batch_size = min(batch_size, 4)
            self.logger.info(
                "Modo CPU detectado - ajustando límites",
                workers=num_workers,
                buffer=buffer_size,
                batch=batch_size
            )

        frame_shape = (self.config.camera.height, self.config.camera.width, 3)
        self._buffer = CircularFrameBuffer(
            max_size=buffer_size,
            frame_shape=frame_shape,
            drop_policy="oldest"
        )

        self._pool = OptimizedThreadPool(
            num_workers=num_workers,
            max_queue_size=buffer_size * 2,
            enable_auto_scale=False,
            logger=self.logger.logger if hasattr(self.logger, 'logger') else None
        )

        self._capture_manager = CaptureManager(
            config=self.config,
            buffer=self._buffer,
            stop_event=self._stop_event,
            pause_event=self._pause_event,
            is_cpu_mode=self._is_cpu_mode,
            capture_interval=self._capture_interval
        )

        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size

        self._batch_buffer: List[tuple] = []
        self._last_batch_submit: float = time.time()
        self._batch_timeout: float = 0.02

        self.render_callback = render_callback
        self.on_frame_processed: Optional[Callable[[ProcessingResult], None]] = None
        self.on_frame_dropped: Optional[Callable[[int], None]] = None

        self._state = PipelineState.IDLE
        self._capture_thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self._frame_count = 0
        self._processed_count = 0
        self._dropped_count = 0
        self._start_time: Optional[float] = None
        self._fps_timer = time.time()
        self._fps_counter = 0
        self._current_fps = 0.0

        self._last_memory_check = time.time()
        self._last_gc_time = time.time()
        self._gc_interval = 30

        self._render_queue: List[ProcessingResult] = []
        self._render_queue_lock = threading.Lock()

        self._max_render_queue = min(3, buffer_size // 3)

        self._source_info = {
            "source": self.config.camera.source,
            "width": self.config.camera.width,
            "height": self.config.camera.height,
            "fps": 0.0,
            "is_opened": False,
        }

        self._dynamic_buffer_size = False
        self._target_buffer_usage = 0.5
        self._last_buffer_adjustment = time.time()
        self._buffer_adjustment_interval = 5.0

        self._flow_control_enabled = True
        self._frame_skip_counter = 0
        self._max_frame_skip = 2
        self._consecutive_skips = 0

        self._health_check_interval = 10.0
        self._last_health_check = time.time()
        self._pipeline_healthy = True
        self._health_issues: List[str] = []

        self._perf_log_interval = 5.0
        self._last_perf_log = time.time()
        self._perf_log_file = Path("data/logs/performance.jsonl")
        self._perf_log_file.parent.mkdir(parents=True, exist_ok=True)

        if is_cpu:
            self._capture_fps_target = self.MIN_CAPTURE_FPS_CPU
            self._capture_interval = 1.0 / self.MIN_CAPTURE_FPS_CPU
            self._min_capture_fps = self.MIN_CAPTURE_FPS_CPU
            self._max_capture_fps = self.MAX_CAPTURE_FPS_CPU
        else:
            self._capture_fps_target = 30.0
            self._capture_interval = 1.0 / 30.0
            self._min_capture_fps = 5.0
            self._max_capture_fps = 30.0

        self._last_capture_time = time.time()

        if getattr(self.config.optimization, "enable_frame_pool", True):
            try:
                from core.frame_pool import FramePool
                self._frame_pool = FramePool(
                    pool_size=3,
                    frame_shape=(self.config.camera.height, self.config.camera.width, 3)
                )
                self.logger.info("✅ FramePool activado (tamaño: 3)")
            except Exception as e:
                self._frame_pool = None
                self.logger.warning(f"FramePool no disponible: {e}")
        else:
            self._frame_pool = None

        self.logger.info(
            "Pipeline asíncrono inicializado",
            buffer_size=buffer_size,
            num_workers=num_workers,
            batch_processing=enable_batch_processing,
            cpu_mode=is_cpu,
            optimized_detector=self._using_optimized_detector,
            frame_pool=self._frame_pool is not None,
            render_queue_size=self._max_render_queue,
            batch_timeout_ms=self._batch_timeout * 1000
        )

    def start(self, source: Optional[str] = None) -> None:
        if self._state == PipelineState.RUNNING:
            self.logger.warning("Pipeline ya está en ejecución")
            return

        self.logger.info("Iniciando pipeline asíncrono")
        self._state = PipelineState.RUNNING
        self._stop_event.clear()
        self._pause_event.clear()

        self._start_time = time.time()
        self._health_issues.clear()
        self._pipeline_healthy = True

        self._batch_buffer.clear()
        self._last_batch_submit = time.time()

        if self._is_cpu_mode:
            self._capture_fps_target = self.MIN_CAPTURE_FPS_CPU
            self._capture_interval = 1.0 / self.MIN_CAPTURE_FPS_CPU

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="AsyncCaptureThread",
            daemon=True,
            args=(source,)
        )
        self._capture_thread.start()

        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="AsyncRenderThread",
            daemon=True
        )
        self._render_thread.start()

        self._capture_manager.set_on_frame_dropped(self.on_frame_dropped)

        self.logger.info("Pipeline asíncrono iniciado")

    def pause(self) -> None:
        if self._state == PipelineState.RUNNING:
            self._state = PipelineState.PAUSED
            self._pause_event.set()
            self.logger.info("Pipeline pausado")

    def resume(self) -> None:
        if self._state == PipelineState.PAUSED:
            self._state = PipelineState.RUNNING
            self._pause_event.clear()
            self.logger.info("Pipeline reanudado")

    def stop(self) -> None:
        if self._state == PipelineState.STOPPED:
            return

        self.logger.info("Deteniendo pipeline...")
        self._state = PipelineState.STOPPING
        self._stop_event.set()
        self._pause_event.set()

        time.sleep(0.1)

        current_thread = threading.current_thread()

        try:
            self._pool.stop(wait=True, timeout=3.0)
        except Exception as e:
            self.logger.warning(f"Error deteniendo pool: {e}")

        if self._capture_thread and self._capture_thread.is_alive():
            if self._capture_thread != current_thread:
                self._capture_thread.join(timeout=2.0)
            else:
                self.logger.warning("No se puede esperar al hilo de captura (es el actual)")
                self._capture_thread = None

        if self._render_thread and self._render_thread.is_alive():
            if self._render_thread != current_thread:
                self._render_thread.join(timeout=2.0)
            else:
                self.logger.warning("No se puede esperar al hilo de renderizado (es el actual)")
                self._render_thread = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            self._buffer.clear()
        except Exception:
            pass

        self._state = PipelineState.STOPPED
        self.logger.info("Pipeline detenido")

        self._log_performance_stats()

    def process_single_frame(self, frame: np.ndarray) -> ProcessingResult:
        if not self._validate_frame(frame):
            raise ValueError("Frame inválido")

        start_time = time.perf_counter()
        frame_number = self._frame_count
        self._frame_count += 1

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        stats = self.counter.process(tracks, frame)
        processed_frame = self._render_frame(frame, tracks, stats)

        processing_time = (time.perf_counter() - start_time) * 1000

        return ProcessingResult(
            frame_number=frame_number,
            detections=detections,
            tracks=tracks,
            stats=stats,
            processed_frame=processed_frame,
            processing_time_ms=processing_time,
            capture_time_ms=0.0,
            timestamp=time.time()
        )

    def get_stats(self) -> Dict[str, Any]:
        pool_stats = self._pool.get_stats()
        buffer_stats = self._buffer.get_stats()

        runtime = time.time() - self._start_time if self._start_time else 0

        return {
            "state": self._state.name,
            "runtime_seconds": runtime,
            "total_frames_captured": self._frame_count,
            "total_frames_processed": self._processed_count,
            "total_frames_dropped": self._dropped_count,
            "current_fps": self._current_fps,
            "avg_fps": self._processed_count / max(1, runtime),
            "buffer": buffer_stats,
            "pool": pool_stats,
            "source": self._source_info,
            "memory": get_memory_usage(),
            "pipeline_healthy": self._pipeline_healthy,
            "health_issues": self._health_issues[-5:],
            "cpu_mode": self._is_cpu_mode,
            "batch_buffer_size": len(self._batch_buffer),
            "components": {
                "detector": self.detector.get_performance_stats(),
                "tracker": self.tracker.get_stats(),
                "counter": self.counter.get_stats(),
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "healthy": self._pipeline_healthy,
            "issues": self._health_issues[-10:],
            "buffer_usage": self._buffer.count / self._buffer.max_size if self._buffer.max_size > 0 else 0,
            "pool_queue": self._pool.get_stats().get('queue_size', 0),
            "fps": self._current_fps,
            "dropped_frames": self._dropped_count,
            "cpu_mode": self._is_cpu_mode,
        }

    def _capture_loop(self, source: Optional[str] = None):
        source = source or self.config.camera.source

        self.logger.info(f"Capturando desde: {source}")

        cap = None
        reconnect_delay = self.config.camera.reconnect_delay

        capture_buffer_size = getattr(self.config.camera, "capture_buffer_size", 1)

        frame_skip_counter = 0
        consecutive_errors = 0
        self._last_capture_time = time.time()

        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - self._last_capture_time < self._capture_interval:
                    time.sleep(0.001)
                    continue
                self._last_capture_time = current_time

                if cap is None or not cap.isOpened():
                    cap = self._connect_source(source, capture_buffer_size)
                    if cap is None:
                        self.logger.warning("No se pudo conectar, reintentando...")
                        time.sleep(reconnect_delay)
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            self._add_health_issue("Fallo de conexión persistente a la fuente")
                            consecutive_errors = 0
                        continue

                    consecutive_errors = 0
                    self._source_info["is_opened"] = True
                    self._source_info["fps"] = cap.get(cv2.CAP_PROP_FPS)

                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        self.logger.warning("Frame de prueba falló, reconectando...")
                        cap.release()
                        cap = None
                        continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    self.logger.warning("Error leyendo frame, reconectando...")
                    cap.release()
                    cap = None
                    continue

                if self._pause_event.is_set():
                    time.sleep(0.01)
                    continue

                if not self._validate_frame(frame):
                    self.logger.debug("Frame inválido, saltando...")
                    continue

                if self._flow_control_enabled:
                    buffer_usage = self._buffer.count / self._buffer.max_size if self._buffer.max_size > 0 else 0

                    if buffer_usage > 0.8:
                        frame_skip_counter += 1
                        if frame_skip_counter < self._max_frame_skip:
                            self._dropped_count += 1
                            self._consecutive_skips += 1
                            if self._consecutive_skips > 5:
                                self._add_health_issue(f"Buffer crítico: {buffer_usage*100:.1f}%")
                                if self._is_cpu_mode:
                                    self._capture_fps_target = max(
                                        self._min_capture_fps,
                                        self._capture_fps_target * 0.9
                                    )
                                    self._capture_interval = 1.0 / self._capture_fps_target
                            continue
                        else:
                            frame_skip_counter = 0
                            self._consecutive_skips = max(0, self._consecutive_skips - 2)

                    elif buffer_usage < 0.3:
                        frame_skip_counter = 0
                        self._consecutive_skips = max(0, self._consecutive_skips - 2)
                        if self._capture_fps_target < self._max_capture_fps:
                            self._capture_fps_target = min(
                                self._max_capture_fps,
                                self._capture_fps_target + 0.5
                            )
                            self._capture_interval = 1.0 / self._capture_fps_target

                    elif buffer_usage < 0.6:
                        self._consecutive_skips = max(0, self._consecutive_skips - 1)

                metadata = FrameMetadata(
                    timestamp=time.time(),
                    frame_number=self._frame_count,
                    source_fps=self._source_info["fps"],
                    capture_time_ms=0.0,
                )

                if not self._buffer.put(frame, metadata):
                    self._dropped_count += 1
                    if self.on_frame_dropped:
                        self.on_frame_dropped(self._frame_count)
                    self.logger.debug(f"Frame {self._frame_count} descartado (buffer lleno)")
                    continue

                self._frame_count += 1

                self._submit_processing_task()
                self._update_capture_fps()
                self._check_pipeline_health()
                self._log_performance_stats()

            except Exception as e:
                self.logger.error(f"Error en captura: {e}", exc_info=True)
                self._add_health_issue(f"Error en captura: {str(e)[:50]}")
                if cap:
                    cap.release()
                    cap = None
                time.sleep(reconnect_delay)

        if cap:
            cap.release()

        self._capture_manager.run(source)
        self.logger.info("Bucle de captura terminado")

    def _connect_source(self, source: str, buffer_size: int = 1):
        try:
            if isinstance(source, str) and source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(source)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

                self.logger.info(f"Conectado a {source} (buffer_size={buffer_size})")
                return cap
            else:
                self.logger.warning(f"No se pudo abrir {source}")
                return None
        except Exception as e:
            self.logger.error(f"Error conectando a {source}: {e}")
            return None

    def _submit_processing_task(self):
        """
        Envía tareas de procesamiento al pool optimizando el batch processing.
        """
        if self.enable_batch_processing:
            result = self._buffer.get(block=False)
            if result is not None:
                frame, metadata = result
                self._batch_buffer.append((frame, metadata))

                should_submit = False
                current_time = time.time()

                if len(self._batch_buffer) >= self.batch_size:
                    should_submit = True

                elif current_time - self._last_batch_submit > self._batch_timeout:
                    should_submit = True

                if should_submit and self._batch_buffer:
                    batch_to_process = self._batch_buffer.copy()
                    self._batch_buffer.clear()
                    self._last_batch_submit = current_time

                    self._pool.submit(
                        self._process_batch,
                        batch_to_process,
                        priority=TaskPriority.HIGH,
                        callback=self._on_batch_complete,
                        error_callback=self._on_processing_error
                    )
        else:
            result = self._buffer.get(block=False)
            if result is not None:
                frame, metadata = result
                self._pool.submit(
                    self._process_single_with_metadata,
                    frame,
                    metadata,
                    priority=TaskPriority.HIGH,
                    callback=self._on_frame_complete,
                    error_callback=self._on_processing_error
                )

    def _process_single_with_metadata(self, frame: np.ndarray, metadata: FrameMetadata) -> Optional[ProcessingResult]:
        """
        Procesa un frame individual con su metadata.
        """
        capture_start = time.perf_counter()

        if self._frame_pool is not None:
            processed_frame = self._frame_pool.acquire()
            processed_frame[:] = frame
        else:
            processed_frame = frame.copy()

        process_start = time.perf_counter()

        try:
            detections = self.detector.detect(processed_frame)
            tracks = self.tracker.update(detections, processed_frame)
            stats = self.counter.process(tracks, processed_frame)
        except Exception as e:
            self.logger.error(f"Error procesando frame {metadata.frame_number}: {e}")
            return None

        process_time = (time.perf_counter() - process_start) * 1000
        capture_time = (time.perf_counter() - capture_start) * 1000

        if self.render_callback:
            processed_frame = self._render_frame(processed_frame, tracks, stats)

        return ProcessingResult(
            frame_number=metadata.frame_number,
            detections=detections,
            tracks=tracks,
            stats=stats,
            processed_frame=processed_frame,
            processing_time_ms=process_time,
            capture_time_ms=capture_time,
            timestamp=metadata.timestamp
        )

    def _process_single(self) -> Optional[ProcessingResult]:
        """Método legacy - mantener para compatibilidad."""
        result = self._buffer.get(block=True, timeout=0.1)
        if result is None:
            return None

        frame, metadata = result
        return self._process_single_with_metadata(frame, metadata)

    def _process_batch(self, batch: List[tuple]) -> List[ProcessingResult]:
        """
        Procesa un lote de frames usando batch inference si está disponible.

        Args:
            batch: Lista de tuplas (frame, metadata)

        Returns:
            List[ProcessingResult]: Resultados del procesamiento del lote
        """
        if not batch:
            return []

        frames = [frame for frame, _ in batch]
        metadatas = [metadata for _, metadata in batch]

        results = []

        try:
            if hasattr(self.detector, 'detect_batch'):
                batch_detections = self.detector.detect_batch(frames)
            else:
                batch_detections = [self.detector.detect(frame) for frame in frames]

            for i, (frame, metadata, detections) in enumerate(zip(frames, metadatas, batch_detections)):
                try:
                    process_start = time.perf_counter()

                    tracks = self.tracker.update(detections, frame)
                    stats = self.counter.process(tracks, frame)

                    process_time = (time.perf_counter() - process_start) * 1000

                    if self.render_callback:
                        processed_frame = self._render_frame(frame, tracks, stats)
                    else:
                        processed_frame = frame.copy()

                    results.append(ProcessingResult(
                        frame_number=metadata.frame_number,
                        detections=detections,
                        tracks=tracks,
                        stats=stats,
                        processed_frame=processed_frame,
                        processing_time_ms=process_time,
                        capture_time_ms=0.0,
                        timestamp=metadata.timestamp
                    ))
                except Exception as e:
                    self.logger.error(f"Error procesando frame en lote: {e}")
                    continue

            return results

        except Exception as e:
            self.logger.error(f"Error en batch processing: {e}")
            for frame, metadata in batch:
                result = self._process_single_with_metadata(frame, metadata)
                if result is not None:
                    results.append(result)
            return results

    def _render_loop(self):
        self.logger.info("Bucle de renderizado iniciado")

        window_name = WINDOW_NAME

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.config.camera.width, self.config.camera.height)
        except Exception as e:
            self.logger.warning(f"Error creando ventana: {e}")

        last_frame = None
        frame_timeout_counter = 0

        while not self._stop_event.is_set():
            try:
                if self._pause_event.is_set():
                    time.sleep(0.01)
                    continue

                result = self._get_render_result()

                if result is None:
                    if last_frame is not None:
                        try:
                            cv2.imshow(window_name, last_frame)
                        except Exception:
                            pass

                    frame_timeout_counter += 1
                    if frame_timeout_counter > 100:
                        try:
                            h, w = self.config.camera.height, self.config.camera.width
                            msg_frame = np.zeros((h, w, 3), dtype=np.uint8)
                            cv2.putText(msg_frame, "Esperando frames...", (w//4, h//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.imshow(window_name, msg_frame)
                        except Exception:
                            pass
                        frame_timeout_counter = 0

                    time.sleep(0.001)
                    continue

                frame_timeout_counter = 0

                if result.processed_frame is not None and result.processed_frame.size > 0:
                    try:
                        cv2.imshow(window_name, result.processed_frame)
                        last_frame = result.processed_frame
                    except Exception as e:
                        self.logger.debug(f"Error mostrando frame: {e}")

                    if self.on_frame_processed:
                        try:
                            self.on_frame_processed(result)
                        except Exception as e:
                            self.logger.error(f"Error en callback de frame: {e}")

                self._processed_count += 1
                self._update_render_fps()

                try:
                    key = cv2.waitKey(1) & 0xFF
                    if not self._handle_key(key):
                        break
                except Exception as e:
                    self.logger.debug(f"Error manejando teclas: {e}")

                self._check_memory()

            except Exception as e:
                self.logger.error(f"Error en renderizado: {e}", exc_info=True)
                self._add_health_issue(f"Error en renderizado: {str(e)[:50]}")
                time.sleep(0.01)

        if self._state == PipelineState.RUNNING:
            set_system_status("RUNNING")
        elif self._state == PipelineState.PAUSED:
            set_system_status("PAUSED")
        elif self._state == PipelineState.STOPPED or self._state == PipelineState.STOPPING:
            set_system_status("STOPPED")
        elif self._state == PipelineState.ERROR:
            set_system_status("ERROR")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.logger.info("Bucle de renderizado terminado")

    def _get_render_result(self) -> Optional[ProcessingResult]:
        with self._render_queue_lock:
            if self._render_queue:
                return self._render_queue.pop(0)
        return None

    def _on_frame_complete(self, result: Optional[ProcessingResult]):
        if result is None:
            return

        with self._render_queue_lock:
            self._render_queue.append(result)

            if len(self._render_queue) > self._max_render_queue:
                dropped = self._render_queue.pop(0)
                self._dropped_count += 1
                self.logger.debug(f"Frame {dropped.frame_number} descartado de cola de renderizado")

    def _on_batch_complete(self, results: List[ProcessingResult]):
        if not results:
            return

        with self._render_queue_lock:
            self._render_queue.extend(results)

            while len(self._render_queue) > self._max_render_queue:
                dropped = self._render_queue.pop(0)
                self._dropped_count += 1

    def _on_processing_error(self, error: Exception):
        self.logger.error(f"Error en procesamiento: {error}")
        self._add_health_issue(f"Error en procesamiento: {str(error)[:50]}")

    def _check_pipeline_health(self):
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return

        self._last_health_check = current_time

        issues = []
        healthy = True

        buffer_usage = self._buffer.count / self._buffer.max_size if self._buffer.max_size > 0 else 0
        if buffer_usage > 0.85:
            issues.append(f"Buffer crítico: {buffer_usage*100:.1f}%")
            healthy = False
        elif buffer_usage > 0.7:
            issues.append(f"Buffer alto: {buffer_usage*100:.1f}%")

        pool_stats = self._pool.get_stats()
        queue_size = pool_stats.get('queue_size', 0)
        if queue_size > 30:
            issues.append(f"Cola grande: {queue_size}")
            healthy = False
        elif queue_size > 15:
            issues.append(f"Cola moderada: {queue_size}")

        if self._current_fps > 0 and self._current_fps < 3:
            issues.append(f"FPS bajo: {self._current_fps:.1f}")
            healthy = False
        elif self._current_fps > 0 and self._current_fps < 8:
            issues.append(f"FPS moderado: {self._current_fps:.1f}")

        total = self._frame_count + self._dropped_count
        if total > 50:
            drop_rate = self._dropped_count / total
            if drop_rate > 0.3:
                issues.append(f"Alta tasa de drop: {drop_rate*100:.1f}%")
                healthy = False
            elif drop_rate > 0.1:
                issues.append(f"Drop rate moderado: {drop_rate*100:.1f}%")

        self._pipeline_healthy = healthy
        for issue in issues:
            self._add_health_issue(issue)

        if not healthy:
            self.logger.warning(f"Pipeline no saludable: {', '.join(issues)}")

    def _add_health_issue(self, issue: str):
        timestamp = time.strftime("%H:%M:%S")
        formatted_issue = f"[{timestamp}] {issue}"
        self._health_issues.append(formatted_issue)
        if len(self._health_issues) > 100:
            self._health_issues = self._health_issues[-50:]

    def _validate_frame(self, frame: np.ndarray) -> bool:
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0:
            return False
        if len(frame.shape) not in [2, 3]:
            return False
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            return False
        return True

    def _render_frame(self, frame: np.ndarray, tracks: dict, stats: dict) -> np.ndarray:
        try:
            from core.pipeline import VehicleCountingPipeline

            pipeline = VehicleCountingPipeline()
            return pipeline._render(frame, tracks, stats)
        except Exception as e:
            self.logger.debug(f"Error en renderizado: {e}")
            result = frame.copy()
            for obj_id, track in tracks.items():
                try:
                    if not isinstance(track, dict):
                        continue
                    centroid = track.get("centroid")
                    if centroid and isinstance(centroid, (tuple, list)) and len(centroid) == 2:
                        cv2.circle(result, tuple(centroid), 4, (0, 255, 0), -1)
                except Exception:
                    continue
            return result

    def _handle_key(self, key: int) -> bool:
        if key == ord('q') or key == 27:
            self.logger.info("Tecla de salida presionada")
            self.stop()
            return False
        elif key == ord(' '):
            if self._state == PipelineState.RUNNING:
                self.pause()
                print("⏸️ Pipeline pausado")
            elif self._state == PipelineState.PAUSED:
                self.resume()
                print("▶️ Pipeline reanudado")
        elif key == ord('s'):
            self._save_screenshot()
        elif key == ord('r'):
            self.counter.reset()
            self.tracker.reset()
            self.logger.info("Contadores y tracker reiniciados")
            print("🔄 Sistema reiniciado")
        elif key == ord('h'):
            self._show_help()
        elif key == ord('d'):
            self._show_diagnostic()
        return True

    def _save_screenshot(self):
        try:
            from utils.helpers import ensure_directory_exists, get_timestamp_filename
            import os

            path = os.path.join(self.config.output.screenshots_dir, get_timestamp_filename("capture", "jpg"))
            ensure_directory_exists(self.config.output.screenshots_dir)

            with self._render_queue_lock:
                if self._render_queue:
                    last_result = self._render_queue[-1]
                    if last_result.processed_frame is not None:
                        cv2.imwrite(path, last_result.processed_frame)
                        self.logger.info(f"Captura guardada: {path}")
                        print(f"📸 Captura guardada: {path}")
        except Exception as e:
            self.logger.error(f"Error guardando captura: {e}")

    def _show_help(self):
        print("""
        ═══════════════════════════════════════════════════
        🎮 CONTROLES DEL SISTEMA (Modo Asíncrono)
        ═══════════════════════════════════════════════════
        q / ESC  → Salir
        SPACE    → Pausar/Reanudar
        s        → Captura de pantalla
        r        → Reiniciar contadores
        d        → Mostrar diagnóstico
        h        → Esta ayuda
        ═══════════════════════════════════════════════════
        """)

    def _show_diagnostic(self):
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("🔍 DIAGNÓSTICO DEL PIPELINE")
        print("=" * 60)

        print(f"\n📊 ESTADO GENERAL:")
        print(f"  Estado: {stats['state']}")
        print(f"  Modo CPU: {'✅' if stats['cpu_mode'] else '❌'}")
        print(f"  Tiempo ejecución: {stats['runtime_seconds']:.1f}s")
        print(f"  FPS: {stats['current_fps']:.1f}")
        print(f"  Salud: {'✅ OK' if stats['pipeline_healthy'] else '⚠️ Problemas'}")

        print(f"\n📹 FRAMES:")
        print(f"  Capturados: {stats['total_frames_captured']}")
        print(f"  Procesados: {stats['total_frames_processed']}")
        print(f"  Descartados: {stats['total_frames_dropped']}")
        drop_rate = stats['total_frames_dropped'] / max(1, stats['total_frames_captured'])
        print(f"  Tasa de drop: {drop_rate*100:.1f}%")
        print(f"  Batch buffer: {stats['batch_buffer_size']}")

        print(f"\n📦 BUFFER:")
        print(f"  Uso: {stats['buffer']['capacity_ratio']*100:.1f}%")
        print(f"  Tamaño: {stats['buffer']['size']}/{stats['buffer']['max_size']}")
        print(f"  Estado: {stats['buffer']['status']}")

        print(f"\n🧵 THREAD POOL:")
        print(f"  Workers: {stats['pool']['num_workers']}")
        print(f"  Cola: {stats['pool']['queue_size']}")
        print(f"  Tareas completadas: {stats['pool']['total_tasks_completed']}")

        if stats.get('health_issues'):
            print(f"\n⚠️ ISSUES RECIENTES:")
            for issue in stats['health_issues'][-5:]:
                print(f"  • {issue}")

        print("=" * 60 + "\n")

    def _update_capture_fps(self):
        self._fps_counter += 1
        if time.time() - self._fps_timer >= 1.0:
            self._current_fps = self._fps_counter
            self._fps_counter = 0
            self._fps_timer = time.time()

    def _update_render_fps(self):
        pass

    def _check_memory(self):
        """Verifica memoria y ejecuta GC si es necesario."""
        current_time = time.time()

        if current_time - self._last_memory_check >= MEMORY_CHECK_INTERVAL:
            self._last_memory_check = current_time

            mem = get_memory_usage()
            mem_percent = mem.get("percent", 0)

            if mem_percent > 70:
                self.logger.warning(f"Memoria alta: {mem_percent:.1f}%, limpiando...")
                self.detector.clear_cache()
                force_garbage_collection()

                if hasattr(self, '_render_queue'):
                    with self._render_queue_lock:
                        self._render_queue.clear()

                if mem_percent > 80:
                    self._add_health_issue(f"Memoria crítica: {mem_percent:.1f}%")
                    import gc
                    gc.collect()

        if current_time - self._last_gc_time >= self._gc_interval:
            self._last_gc_time = current_time
            import gc
            gc.collect()

    def _log_performance_stats(self):
        """
        Registra estadísticas de rendimiento en formato JSON.
        """
        current_time = time.time()
        if current_time - self._last_perf_log < self._perf_log_interval:
            return

        self._last_perf_log = current_time

        try:
            runtime = time.time() - self._start_time if self._start_time else 0
            mem = get_memory_usage()

            stats = {
                "timestamp": datetime.now().isoformat(),
                "runtime_seconds": round(runtime, 2),
                "fps": round(self._current_fps, 2),
                "frame_count": self._frame_count,
                "processed_count": self._processed_count,
                "dropped_count": self._dropped_count,
                "processing_time_ms": round(self.processing_time, 2) if hasattr(self, 'processing_time') else 0,
                "active_tracks": len(self.tracker.tracks) if hasattr(self.tracker, 'tracks') else 0,
                "buffer_usage": round(self._buffer.count / max(1, self._buffer.max_size), 3),
                "pool_queue": self._pool.get_stats().get('queue_size', 0),
                "memory_usage_mb": round(mem.get("rss_mb", 0), 2),
                "memory_percent": round(mem.get("percent", 0), 2),
                "cpu_mode": self._is_cpu_mode,
                "pipeline_healthy": self._pipeline_healthy,
            }

            with open(self._perf_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

        except Exception as e:
            self.logger.debug(f"Error en logging de rendimiento: {e}")

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == PipelineState.RUNNING

    @property
    def is_paused(self) -> bool:
        return self._state == PipelineState.PAUSED

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def total_frames(self) -> int:
        return self._frame_count

    @property
    def processed_frames(self) -> int:
        return self._processed_count

    @property
    def dropped_frames(self) -> int:
        return self._dropped_count

    @property
    def is_healthy(self) -> bool:
        return self._pipeline_healthy

    @property
    def is_cpu_mode(self) -> bool:
        return self._is_cpu_mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
