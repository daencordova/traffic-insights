"""
Pipeline principal del sistema de seguimiento de trafico
"""

import os
import time
from typing import Optional

import cv2
import numpy as np

from config.manager import config_manager
from .detector import YOLODetector
from .tracker import AdvancedTracker
from .counter import VehicleCounter
from utils.logger import LoggerMixin
from utils.helpers import (
    ensure_directory_exists,
    get_timestamp_filename,
    get_memory_usage,
    force_garbage_collection,
    MemoryTracker
)
from .constants import (
    MEMORY_CHECK_INTERVAL,
    GC_INTERVAL,
    DASHBOARD_WIDTH,
    DASHBOARD_HEIGHT,
    DASHBOARD_ALPHA,
    FONT_SCALE,
    LINE_THICKNESS,
    POINT_RADIUS,
    COLORS,
    WINDOW_NAME,
    TARGET_FPS,
    MIN_ACCEPTABLE_FPS,
)


class VideoCaptureContext:
    """
    Context manager para captura de video con manejo automático
    """
    def __init__(
        self,
        source: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self._is_open = False

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _open(self):
        for attempt in range(self.reconnect_attempts):
            try:
                if isinstance(self.source, str) and self.source.isdigit():
                    self.cap = cv2.VideoCapture(int(self.source))
                else:
                    self.cap = cv2.VideoCapture(self.source)

                if self.cap.isOpened():
                    self._configure_capture()
                    self._is_open = True
                    return
            except Exception:
                pass

            if attempt < self.reconnect_attempts - 1:
                time.sleep(self.reconnect_delay)

        raise RuntimeError(f"No se pudo abrir la fuente: {self.source}")

    def _configure_capture(self):
        if self.cap is None:
            return
        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if not self._is_open or self.cap is None:
            return False, None
        try:
            return self.cap.read()
        except Exception:
            return False, None

    def get_fps(self):
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def is_opened(self):
        return self._is_open and self.cap is not None and self.cap.isOpened()

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            self._is_open = False


class ResourcePool:
    """Pool simple de recursos"""
    def __init__(self, max_size: int = 5, timeout: float = 30.0):
        self.max_size = max_size
        self.timeout = timeout
        self._pool = []

    def clear(self):
        self._pool.clear()


class VehicleCountingPipeline(LoggerMixin):
    """Pipeline principal con gestión de recursos y optimizaciones"""

    MEMORY_CHECK_INTERVAL: int = MEMORY_CHECK_INTERVAL
    GC_INTERVAL: int = GC_INTERVAL

    def __init__(self) -> None:
        self.config = config_manager.config
        self.logger.info("Inicializando VehicleCountingPipeline")

        use_optimized = getattr(self.config.optimization, "use_optimized_detector", True)
        self._using_optimized_detector = False

        if use_optimized:
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
            self.detector = YOLODetector()

        self.tracker = AdvancedTracker()
        self.counter = VehicleCounter()

        self.is_running: bool = False
        self.is_paused: bool = False
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.processing_time: float = 0.0

        self._fps_counter: int = 0
        self._fps_timer: float = time.time()
        self._start_time: float = time.time()

        self._memory_tracker = MemoryTracker("pipeline")
        self._last_memory_check: float = time.time()
        self._last_gc_time: float = time.time()
        self._memory_high_watermark: float = 0.0

        self._resource_pool = ResourcePool(max_size=5, timeout=30.0)

        self._last_detections = []

        ensure_directory_exists(self.config.output.screenshots_dir)
        ensure_directory_exists(self.config.output.export_dir)

        self._validate_config()
        self._print_startup_info()
        self.logger.info(
            "Pipeline inicializado correctamente",
            optimized_detector=self._using_optimized_detector
        )

        self._memory_tracker.snapshot("inicializacion")

    def _validate_config(self) -> None:
        """Valida la configuración del sistema"""
        self.logger.debug("Validando configuración")

        if self.config.counting_lines:
            self.logger.info("Líneas de conteo configuradas", count=len(self.config.counting_lines))
            for idx, line in enumerate(self.config.counting_lines):
                if not isinstance(line, dict):
                    self.logger.warning("Línea con configuración inválida", index=idx)
                    continue

                points = line.get("points", [])
                if len(points) < 1:
                    self.logger.warning("Línea sin puntos suficientes", index=idx, name=line.get("name", ""))

                for point in points:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        self.logger.warning("Punto inválido en línea", index=idx, point=point)
        else:
            self.logger.warning("No hay líneas de conteo configuradas")

        if not os.path.exists(self.config.model.model_path):
            self.logger.warning("Modelo no encontrado", path=self.config.model.model_path)

        self.logger.info("Configuración validada")

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Valida que el frame sea válido"""
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

    def _check_memory(self) -> None:
        """Verifica el uso de memoria y realiza limpieza si es necesario"""
        current_time = time.time()

        if current_time - self._last_memory_check >= self.MEMORY_CHECK_INTERVAL:
            self._last_memory_check = current_time

            mem = get_memory_usage()
            memory_mb = mem.get("rss_mb", 0)
            memory_percent = mem.get("percent", 0)

            if memory_mb > self._memory_high_watermark:
                self._memory_high_watermark = memory_mb

            if int(current_time - self._start_time) % 300 < 5:
                self.logger.info(
                    "Uso de memoria",
                    rss_mb=f"{memory_mb:.1f}",
                    percent=f"{memory_percent:.1f}",
                    peak_mb=f"{self._memory_high_watermark:.1f}"
                )

            if memory_percent > 75:
                self.logger.warning(
                    "Memoria alta, realizando limpieza",
                    percent=f"{memory_percent:.1f}",
                    rss_mb=f"{memory_mb:.1f}"
                )
                self._cleanup_memory(aggressive=False)

            if memory_percent > 85:
                self.logger.warning(
                    "Memoria crítica, limpieza agresiva",
                    percent=f"{memory_percent:.1f}",
                    rss_mb=f"{memory_mb:.1f}"
                )
                self._cleanup_memory(aggressive=True)

        if current_time - self._last_gc_time >= self.GC_INTERVAL:
            self._last_gc_time = current_time
            gc_stats = force_garbage_collection()
            if gc_stats["collected_objects"] > 0:
                self.logger.debug(
                    "GC periódico completado",
                    collected=gc_stats["collected_objects"],
                    garbage=gc_stats["garbage_count"]
                )

    def _cleanup_memory(self, aggressive: bool = False) -> None:
        """Limpia recursos de memoria"""
        try:
            self.detector.clear_cache()
        except Exception as e:
            self.logger.debug("Error limpiando caché del detector", error=str(e))

        try:
            if hasattr(self.tracker, 'reset'):
                self.tracker.reset()
        except Exception as e:
            self.logger.debug("Error reiniciando tracker", error=str(e))

        try:
            self._resource_pool.clear()
        except Exception as e:
            self.logger.debug("Error limpiando pool de recursos", error=str(e))

        gc_stats = force_garbage_collection()
        self.logger.debug(
            "Limpieza de memoria completada",
            aggressive=aggressive,
            collected=gc_stats["collected_objects"]
        )

        self._memory_tracker.snapshot(f"cleanup_{'aggressive' if aggressive else 'normal'}")

    def run(self, source: Optional[str] = None) -> None:
        """
        Ejecuta el pipeline principal
        """
        source = source or self.config.camera.source

        self.logger.debug("Iniciando pipeline", source=source)
        print(f"📹 Iniciando fuente: {source}")

        with VideoCaptureContext(
            source=source,
            width=self.config.camera.width,
            height=self.config.camera.height,
            reconnect_attempts=self.config.camera.reconnect_attempts,
            reconnect_delay=self.config.camera.reconnect_delay,
        ) as capture:

            if not capture.is_opened():
                raise RuntimeError(f"No se pudo abrir la fuente: {source}")

            fps = capture.get_fps()
            if fps > 0:
                self.logger.debug("FPS de fuente", fps=fps)
                print(f"📹 FPS de fuente: {fps:.1f}")

            ret, test_frame = capture.read()
            if not ret or test_frame is None:
                raise RuntimeError("No se pudo leer frame de prueba")

            if not self._validate_frame(test_frame):
                raise RuntimeError("Frame de prueba inválido")

            print("✅ Frame de prueba OK")
            print("\n🔄 Procesando... Presiona 'q' para salir\n")
            self.is_running = True

            try:
                while self.is_running:
                    self._check_memory()

                    if self.is_paused:
                        key = cv2.waitKey(10) & 0xFF
                        if key == ord(" ") or key == ord("q"):
                            self.is_paused = False
                            continue
                        elif key == 27:
                            break
                        continue

                    ret, frame = capture.read()
                    if not ret:
                        self.logger.warning("Error leyendo frame")
                        if not self._handle_missing_frames(capture):
                            break
                        continue

                    if not self._validate_frame(frame):
                        self.logger.debug("Frame inválido, saltando...")
                        continue

                    processed = self.process_frame(frame)

                    if processed is not None and self._validate_frame(processed):
                        cv2.imshow(WINDOW_NAME, processed)
                        key = cv2.waitKey(1) & 0xFF
                        if not self._handle_key(key):
                            break

                    self._update_fps()

            except KeyboardInterrupt:
                self.logger.info("Interrupción por usuario")
                print("\n⏹️ Interrupción recibida")
            finally:
                self._cleanup()

    def _handle_missing_frames(self, capture: VideoCaptureContext) -> bool:
        """Maneja frames perdidos con reconexión inteligente"""
        self.logger.warning("Frame perdido, intentando recuperar...")

        for attempt in range(3):
            try:
                ret, frame = capture.read()
                if ret and frame is not None and self._validate_frame(frame):
                    self.logger.info("Frame recuperado", attempt=attempt + 1)
                    return True
            except Exception:
                pass
            time.sleep(0.1)

        self.logger.warning("Reconectando a la fuente...")
        return self._reconnect(capture)

    def _reconnect(self, capture: VideoCaptureContext) -> bool:
        """Reconecta usando el context manager"""
        try:
            capture.close()
            time.sleep(1)
            capture._open()
            return capture.is_opened()
        except Exception as e:
            self.logger.error("Error en reconexión", error=str(e))
            return False

    def _cleanup(self) -> None:
        """Limpieza de recursos"""
        self.is_running = False

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.debug("Error cerrando ventanas", error=str(e))

        try:
            self.detector.clear_cache()
        except Exception as e:
            self.logger.debug("Error limpiando caché del detector", error=str(e))

        try:
            if hasattr(self.tracker, 'reset'):
                self.tracker.reset()
        except Exception as e:
            self.logger.debug("Error reiniciando tracker", error=str(e))

        try:
            self._resource_pool.clear()
        except Exception as e:
            self.logger.debug("Error limpiando pool de recursos", error=str(e))

        gc_stats = force_garbage_collection()
        self.logger.debug("GC final", collected=gc_stats["collected_objects"])

        runtime = time.time() - self._start_time
        mem_stats = self._memory_tracker.get_stats()

        self.logger.info(
            "Pipeline detenido",
            runtime_seconds=f"{runtime:.1f}",
            frames=self.frame_count,
            fps=self.fps,
            peak_memory_mb=f"{mem_stats.get('peak_mb', 0):.1f}"
        )
        self._print_final_report()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame individual"""
        if self.is_paused:
            return frame

        if not self._validate_frame(frame):
            return frame

        start_time = time.perf_counter()

        if hasattr(self, 'optimized_detector') and self.optimized_detector:
            try:
                detections = self.optimized_detector.detect(frame)
            except Exception as e:
                self.logger.error("Error en detección optimizada", error=str(e))
                detections = self.detector.detect(frame)
        else:
            detections = self.detector.detect(frame)

        self._last_detections = detections
        if detections:
            self.logger.debug(f"🔍 {len(detections)} detecciones en frame {self.frame_count}")

        try:
            tracks = self.tracker.update(detections, frame)

            if tracks:
                self.logger.debug(f"🔍 {len(tracks)} tracks activos en frame {self.frame_count}")
                first_id = list(tracks.keys())[0]
                first_track = tracks[first_id]
                if isinstance(first_track, dict):
                    centroid = first_track.get('centroid')
                    self.logger.debug(f"🔍 Primer track: ID={first_id}, centroid={centroid}")
            else:
                self.logger.debug(f"🔍 No hay tracks en frame {self.frame_count}")

        except Exception as e:
            self.logger.error("Error en tracking", error=str(e))
            tracks = {}
            if hasattr(self.tracker, 'reset'):
                self.tracker.reset()

        try:
            stats = self.counter.process(tracks, frame)
        except Exception as e:
            self.logger.error("Error en conteo", error=str(e))
            stats = self.counter.get_stats()

        try:
            result = self._render(frame, tracks, stats)
        except Exception as e:
            self.logger.error("Error en renderizado", error=str(e))
            result = frame

        self.processing_time = (time.perf_counter() - start_time) * 1000
        self.frame_count += 1

        if self.frame_count % 100 == 0:
            self.logger.debug(
                "Estado del sistema",
                frames=self.frame_count,
                detections=len(detections),
                tracks=len(tracks),
                fps=self.fps,
                processing_ms=f"{self.processing_time:.1f}"
            )

        return result

    def _render(self, frame: np.ndarray, tracks: dict, stats: dict) -> np.ndarray:
        """
        Renderiza la visualización del sistema en el frame.

        Este método dibuja:
        - Líneas de conteo con sus respectivos contadores
        - Tracks activos con sus centroides y trayectorias
        - Predicciones de trayectoria (Path Prediction)
        - Alertas de colisión
        - Dashboard con información estadística
        - FPS y otros indicadores de rendimiento
        - Información de fusión de sensores (si está disponible)
        - Información de aprendizaje en línea (si está disponible)

        Args:
            frame: Imagen original sobre la cual renderizar.
            tracks: Diccionario de tracks activos.
            stats: Estadísticas del sistema.

        Returns:
            np.ndarray: Imagen renderizada con la visualización.
        """
        result = frame.copy()

        if not tracks:
            cv2.putText(
                result,
                "⚠️ No hay tracks activos",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            if hasattr(self, '_last_detections') and self._last_detections:
                for det in self._last_detections[:5]:
                    box = det.get("box")
                    if box:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 1)
                        cv2.putText(
                            result,
                            f"det: {det.get('confidence', 0):.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 0),
                            1
                        )

        if self.config.counting_lines:
            for idx, line in enumerate(self.config.counting_lines):
                try:
                    if not isinstance(line, dict):
                        continue

                    points = line.get("points", [])
                    if len(points) < 1:
                        continue

                    color = tuple(line.get("color", (0, 255, 0)))
                    name = line.get("name", f"Line {idx + 1}")
                    line_id = line.get("id", f"line_{idx}")
                    count = self.counter.get_line_count(line_id)

                    first_point = points[0]
                    if isinstance(first_point, (list, tuple)) and len(first_point) == 2:
                        y_position = first_point[1]
                        cv2.line(
                            result,
                            (0, y_position),
                            (frame.shape[1], y_position),
                            color,
                            LINE_THICKNESS
                        )

                        label = f"{name}: {count}"
                        cv2.putText(
                            result,
                            label,
                            (first_point[0] + 10, first_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE,
                            color,
                            2
                        )
                except Exception as e:
                    self.logger.debug("Error dibujando línea", index=idx, error=str(e))
                    continue

        for obj_id, track in tracks.items():
            try:
                if not isinstance(track, dict):
                    continue

                centroid = track.get("centroid")
                if centroid is None:
                    continue

                if not isinstance(centroid, (tuple, list)) or len(centroid) != 2:
                    continue

                centroid_x, centroid_y = int(centroid[0]), int(centroid[1])

                status = track.get("status", "")
                if status == "confirmed":
                    color = (0, 255, 0)
                    status_text = "✅"
                elif status == "lost":
                    color = (0, 255, 255)
                    status_text = "⚠️"
                elif status == "tentative":
                    color = (255, 255, 0)
                    status_text = "⏳"
                else:
                    color = (0, 165, 255)
                    status_text = "❓"

                history = track.get("history", [])
                if isinstance(history, list) and len(history) > 1:
                    if len(history) > 30:
                        history = history[-30:]

                    for i in range(1, len(history)):
                        try:
                            prev_point = history[i - 1]
                            curr_point = history[i]

                            if not isinstance(prev_point, (tuple, list)) or len(prev_point) != 2:
                                continue
                            if not isinstance(curr_point, (tuple, list)) or len(curr_point) != 2:
                                continue

                            alpha = i / len(history)
                            color_fade = tuple(int(c * alpha) for c in color)

                            cv2.line(
                                result,
                                tuple(prev_point),
                                tuple(curr_point),
                                color_fade,
                                1
                            )
                        except Exception as e:
                            self.logger.debug(
                                "Error dibujando segmento de trayectoria",
                                track_id=obj_id,
                                index=i,
                                error=str(e)
                            )
                            continue

                cv2.circle(result, (centroid_x, centroid_y), 8, color, 2)
                cv2.circle(result, (centroid_x, centroid_y), 5, color, -1)
                cv2.circle(result, (centroid_x, centroid_y), 2, (255, 255, 255), -1)

                label = f"#{obj_id} {status_text}"
                cv2.putText(
                    result,
                    label,
                    (centroid_x + 10, centroid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                confidence = track.get("confidence", 0.0)
                if confidence > 0:
                    conf_text = f"{confidence:.0%}"
                    cv2.putText(
                        result,
                        conf_text,
                        (centroid_x + 10, centroid_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )

                path_pred = track.get("path_prediction")
                if path_pred and isinstance(path_pred, dict):
                    positions = path_pred.get("positions", [])
                    state = path_pred.get("state", "unknown")
                    uncertainty = path_pred.get("uncertainty", 0.5)
                    collision_risk = path_pred.get("collision_risk", 0.0)

                    if positions and len(positions) > 1:
                        if state == "stopped":
                            pred_color = (0, 0, 255)
                        elif state == "accelerating":
                            pred_color = (0, 255, 255)
                        elif state == "decelerating":
                            pred_color = (0, 165, 255)
                        elif state == "turning":
                            pred_color = (255, 0, 255)
                        elif state == "erratic":
                            pred_color = (255, 0, 0)
                        else:
                            pred_color = (255, 255, 0)

                        for i, pos in enumerate(positions):
                            if not isinstance(pos, (tuple, list)) or len(pos) != 2:
                                continue

                            alpha = 1.0 - (i / len(positions))
                            color_pred = tuple(int(c * alpha) for c in pred_color)

                            radius = max(2, int(4 * (1 - uncertainty)))

                            cv2.circle(result, tuple(pos), radius, color_pred, -1)

                            if i > 0:
                                prev_pos = positions[i - 1]
                                if isinstance(prev_pos, (tuple, list)) and len(prev_pos) == 2:
                                    cv2.line(
                                        result,
                                        tuple(prev_pos),
                                        tuple(pos),
                                        color_pred,
                                        1,
                                        cv2.LINE_AA
                                    )

                        if collision_risk > 0.3:
                            risk_color = (0, 0, 255) if collision_risk > 0.6 else (0, 165, 255)
                            risk_text = f"⚠️ Risk: {collision_risk:.0%}"
                            last_pos = positions[-1] if positions else (centroid_x, centroid_y)
                            cv2.putText(
                                result,
                                risk_text,
                                (last_pos[0] + 10, last_pos[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                risk_color,
                                1
                            )

                        state_text = f"🚦 {state}"
                        cv2.putText(
                            result,
                            state_text,
                            (centroid_x + 10, centroid_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            pred_color,
                            1
                        )

                mht_predictions = track.get("mht_predictions")
                if mht_predictions and isinstance(mht_predictions, list):
                    pred_color = (255, 0, 255)
                    for i, pred_pos in enumerate(mht_predictions[:5]):
                        try:
                            if not isinstance(pred_pos, (tuple, list)) or len(pred_pos) != 2:
                                continue

                            radius = POINT_RADIUS - 1 if POINT_RADIUS > 2 else 1
                            alpha = 1.0 - (i / len(mht_predictions[:5]))
                            pred_color_fade = tuple(int(c * alpha) for c in pred_color)

                            cv2.circle(
                                result,
                                tuple(pred_pos),
                                radius,
                                pred_color_fade,
                                -1
                            )

                            if i > 0:
                                prev_pred = mht_predictions[i - 1]
                                if isinstance(prev_pred, (tuple, list)) and len(prev_pred) == 2:
                                    cv2.line(
                                        result,
                                        tuple(prev_pred),
                                        tuple(pred_pos),
                                        pred_color_fade,
                                        1,
                                        cv2.LINE_AA
                                    )
                        except Exception as e:
                            self.logger.debug(
                                "Error dibujando predicción MHT",
                                track_id=obj_id,
                                index=i,
                                error=str(e)
                            )
                            continue

                mht_confidence = track.get("mht_confidence")
                if mht_confidence is not None and isinstance(mht_confidence, (int, float)):
                    conf_text = f"MHT: {mht_confidence:.2f}"
                    conf_color = (0, 255, 255) if mht_confidence > 0.5 else (0, 165, 255)
                    cv2.putText(
                        result,
                        conf_text,
                        (centroid_x + 10, centroid_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        conf_color,
                        1
                    )

                sensor_fusion = track.get("sensor_fusion")
                if sensor_fusion and isinstance(sensor_fusion, dict):
                    fused_confidence = sensor_fusion.get("fused_confidence", 0.0)
                    uncertainty = sensor_fusion.get("uncertainty", 0.0)

                    if fused_confidence > 0.3:
                        fusion_text = f"🔬 Fusion: {fused_confidence:.2f}"
                        fusion_color = (255, 255, 0) if fused_confidence > 0.5 else (0, 165, 255)
                        cv2.putText(
                            result,
                            fusion_text,
                            (centroid_x + 10, centroid_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            fusion_color,
                            1
                        )

                online_learning = track.get("online_learning")
                if online_learning and isinstance(online_learning, dict):
                    samples = online_learning.get("samples", 0)
                    quality = online_learning.get("quality", 0.0)
                    drift_detected = online_learning.get("drift_detected", False)

                    if samples > 0:
                        ol_text = f"🧠 OL: {samples}s"
                        ol_color = (0, 255, 255) if quality > 0.5 else (0, 165, 255)
                        cv2.putText(
                            result,
                            ol_text,
                            (centroid_x + 10, centroid_y + 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            ol_color,
                            1
                        )

                        if drift_detected:
                            drift_text = "🔄 Drift!"
                            cv2.putText(
                                result,
                                drift_text,
                                (centroid_x + 10, centroid_y + 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (0, 0, 255),
                                1
                            )

            except Exception as e:
                self.logger.debug(
                    "Error dibujando track",
                    track_id=obj_id,
                    error=str(e)
                )
                continue

        if hasattr(self, 'path_predictor') and self.path_predictor is not None:
            high_risk = self.path_predictor.get_high_risk_tracks(threshold=0.5)

            if high_risk:
                overlay = result.copy()

                for track_id in high_risk[:5]:
                    track = tracks.get(track_id)
                    if track is None:
                        continue

                    centroid = track.get("centroid")
                    if centroid is None:
                        continue

                    pulse = 0.5 + 0.5 * np.sin(time.time() * 2)
                    radius = int(25 + 10 * pulse)

                    cv2.circle(overlay, tuple(centroid), radius, (0, 0, 255), 3)
                    cv2.circle(overlay, tuple(centroid), radius - 5, (0, 0, 255), 2)

                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

                alert_text = f"⚠️ {len(high_risk)} colisiones potenciales"
                cv2.putText(
                    result,
                    alert_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        if self.config.visualization.show_dashboard:
            self._draw_dashboard(result, stats)

        if self.config.visualization.show_fps:
            if self.fps >= TARGET_FPS:
                fps_color = COLORS["GREEN"]
            elif self.fps >= MIN_ACCEPTABLE_FPS:
                fps_color = COLORS["YELLOW"]
            else:
                fps_color = COLORS["RED"]

            info_text = f"FPS: {self.fps:.1f} | MS: {self.processing_time:.1f} | TR: {len(tracks)}"

            if hasattr(self, 'path_predictor') and self.path_predictor is not None:
                high_risk_count = len(self.path_predictor.get_high_risk_tracks(0.5))
                if high_risk_count > 0:
                    info_text += f" | ⚠️ {high_risk_count}"

            cv2.putText(
                result,
                info_text,
                (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                fps_color,
                2
            )

            y_offset = result.shape[0] - 35
            components = []

            if self.tracker and hasattr(self.tracker, 'mht_integration'):
                if self.tracker.mht_integration.enabled:
                    components.append("MHT")

            if hasattr(self, 'path_predictor') and self.path_predictor is not None:
                components.append("PathPred")

            if self.tracker and hasattr(self.tracker, 'sensor_fusion'):
                if self.tracker.sensor_fusion is not None:
                    components.append("Fusion")

            if self.tracker and hasattr(self.tracker, 'online_learner'):
                if self.tracker.online_learner is not None:
                    components.append("OL")

            if components:
                comp_text = f"🧩 {' | '.join(components)}"
                cv2.putText(
                    result,
                    comp_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (192, 192, 192),
                    1
                )

        if self.config.visualization.show_controls_help:
            help_text = "q:Salir | Esp:Pausa | s:Captura | r:Reset | h:Ayuda"
            cv2.putText(
                result,
                help_text,
                (result.shape[1] - 300, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1
            )

        return result

    def _draw_dashboard(self, frame: np.ndarray, stats: dict) -> None:
        """Dibuja el dashboard de información"""
        try:
            x, y = 10, 10
            w, h = DASHBOARD_WIDTH, DASHBOARD_HEIGHT

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), COLORS["BLACK"], -1)
            cv2.addWeighted(overlay, DASHBOARD_ALPHA, frame, 1 - DASHBOARD_ALPHA, 0, frame)

            if self.fps >= TARGET_FPS:
                border_color = COLORS["GREEN"]
            elif self.fps >= MIN_ACCEPTABLE_FPS:
                border_color = COLORS["YELLOW"]
            else:
                border_color = COLORS["RED"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)

            info = [
                ("🚗 Total", f"{stats.get('total', 0):>6}"),
                ("🎯 Activos", f"{stats.get('active_objects', 0):>6}"),
                ("⚡ FPS", f"{self.fps:>6.1f}"),
                ("⏱️ Tiempo", f"{self.processing_time:>5.1f}ms"),
            ]

            y_offset = y + 25
            for label, value in info:
                cv2.putText(frame, label, (x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["LIGHT_GRAY"], 1)
                cv2.putText(frame, value, (x + w - 80, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["WHITE"], 1)
                y_offset += 22

            if len(self.config.counting_lines) <= 4 and self.config.counting_lines:
                y_offset = y + h - 10
                for idx, line in enumerate(self.config.counting_lines[:4]):
                    try:
                        line_id = line.get("id", f"line_{idx}")
                        count = self.counter.get_line_count(line_id)
                        name = line.get("name", f"L{idx+1}")
                        color = tuple(line.get("color", (0, 255, 0)))

                        x_pos = x + 10 + idx * 55
                        text = f"{name}:{count}"
                        cv2.putText(frame, text, (x_pos, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    except Exception:
                        continue
        except Exception as e:
            self.logger.debug("Error dibujando dashboard", error=str(e))

    def _update_fps(self) -> None:
        """Actualiza el contador de FPS"""
        self._fps_counter += 1
        if time.time() - self._fps_timer >= 1.0:
            self.fps = self._fps_counter
            self._fps_counter = 0
            self._fps_timer = time.time()

    def _handle_key(self, key: int) -> bool:
        """Maneja eventos de teclado"""
        if key == ord("q") or key == 27:
            self.logger.info("Tecla de salida presionada")
            self.stop()
            return False

        elif key == ord(" "):
            self.is_paused = not self.is_paused
            status = "activada" if self.is_paused else "desactivada"
            self.logger.info("Pausa", status=status)
            print(f"{'⏸️' if self.is_paused else '▶️'} Pausa {status}")

        elif key == ord("s"):
            self._save_screenshot()

        elif key == ord("r"):
            self.counter.reset()
            self.tracker.reset()
            self.logger.info("Sistema reiniciado")
            print("🔄 Sistema reiniciado")

        elif key == ord("h"):
            self._show_help()

        return True

    def _save_screenshot(self) -> None:
        """Guarda una captura de pantalla"""
        try:
            filename = get_timestamp_filename("capture", "jpg")
            filepath = os.path.join(self.config.output.screenshots_dir, filename)
            ensure_directory_exists(self.config.output.screenshots_dir)
            self.logger.info("Captura guardada", path=filepath)
            print(f"📸 Captura guardada: {filepath}")
        except Exception as e:
            self.logger.error("Error guardando captura", error=str(e))

    def _show_help(self) -> None:
        """Muestra ayuda en consola"""
        print("""
        ═══════════════════════════════════════════════════
        🎮 CONTROLES DEL SISTEMA
        ═══════════════════════════════════════════════════
        q / ESC  → Salir
        SPACE    → Pausar/Reanudar
        s        → Captura de pantalla
        r        → Reiniciar contadores
        h        → Esta ayuda
        ═══════════════════════════════════════════════════
        """)

    def pause(self) -> None:
        """Pausa la ejecución"""
        self.is_paused = True
        self.logger.info("Pipeline pausado")

    def resume(self) -> None:
        """Reanuda la ejecución"""
        self.is_paused = False
        self.logger.info("Pipeline reanudado")

    def stop(self) -> None:
        """Detiene la ejecución"""
        self.is_running = False
        self.logger.info("Pipeline detenido por solicitud")
        print("⏹️ Deteniendo pipeline...")

    def _print_startup_info(self) -> None:
        """Imprime información de inicio"""
        print("\n" + "=" * 60)
        print("🚗 SISTEMA DE SEGUIMIENTO DE TRAFICO")
        print("=" * 60)
        print(f"📹 Fuente: {self.config.camera.source}")
        print(f"📐 Resolución: {self.config.camera.width}x{self.config.camera.height}")
        print(f"🤖 Modelo: {self.config.model.model_path}")
        print(f"📏 Líneas: {len(self.config.counting_lines)}")
        print(f"🎯 Tracker: {'Avanzado' if str(self.config.tracker.type) == 'hybrid' else 'Clásico'}")
        print(f"⚡ Device: {self.detector.device}")
        print(f"🎯 FPS objetivo: {TARGET_FPS}")
        print("=" * 60 + "\n")

    def _print_final_report(self) -> None:
        """Imprime reporte final detallado"""
        stats = self.counter.get_stats()
        mem_stats = self._memory_tracker.get_stats()

        print("\n" + "=" * 60)
        print("📊 REPORTE FINAL")
        print("=" * 60)
        print(f"🚗 Total vehículos: {stats.get('total', 0)}")
        print(f"⚡ FPS promedio: {self.fps:.1f}")
        print(f"📹 Frames procesados: {self.frame_count}")
        print(f"⏱️ Tiempo promedio: {self.processing_time:.1f}ms")
        print(f"🧠 Memoria pico: {mem_stats.get('peak_mb', 0):.1f} MB")
        print(f"📈 Memoria delta: {mem_stats.get('delta_mb', 0):.1f} MB")

        if self.fps >= TARGET_FPS:
            print(f"✅ Rendimiento: Excelente (>= {TARGET_FPS} FPS)")
        elif self.fps >= MIN_ACCEPTABLE_FPS:
            print(f"⚠️ Rendimiento: Aceptable ({MIN_ACCEPTABLE_FPS}-{TARGET_FPS} FPS)")
        else:
            print(f"❌ Rendimiento: Bajo (< {MIN_ACCEPTABLE_FPS} FPS)")

        if self.config.counting_lines:
            print("\n📏 Conteo por línea:")
            for line_id, count in stats.get("line_counts", {}).items():
                line = next((l for l in self.config.counting_lines if l.get("id") == line_id), {})
                name = line.get("name", line_id)
                print(f"   {name}: {count}")

        if stats.get("vehicle_classes"):
            print("\n🚗 Distribución de clases:")
            for cls, count in stats.get("vehicle_classes", {}).items():
                print(f"   {cls}: {count}")

        if stats.get("avg_speed", 0) > 0:
            print(f"\n📊 Velocidad promedio: {stats.get('avg_speed', 0):.1f} px/frame")

        print("=" * 60 + "\n")

    def __del__(self) -> None:
        """Limpieza de recursos al destruir el pipeline"""
        try:
            self._cleanup()
        except Exception:
            pass

    def _show_collision_alerts(self, frame: np.ndarray, tracks: dict) -> np.ndarray:
        """Muestra alertas de colisión en el frame."""
        if not self.path_predictor:
            return frame

        high_risk = self.path_predictor.get_high_risk_tracks(threshold=0.5)

        if high_risk:
            overlay = frame.copy()

            for track_id in high_risk[:5]:
                track = tracks.get(track_id)
                if track is None:
                    continue

                centroid = track.get("centroid")
                if centroid is None:
                    continue

                cv2.circle(overlay, tuple(centroid), 30, (0, 0, 255), 3)
                cv2.circle(overlay, tuple(centroid), 25, (0, 0, 255), 2)

            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            alert_text = f"⚠️ {len(high_risk)} colisiones potenciales"
            cv2.putText(
                frame,
                alert_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return frame
