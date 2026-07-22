"""
Pipeline síncrono del sistema de seguimiento de tráfico.

Este módulo implementa el pipeline síncrono (legacy) para el sistema
de seguimiento de tráfico. Se recomienda usar el pipeline asíncrono para
mejor rendimiento.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from core.detector import YOLODetector
from core.tracker import AdvancedTracker
from core.counter import VehicleCounter
from core.pipeline.renderer import FrameRenderer
from core.pipeline.controls import ControlHandler
from core.context_managers import VideoCaptureContext
from utils.logger import LoggerMixin
from utils.helpers import get_memory_usage, force_garbage_collection
from core.pipeline.system_info import set_system_status
from core.validators import validate_frame, ensure_valid_frame
from core.constants import (
    MEMORY_CHECK_INTERVAL,
    GC_INTERVAL,
    WINDOW_NAME,
    TARGET_FPS,
    MIN_ACCEPTABLE_FPS
)



class VehicleCountingPipeline(LoggerMixin):
    """
    Pipeline síncrono del sistema de seguimiento de tráfico.

    Este pipeline procesa frames de forma secuencial (bloqueante).
    Para mejor rendimiento, usar AsyncVehicleCountingPipeline.

    Attributes:
        config: Configuración del sistema
        detector: Detector de objetos
        tracker: Tracker de objetos
        counter: Contador de vehículos
        renderer: Renderizador de frames
        controls: Manejador de controles
        is_running: Estado de ejecución
        fps: FPS actual
    """

    DEFAULT_FRAME_SHAPE = (480, 640)
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY = 0.1

    def __init__(self) -> None:
        """Inicializa el pipeline síncrono."""
        from config.manager import config_manager
        self.config = config_manager.config
        self.logger.info("Inicializando VehicleCountingPipeline")

        use_optimized = getattr(
            self.config.optimization,
            "use_optimized_detector",
            True
        )
        self._using_optimized_detector = False

        if use_optimized:
            try:
                from core.detector import OptimizedYOLODetector
                self.detector = OptimizedYOLODetector()
                self._using_optimized_detector = True
                self.logger.info("✅ Detector optimizado activado")
            except (ImportError, Exception) as e:
                self.logger.warning(
                    f"Detector optimizado no disponible: {e}. Usando estándar."
                )
                self.detector = YOLODetector()
        else:
            self.detector = YOLODetector()

        self.tracker = AdvancedTracker()
        self.counter = VehicleCounter()
        self.renderer = FrameRenderer(self.config)
        self.controls = ControlHandler(self.config)

        self.renderer.set_pipeline_reference(self)
        self.controls.register_callback("on_reset", self._reset_system)

        self.is_running = False
        self.is_paused = False
        self.fps = 0.0
        self.frame_count = 0
        self.processing_time = 0.0

        self._fps_counter = 0
        self._fps_timer = time.time()
        self._start_time = time.time()
        self._last_memory_check = time.time()
        self._last_gc_time = time.time()
        self._last_valid_frame = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        self._print_startup_info()
        self.logger.info("Pipeline inicializado")


    def run(self, source: Optional[str] = None) -> None:
        """
        Ejecuta el pipeline principal.

        Args:
            source: Fuente de video (opcional, usa config por defecto)
        """
        source = source or self.config.camera.source
        self.logger.info(f"Iniciando pipeline desde: {source}")

        try:
            with VideoCaptureContext(
                source=source,
                width=self.config.camera.width,
                height=self.config.camera.height,
                reconnect_attempts=self.config.camera.reconnect_attempts,
                reconnect_delay=self.config.camera.reconnect_delay,
            ) as capture:

                if not capture.is_opened():
                    raise RuntimeError(f"No se pudo abrir la fuente: {source}")

                if not self._verify_capture(capture):
                    raise RuntimeError("Verificación de captura fallida")

                self.logger.info("✅ Fuente verificada correctamente")
                self.logger.info("\n🔄 Procesando... Presiona 'q' para salir\n")

                self.is_running = True
                self._start_time = time.time()

                while self.is_running:
                    try:
                        self._check_memory()

                        if self.controls.is_paused:
                            self._handle_paused_state(capture)
                            continue

                        ret, frame = capture.read()
                        if not ret:
                            self._handle_read_error(capture)
                            continue

                        if not self._validate_frame(frame):
                            self.logger.debug("Frame inválido, saltando...")
                            self._show_error_frame("Frame inválido")
                            continue

                        try:
                            processed = self.process_frame(frame)
                        except Exception as e:
                            self.logger.error(f"Error procesando frame: {e}", exc_info=True)
                            self._show_error_frame(f"Error: {str(e)[:30]}")
                            self._consecutive_errors += 1

                            if self._consecutive_errors > self._max_consecutive_errors:
                                self.logger.error("Demasiados errores consecutivos, deteniendo...")
                                break

                            time.sleep(0.05)
                            continue

                        self._consecutive_errors = 0

                        if processed is not None and self._validate_frame(processed):
                            self._display_frame(processed)
                        else:
                            self._display_fallback_frame()

                        key = cv2.waitKey(1) & 0xFF
                        if not self.controls.process_key(key):
                            self.is_running = False
                            break

                        self._update_fps()

                    except KeyboardInterrupt:
                        self.logger.info("Interrupción por usuario")
                        print("\n⏹️ Interrupción recibida")
                        break
                    except Exception as e:
                        self.logger.error(f"Error en bucle principal: {e}", exc_info=True)
                        time.sleep(0.1)
                        continue

        except KeyboardInterrupt:
            self.logger.info("Interrupción por usuario")
            print("\n⏹️ Interrupción recibida")
        except Exception as e:
            self.logger.error(f"Error fatal: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()


    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un frame individual.

        Args:
            frame: Frame a procesar

        Returns:
            Optional[np.ndarray]: Frame procesado y renderizado, o None si es inválido
        """
        if frame is None or not self._validate_frame(frame):
            self.logger.debug("Frame inválido, saltando procesamiento")
            return None

        if self.controls.is_paused:
            return frame

        start_time = time.perf_counter()
        detections = self._safe_detect(frame)
        tracks = self._safe_track(detections, frame)
        stats = self._safe_count(tracks, frame)
        result = self._safe_render(frame, tracks, stats)

        self.processing_time = (time.perf_counter() - start_time) * 1000
        self.frame_count += 1

        self._update_system_status()

        return result

    def _safe_detect(self, frame: np.ndarray) -> list:
        """
        Ejecuta detección con manejo de errores.

        Args:
            frame: Frame a procesar

        Returns:
            list: Lista de detecciones o lista vacía en caso de error
        """
        try:
            return self.detector.detect(frame)
        except Exception as e:
            self.logger.error(f"Error en detección: {e}")
            return []

    def _safe_track(self, detections: list, frame: np.ndarray) -> dict:
        """
        Ejecuta tracking con manejo de errores.

        Args:
            detections: Lista de detecciones
            frame: Frame actual

        Returns:
            dict: Diccionario de tracks o vacío en caso de error
        """
        try:
            return self.tracker.update(detections, frame)
        except Exception as e:
            self.logger.error(f"Error en tracking: {e}")
            try:
                self.tracker.reset()
            except Exception:
                pass
            return {}

    def _safe_count(self, tracks: dict, frame: np.ndarray) -> dict:
        """
        Ejecuta conteo con manejo de errores.

        Args:
            tracks: Diccionario de tracks
            frame: Frame actual

        Returns:
            dict: Estadísticas de conteo o estadísticas actuales en caso de error
        """
        try:
            return self.counter.process(tracks, frame)
        except Exception as e:
            self.logger.error(f"Error en conteo: {e}")
            return self.counter.get_stats()

    def _safe_render(self, frame: np.ndarray, tracks: dict, stats: dict) -> Optional[np.ndarray]:
        """
        Ejecuta renderizado con manejo de errores.

        Args:
            frame: Frame a renderizar
            tracks: Diccionario de tracks
            stats: Estadísticas del sistema

        Returns:
            Optional[np.ndarray]: Frame renderizado o None en caso de error
        """
        try:
            return self.renderer.render(
                frame,
                tracks,
                stats,
                self.fps,
                self.processing_time,
                self.frame_count
            )
        except Exception as e:
            self.logger.error(f"Error en renderizado: {e}")
            result = frame.copy()
            try:
                cv2.putText(
                    result,
                    f"Error: {str(e)[:20]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            except Exception:
                pass
            return result


    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Valida que el frame sea válido."""
        return validate_frame(frame, min_width=10, min_height=10)

    def _verify_capture(self, capture: VideoCaptureContext) -> bool:
        """
        Verifica que la captura funcione correctamente.

        Args:
            capture: Contexto de captura

        Returns:
            bool: True si la verificación fue exitosa
        """
        self.logger.debug("Verificando captura...")

        ret, test_frame = capture.read()
        if not ret or test_frame is None:
            self.logger.error("No se pudo leer frame de prueba")
            return False

        if not self._validate_frame(test_frame):
            self.logger.error("Frame de prueba inválido")
            return False

        fps = capture.get_fps()
        if fps > 0:
            self.logger.debug(f"FPS de fuente: {fps:.1f}")

        return True


    def _display_frame(self, frame: np.ndarray) -> None:
        """
        Muestra un frame en la ventana.

        Args:
            frame: Frame a mostrar
        """
        try:
            cv2.imshow(WINDOW_NAME, frame)
            self._last_valid_frame = frame
            self.controls.set_last_frame(frame)
        except Exception as e:
            self.logger.error(f"Error mostrando frame: {e}")

    def _display_fallback_frame(self) -> None:
        """Muestra un frame de fallback si no hay frame válido."""
        if self._last_valid_frame is not None:
            try:
                cv2.imshow(WINDOW_NAME, self._last_valid_frame)
            except Exception:
                pass
        else:
            self._show_error_frame("Error procesando frame")

    def _show_error_frame(self, message: str) -> None:
        """
        Muestra un frame de error con un mensaje.

        Args:
            message: Mensaje de error a mostrar
        """
        try:
            h = self.config.camera.height
            w = self.config.camera.width
            error_frame = ensure_valid_frame(None, default_shape=(h, w, 3))

            for i in range(h):
                intensity = int(20 + 10 * (i / h))
                error_frame[i, :] = [0, 0, intensity]

            cv2.rectangle(error_frame, (0, 0), (w-1, h-1), (0, 0, 255), 3)

            cv2.circle(error_frame, (w//2, h//2 - 40), 30, (0, 0, 255), 3)
            cv2.line(
                error_frame,
                (w//2 - 15, h//2 - 55),
                (w//2 + 15, h//2 - 25),
                (0, 0, 255),
                3
            )
            cv2.line(
                error_frame,
                (w//2 - 15, h//2 - 25),
                (w//2 + 15, h//2 - 55),
                (0, 0, 255),
                3
            )

            cv2.putText(
                error_frame,
                message[:40],
                (w//2 - 150, h//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.putText(
                error_frame,
                "Presiona 'q' para salir",
                (w//2 - 100, h//2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                1
            )

            cv2.imshow(WINDOW_NAME, error_frame)
            cv2.waitKey(100)
        except Exception as e:
            self.logger.debug(f"Error mostrando frame de error: {e}")


    def _handle_paused_state(self, capture: VideoCaptureContext) -> None:
        """
        Maneja el estado de pausa.

        Args:
            capture: Contexto de captura
        """
        if self._last_valid_frame is not None:
            try:
                pause_frame = self._last_valid_frame.copy()
                overlay = pause_frame.copy()
                h, w = pause_frame.shape[:2]

                cv2.rectangle(
                    overlay,
                    (w//4, h//3),
                    (3*w//4, 2*h//3),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.5, pause_frame, 0.5, 0, pause_frame)

                cv2.putText(
                    pause_frame,
                    "PAUSADO",
                    (w//2 - 80, h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )
                cv2.putText(
                    pause_frame,
                    "Presiona ESPACIO para reanudar",
                    (w//2 - 120, h//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(WINDOW_NAME, pause_frame)
            except Exception as e:
                self.logger.debug(f"Error mostrando pausa: {e}")

        key = cv2.waitKey(50) & 0xFF
        self.controls.process_key(key)

    def _handle_read_error(self, capture: VideoCaptureContext) -> None:
        """
        Maneja errores de lectura de frames.

        Args:
            capture: Contexto de captura
        """
        self.logger.warning("Error leyendo frame, intentando recuperar...")

        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            try:
                time.sleep(self.RECONNECT_DELAY)
                ret, frame = capture.read()
                if ret and frame is not None and self._validate_frame(frame):
                    self.logger.info(f"Frame recuperado en intento {attempt + 1}")
                    return
            except Exception:
                pass

        self.logger.warning("Reconectando a la fuente...")
        try:
            if capture.reconnect():
                self.logger.info("Reconexión exitosa")
                for _ in range(3):
                    capture.read()
                return
        except Exception as e:
            self.logger.error(f"Error en reconexión: {e}")

        self.logger.error("No se pudo recuperar la fuente")

    def _update_system_status(self) -> None:
        """Actualiza el estado del sistema para el dashboard."""
        if self.is_running:
            if self.controls.is_paused:
                set_system_status("PAUSED")
            else:
                set_system_status("RUNNING")
        else:
            set_system_status("STOPPED")


    def _update_fps(self) -> None:
        """Actualiza el contador de FPS."""
        self._fps_counter += 1
        if time.time() - self._fps_timer >= 1.0:
            self.fps = self._fps_counter
            self._fps_counter = 0
            self._fps_timer = time.time()

    def _check_memory(self) -> None:
        """Verifica el uso de memoria y realiza limpieza si es necesario."""
        current_time = time.time()

        if current_time - self._last_memory_check >= MEMORY_CHECK_INTERVAL:
            self._last_memory_check = current_time

            try:
                mem = get_memory_usage()
                mem_percent = mem.get("percent", 0)

                if mem_percent > 75:
                    self.logger.warning(
                        f"Memoria alta: {mem_percent:.1f}%, limpiando..."
                    )
                    try:
                        self.detector.clear_cache()
                    except Exception:
                        pass
                    force_garbage_collection()

                    if mem_percent > 85:
                        self.logger.warning("Memoria crítica, limpieza agresiva")
                        import gc
                        gc.collect()
                        force_garbage_collection()
            except Exception as e:
                self.logger.debug(f"Error verificando memoria: {e}")

        if current_time - self._last_gc_time >= GC_INTERVAL:
            self._last_gc_time = current_time
            try:
                import gc
                gc.collect()
            except Exception:
                pass


    def _cleanup(self) -> None:
        """Limpieza de recursos al finalizar."""
        self.is_running = False

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.debug(f"Error cerrando ventanas: {e}")

        try:
            self.detector.clear_cache()
        except Exception as e:
            self.logger.debug(f"Error limpiando caché del detector: {e}")

        try:
            self.tracker.reset()
        except Exception as e:
            self.logger.debug(f"Error reiniciando tracker: {e}")

        try:
            force_garbage_collection()
        except Exception:
            pass

        runtime = time.time() - self._start_time
        self.logger.info(
            "Pipeline detenido",
            runtime_seconds=f"{runtime:.1f}",
            frames=self.frame_count,
            fps=self.fps
        )
        self._print_final_report()

    def _reset_system(self) -> None:
        """Reinicia el sistema."""
        try:
            self.counter.reset()
        except Exception as e:
            self.logger.error(f"Error reiniciando contador: {e}")

        try:
            self.tracker.reset()
        except Exception as e:
            self.logger.error(f"Error reiniciando tracker: {e}")

        self.frame_count = 0
        self.fps = 0.0
        self.processing_time = 0.0
        self._start_time = time.time()
        self._last_valid_frame = None
        self._consecutive_errors = 0

        self.logger.info("Sistema reiniciado")

    def pause(self) -> None:
        """Pausa la ejecución."""
        self.controls.toggle_pause()

    def resume(self) -> None:
        """Reanuda la ejecución."""
        if self.controls.is_paused:
            self.controls.toggle_pause()

    def stop(self) -> None:
        """Detiene la ejecución."""
        self.is_running = False
        self.logger.info("Pipeline detenido por solicitud")

    def _print_startup_info(self) -> None:
        """Imprime información de inicio en consola."""
        self.logger.info("=" * 60)
        self.logger.info("🚗 SISTEMA DE SEGUIMIENTO DE TRÁFICO")
        self.logger.info("=" * 60)
        self.logger.info(f"📹 Fuente: {self.config.camera.source}")
        self.logger.info(f"📐 Resolución: {self.config.camera.width}x{self.config.camera.height}")
        self.logger.info(f"🤖 Modelo: {self.config.model.model_path}")
        self.logger.info(f"📏 Líneas: {len(self.config.counting_lines)}")
        self.logger.info(f"⚡ Device: {self.detector.device if hasattr(self.detector, 'device') else 'cpu'}")
        self.logger.info(f"🎯 FPS objetivo: {TARGET_FPS}")
        self.logger.info(f"⚡ Detector optimizado: {'✅' if self._using_optimized_detector else '❌'}")
        self.logger.info("=" * 60)

    def _print_final_report(self) -> None:
        """Imprime reporte final en consola."""
        try:
            stats = self.counter.get_stats()
        except Exception:
            stats = {}

        self.logger.info("=" * 60)
        self.logger.info("📊 REPORTE FINAL")
        self.logger.info("=" * 60)
        self.logger.info(f"🚗 Total vehículos: {stats.get('total', 0)}")
        self.logger.info(f"⚡ FPS promedio: {self.fps:.1f}")
        self.logger.info(f"📹 Frames procesados: {self.frame_count}")
        self.logger.info(f"⏱️ Tiempo promedio: {self.processing_time:.1f}ms")

        if self.fps >= TARGET_FPS:
            self.logger.info(f"✅ Rendimiento: Excelente (>= {TARGET_FPS} FPS)")
        elif self.fps >= MIN_ACCEPTABLE_FPS:
            self.logger.info(f"⚠️ Rendimiento: Aceptable ({MIN_ACCEPTABLE_FPS}-{TARGET_FPS} FPS)")
        else:
            self.logger.info(f"❌ Rendimiento: Bajo (< {MIN_ACCEPTABLE_FPS} FPS)")

        line_counts = stats.get("line_counts", {})
        if line_counts and self.config.counting_lines:
            self.logger.info("📏 Conteo por línea:")
            for line_id, count in line_counts.items():
                line = next(
                    (l for l in self.config.counting_lines if l.get("id") == line_id),
                    {}
                )
                name = line.get("name", line_id)
                self.logger.info(f"   {name}: {count}")

        vehicle_classes = stats.get("vehicle_classes", {})
        if vehicle_classes:
            self.logger.info("🚗 Distribución de clases:")
            for cls_name, count in vehicle_classes.items():
                self.logger.info(f"   {cls_name}: {count}")

        self.logger.info("=" * 60)


    def __del__(self) -> None:
        """Limpieza de recursos al destruir el pipeline."""
        try:
            self._cleanup()
        except Exception:
            pass
