"""
Renderizador principal de frames.

Coordina el dibujo de todos los elementos visuales en el frame,
delegando en sub-renderizadores especializados.
"""

import time
from typing import Dict, Any, Optional, Tuple, List, Final, TypedDict

import cv2
import numpy as np

from core.pipeline.renderer_config import RendererConfig
from core.pipeline.render_pipeline import RenderPipeline, RenderLayer
from core.pipeline.text_utils import TextMetricsCache
from core.pipeline.system_info import SystemInfoCollector
from core.pipeline.system_info_renderer import SystemInfoRenderer
from core.pipeline.dashboard import DashboardRenderer
from core.pipeline.overlay import OverlayRenderer
from utils.logger import LoggerMixin


class RenderStats(TypedDict):
    """Estadísticas de renderizado."""
    frames_rendered: int
    avg_render_time_ms: float
    render_times: List[float]
    errors: int
    last_render_time_ms: float
    min_render_time_ms: float
    max_render_time_ms: float


class FrameRenderer(LoggerMixin):
    """
    Renderizador principal que coordina el dibujo de todos los elementos.

    Características:
    - Sistema de capas para renderizado modular
    - Caché de métricas de texto para mejor rendimiento
    - Estadísticas detalladas de rendimiento
    - Recuperación elegante de errores
    - __slots__ para optimización de memoria
    """

    __slots__ = (
        "_config",
        "_overlay_renderer",
        "_dashboard_renderer",
        "_system_info_renderer",
        "_text_cache",
        "_render_pipeline",
        "_stats",
        "_pipeline",
        "_last_valid_frame",
        "_max_render_times",
        "_last_error_time",
        "_error_cooldown",
        "_initialized",
    )

    DEFAULT_FRAME_SHAPE: Final[Tuple[int, int]] = (480, 640)
    MAX_RENDER_TIMES: Final[int] = 100
    ERROR_COOLDOWN: Final[float] = 1.0

    def __init__(self, config=None):
        """
        Inicializa el renderizador.

        Args:
            config: Configuración del sistema
        """
        self._config = RendererConfig.from_global_config(config)
        self._overlay_renderer = OverlayRenderer(config)
        self._dashboard_renderer = DashboardRenderer(config)
        self._system_info_collector = SystemInfoCollector(cache_seconds=0.5)
        self._text_cache = TextMetricsCache(
            font=self._config.font,
            scale=self._config.font_scale,
            thickness=self._config.font_thickness,
        )

        self._system_info_renderer = SystemInfoRenderer(
            config=self._config,
            system_info_collector=self._system_info_collector,
            text_cache=self._text_cache,
        )

        self._stats: RenderStats = {
            "frames_rendered": 0,
            "avg_render_time_ms": 0.0,
            "render_times": [],
            "errors": 0,
            "last_render_time_ms": 0.0,
            "min_render_time_ms": float("inf"),
            "max_render_time_ms": 0.0,
        }

        self._pipeline = None
        self._last_valid_frame: Optional[np.ndarray] = None
        self._max_render_times = self.MAX_RENDER_TIMES
        self._last_error_time = 0.0
        self._error_cooldown = self.ERROR_COOLDOWN
        self._initialized = False

        self._setup_render_pipeline()
        self._initialized = True

        self.logger.info(
            "FrameRenderer inicializado",
            show_system_info=self._config.show_system_info,
            text_cache_size=self._text_cache._max_size,
        )

    def _setup_render_pipeline(self) -> None:
        """Configura el pipeline de renderizado por capas."""
        self._render_pipeline = RenderPipeline()

        self._render_pipeline.add_layer(RenderLayer.OVERLAY, self._render_overlays)

        if self._config.show_system_info:
            self._render_pipeline.add_layer(
                RenderLayer.SYSTEM_INFO,
                self._system_info_renderer.render,
            )

    def render(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Dict[str, Any]],
        stats: Dict[str, Any],
        fps: float = 0.0,
        processing_time_ms: float = 0.0,
        frame_number: int = 0,
    ) -> np.ndarray:
        """
        Renderiza todos los elementos en el frame.

        Args:
            frame: Frame base a renderizar
            tracks: Diccionario de tracks activos
            stats: Estadísticas del sistema
            fps: FPS actual
            processing_time_ms: Tiempo de procesamiento en ms
            frame_number: Número de frame

        Returns:
            np.ndarray: Frame renderizado (siempre retorna un array válido)
        """
        start_time = time.perf_counter()

        valid_frame = self._prepare_frame(frame)

        try:
            status = self._get_pipeline_status()
            self._system_info_renderer.set_pipeline_status(status)

            result = self._render_pipeline.render(
                valid_frame,
                tracks=tracks,
                stats=stats,
                fps=fps,
                processing_time_ms=processing_time_ms,
                frame_number=frame_number,
            )

            render_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(render_time)

            self._last_valid_frame = result
            return result

        except Exception as e:
            self._stats["errors"] += 1
            self.logger.warning(f"Error en renderizado: {e}")
            return self._create_error_frame(valid_frame, str(e)[:50])

    def _render_overlays(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Renderiza los overlays (tracks, líneas, predicciones).

        Args:
            frame: Frame base
            **kwargs: Argumentos adicionales (tracks, stats)

        Returns:
            np.ndarray: Frame con overlays renderizados
        """
        tracks = kwargs.get("tracks", {})
        stats = kwargs.get("stats", {})

        self._overlay_renderer.show_velocity_vectors = self._config.show_velocity_vectors
        self._overlay_renderer.show_trails = self._config.show_trails
        self._overlay_renderer.show_track_arrows = self._config.show_track_arrows
        self._overlay_renderer.show_track_speed = self._config.show_track_speed
        self._overlay_renderer.show_track_confidence = self._config.show_track_confidence
        self._overlay_renderer.track_circle_style = self._config.track_circle_style

        result = self._overlay_renderer.render(frame, tracks, stats)

        if result is None or not isinstance(result, np.ndarray) or result.size == 0:
            self.logger.debug("Overlay renderer falló, usando frame original")
            return frame

        return result

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepara el frame para renderizado, asegurando que sea válido.

        Args:
            frame: Frame a preparar

        Returns:
            np.ndarray: Frame válido para renderizado
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            if self._last_valid_frame is not None:
                return self._last_valid_frame.copy()
            h, w = self._config.default_height, self._config.default_width
            return np.zeros((h, w, 3), dtype=np.uint8)

        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if len(frame.shape) != 3:
            h, w = self._config.default_height, self._config.default_width
            return np.zeros((h, w, 3), dtype=np.uint8)

        return frame.copy()

    def _create_error_frame(self, base_frame: np.ndarray, error_message: str) -> np.ndarray:
        """
        Crea un frame de error con mensaje.

        Args:
            base_frame: Frame base (puede ser None)
            error_message: Mensaje de error a mostrar

        Returns:
            np.ndarray: Frame de error
        """
        current_time = time.time()
        if current_time - self._last_error_time < self._error_cooldown:
            if self._last_valid_frame is not None:
                return self._last_valid_frame.copy()

        self._last_error_time = current_time

        try:
            if self._last_valid_frame is not None:
                result = self._last_valid_frame.copy()
                h, w = result.shape[:2]
            else:
                h, w = self._config.default_height, self._config.default_width
                result = np.zeros((h, w, 3), dtype=np.uint8)

            error_text = f"⚠️ ERROR: {error_message}"
            metrics = self._text_cache.get(error_text)

            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (10, 10),
                (10 + metrics["width"] + 20, 50),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)

            cv2.putText(
                result,
                error_text,
                (20, 40),
                self._config.font,
                0.6,
                self._config.error_color,
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(result, (0, 0), (w - 1, h - 1), self._config.error_color, 3)

            return result

        except Exception:
            result = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                result,
                "ERROR EN RENDERIZADO",
                (20, 240),
                self._config.font,
                1.0,
                self._config.error_color,
                2,
                cv2.LINE_AA,
            )
            return result

    def _get_pipeline_status(self):
        """
        Obtiene el estado actual del pipeline.
        """
        if self._pipeline is not None:
            if hasattr(self._pipeline, "is_running") and not self._pipeline.is_running:
                return "STOPPED"

            if hasattr(self._pipeline, "is_paused") and self._pipeline.is_paused:
                return "PAUSED"

            if hasattr(self._pipeline, "state"):
                state_str = str(self._pipeline.state)
                if "PAUSED" in state_str:
                    return "PAUSED"
                elif "STOPPED" in state_str or "STOPPING" in state_str:
                    return "STOPPED"
                elif "ERROR" in state_str:
                    return "ERROR"

        return "RUNNING"

    def _update_stats(self, render_time_ms: float) -> None:
        """
        Actualiza estadísticas de renderizado.

        Args:
            render_time_ms: Tiempo de renderizado en milisegundos
        """
        self._stats["frames_rendered"] += 1
        self._stats["render_times"].append(render_time_ms)
        self._stats["last_render_time_ms"] = render_time_ms

        if render_time_ms < self._stats["min_render_time_ms"]:
            self._stats["min_render_time_ms"] = render_time_ms
        if render_time_ms > self._stats["max_render_time_ms"]:
            self._stats["max_render_time_ms"] = render_time_ms

        if len(self._stats["render_times"]) > self._max_render_times:
            self._stats["render_times"] = self._stats["render_times"][-self._max_render_times:]

        if self._stats["render_times"]:
            self._stats["avg_render_time_ms"] = (
                sum(self._stats["render_times"]) / len(self._stats["render_times"])
            )

    def set_pipeline_reference(self, pipeline) -> None:
        """Establece referencia al pipeline para obtener estado."""
        self._pipeline = pipeline
        self.logger.debug("Referencia al pipeline establecida")

    def get_stats(self) -> RenderStats:
        """Obtiene estadísticas del renderizador."""
        return self._stats

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Obtiene el último frame renderizado válido."""
        return self._last_valid_frame

    def clear_cache(self) -> None:
        """Limpia el caché de métricas de texto."""
        self._text_cache.clear()
        self.logger.debug("Caché de texto limpiado")

    def reset_stats(self) -> None:
        """Reinicia las estadísticas del renderizador."""
        self._stats = {
            "frames_rendered": 0,
            "avg_render_time_ms": 0.0,
            "render_times": [],
            "errors": 0,
            "last_render_time_ms": 0.0,
            "min_render_time_ms": float("inf"),
            "max_render_time_ms": 0.0,
        }
        self.logger.debug("Estadísticas reiniciadas")
