"""
Renderizador de overlays para tracks, líneas y predicciones.

Dibuja todos los elementos visuales sobre el frame incluyendo:
- Tracks activos con sus trayectorias
- Líneas de conteo con contadores
- Predicciones de trayectoria (Path Prediction)
- Alertas de colisión
- Información de fusión de sensores
- Información de aprendizaje en línea
"""

import time
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np

from utils.logger import LoggerMixin
from core.validators import ensure_valid_frame
from utils.color_manager import get_color_manager
from core.constants import (
    FONT_SCALE,
    LINE_THICKNESS,
    TRACK_ARROW_LENGTH_MIN,
    TRACK_ARROW_LENGTH_MAX,
    TRACK_CIRCLE_RADIUS,
    TRACK_CONFIDENCE_RADIUS_MIN,
    TRACK_CONFIDENCE_RADIUS_MAX,
    TRACK_TRAIL_THICKNESS_MIN,
    TRACK_TRAIL_THICKNESS_MAX,
    TRACK_BBOX_THICKNESS_MIN,
    TRACK_BBOX_THICKNESS_MAX,
    PREDICTION_POINT_RADIUS_MIN,
    PREDICTION_POINT_RADIUS_MAX,
)


class OverlayRenderer(LoggerMixin):
    """
    Renderizador de overlays sobre el frame.

    Responsabilidades:
    - Dibujar tracks con sus trayectorias
    - Dibujar líneas de conteo con contadores
    - Dibujar predicciones de trayectoria
    - Dibujar alertas de colisión
    - Dibujar información de fusión de sensores
    - Dibujar información de aprendizaje en línea

    Attributes:
        config: Configuración del sistema
        trail_length: Longitud de la trayectoria a mostrar
        show_trails: Si mostrar trayectorias
        show_velocity_vectors: Si mostrar vectores de velocidad
        show_track_arrows: Si mostrar flechas de dirección
        show_track_speed: Si mostrar velocidad
        show_track_confidence: Si mostrar confianza
        track_circle_style: Estilo de círculo para tracks
    """

    __slots__ = (
        '_config',
        'trail_length',
        'show_trails',
        'show_velocity_vectors',
        'show_track_arrows',
        'show_track_speed',
        'show_track_confidence',
        'track_circle_style',
        '_default_frame_size',
        '_color_manager',
    )

    STATUS_COLORS = {
        "confirmed": ((0, 255, 0), "✅", "OK"),
        "lost": ((0, 255, 255), "⚠️", "Lost"),
        "tentative": ((255, 255, 0), "⏳", "New"),
        "dead": ((128, 128, 128), "💀", "Dead"),
    }

    PREDICTION_STATE_COLORS = {
        "stopped": (0, 0, 255),
        "accelerating": (0, 255, 255),
        "decelerating": (0, 165, 255),
        "turning": (255, 0, 255),
        "erratic": (255, 0, 0),
        "moving": (255, 255, 0),
        "unknown": (255, 255, 0),
    }

    def __init__(self, config=None):
        """
        Inicializa el renderizador de overlays.

        Args:
            config: Configuración del sistema (opcional)
        """
        from config.manager import config_manager
        self._config = config_manager.config

        self.trail_length = self._config.visualization.trail_length
        self.show_trails = self._config.visualization.show_trails
        self.show_velocity_vectors = self._config.visualization.show_velocity_vectors

        self.show_track_arrows = getattr(
            self._config.visualization, 'show_track_arrows', True
        )
        self.show_track_speed = getattr(
            self._config.visualization, 'show_track_speed', True
        )
        self.show_track_confidence = getattr(
            self._config.visualization, 'show_track_confidence', True
        )
        self.track_circle_style = getattr(
            self._config.visualization, 'track_circle_style', 'solid'
        )
        self._show_track_ids = getattr(
            self._config.visualization, 'show_track_ids', True
        )

        self._default_frame_size = (
            self._config.camera.width,
            self._config.camera.height
        )

        self._color_manager = get_color_manager()

        self.logger.info(
            "OverlayRenderer mejorado inicializado",
            trail_length=self.trail_length,
            show_trails=self.show_trails,
            show_velocity_vectors=self.show_velocity_vectors,
            show_track_arrows=self.show_track_arrows,
            show_track_speed=self.show_track_speed,
            track_circle_style=self.track_circle_style
        )

    def render(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        Renderiza todos los overlays en el frame.

        Args:
            frame: Frame base
            tracks: Diccionario de tracks activos
            stats: Estadísticas del sistema

        Returns:
            np.ndarray: Frame con overlays (siempre retorna un array válido)
        """
        frame = ensure_valid_frame(
            frame,
            default_shape=(self._default_frame_size[1], self._default_frame_size[0], 3)
        )

        result = frame.copy()

        if self._config.counting_lines:
            try:
                result = self._draw_counting_lines(result, stats)
            except Exception as e:
                self.logger.error(f"Error dibujando líneas de conteo: {e}", exc_info=True)

        if tracks:
            try:
                result = self._draw_tracks(result, tracks)
            except Exception as e:
                self.logger.error(f"Error dibujando tracks: {e}", exc_info=True)

        if self._config.visualization.show_detections:
            try:
                result = self._draw_detections(result, tracks)
            except Exception as e:
                self.logger.error(f"Error dibujando detecciones: {e}", exc_info=True)

        try:
            result = self._draw_collision_alerts(result, tracks)
        except Exception as e:
            self.logger.debug(f"Error dibujando alertas de colisión: {e}")

        return result

    def _sanitize_point(
        self,
        point: Any,
        frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Valida y sanitiza un punto (x, y) para estar dentro de los límites del frame.

        Args:
            point: Punto a validar (tuple o list de 2 elementos)
            frame_shape: Dimensiones del frame (height, width)

        Returns:
            Optional[Tuple[int, int]]: Punto sanitizado o None si es inválido
        """
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            return None

        try:
            x, y = point
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return None
        except (TypeError, ValueError):
            return None

        if x < 0 or y < 0:
            return None

        h, w = frame_shape[:2]
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))

        return (x, y)

    def _sanitize_centroid(
        self,
        centroid: Any,
        h: int,
        w: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Valida y sanitiza un centroide, asegurando que está dentro de los límites del frame.

        Args:
            centroid: Centroide a validar (tuple o list de 2 elementos)
            h: Alto del frame
            w: Ancho del frame

        Returns:
            Tuple[Optional[int], Optional[int]]: (x, y) sanitizados o (None, None)
        """
        if not isinstance(centroid, (tuple, list)) or len(centroid) != 2:
            return None, None

        try:
            cx = int(centroid[0])
            cy = int(centroid[1])
        except (TypeError, ValueError):
            return None, None

        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        return cx, cy

    def _get_enhanced_status_info(self, status: str) -> Tuple[Tuple[int, int, int], str, str]:
        """
        Obtiene información visual según el estado del track.

        Args:
            status: Estado del track

        Returns:
            Tuple: (color, icono, texto_estado)
        """
        status_lower = status.lower() if isinstance(status, str) else ""

        for key, (color, icon, text) in self.STATUS_COLORS.items():
            if key in status_lower:
                return color, icon, text

        return ((0, 165, 255), "❓", "Unknown")

    def _get_prediction_color(self, state: str) -> Tuple[int, int, int]:
        """
        Obtiene el color según el estado de predicción.

        Args:
            state: Estado de la predicción

        Returns:
            Tuple[int, int, int]: Color en formato BGR
        """
        return self.PREDICTION_STATE_COLORS.get(state, (255, 255, 0))

    def _draw_counting_lines(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        Dibuja las líneas de conteo con sus contadores.

        Args:
            frame: Frame donde dibujar
            stats: Estadísticas del sistema

        Returns:
            np.ndarray: Frame con líneas de conteo dibujadas
        """
        if not self._config.counting_lines:
            return frame

        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        h, w = frame.shape[:2]

        for idx, line_config in enumerate(self._config.counting_lines):
            try:
                if not isinstance(line_config, dict):
                    continue

                points = line_config.get("points", [])
                if len(points) < 1:
                    continue

                color = tuple(line_config.get("color", (0, 255, 0)))
                name = line_config.get("name", f"Line {idx + 1}")
                line_id = line_config.get("id", f"line_{idx}")

                count = stats.get("line_counts", {}).get(line_id, 0)

                first_point = points[0]
                if isinstance(first_point, (list, tuple)) and len(first_point) == 2:
                    y_position = min(max(0, first_point[1]), h - 1)

                    cv2.line(
                        frame,
                        (0, y_position),
                        (w, y_position),
                        color,
                        LINE_THICKNESS
                    )

                    label = f"{name}: {count}"
                    x_pos = min(max(10, first_point[0] + 10), w - 100)
                    y_pos = max(10, y_position - 10)
                    cv2.putText(
                        frame,
                        label,
                        (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE,
                        color,
                        2
                    )

            except Exception as e:
                self.logger.debug(
                    "Error dibujando línea de conteo",
                    index=idx,
                    error=str(e)
                )
                continue

        return frame

    def _draw_track_arrow(
        self,
        frame: np.ndarray,
        history: List[Tuple[int, int]],
        cx: int,
        cy: int,
        color: Tuple[int, int, int],
        w: int,
        h: int
    ) -> None:
        """
        Dibuja la flecha de dirección del track.

        Args:
            frame: Frame donde dibujar
            history: Historial de posiciones
            cx: Centroide X
            cy: Centroide Y
            color: Color del track
            w: Ancho del frame
            h: Alto del frame
        """
        if not self.show_track_arrows:
            return

        if not isinstance(history, list) or len(history) < 2:
            return

        prev_point = history[-2]
        curr_point = history[-1]

        if not (isinstance(prev_point, (tuple, list)) and len(prev_point) == 2 and
                isinstance(curr_point, (tuple, list)) and len(curr_point) == 2):
            return

        prev_sanitized = self._sanitize_point(prev_point, (h, w))
        curr_sanitized = self._sanitize_point(curr_point, (h, w))

        if prev_sanitized is None or curr_sanitized is None:
            return

        dx = curr_sanitized[0] - prev_sanitized[0]
        dy = curr_sanitized[1] - prev_sanitized[1]

        if abs(dx) <= 2 or abs(dy) <= 2:
            return

        angle = np.arctan2(dy, dx)
        speed = np.sqrt(dx**2 + dy**2)

        arrow_length = min(
            TRACK_ARROW_LENGTH_MAX,
            max(TRACK_ARROW_LENGTH_MIN, int(speed * 1.5))
        )

        end_x = int(cx + arrow_length * np.cos(angle))
        end_y = int(cy + arrow_length * np.sin(angle))

        end_x = max(0, min(w - 1, end_x))
        end_y = max(0, min(h - 1, end_y))

        cv2.arrowedLine(
            frame,
            (cx, cy),
            (end_x, end_y),
            color,
            2,
            tipLength=0.3,
            line_type=cv2.LINE_AA
        )

    def _draw_track_circle(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        color: Tuple[int, int, int],
        track_data: Dict[str, Any]
    ) -> None:
        """
        Dibuja el círculo del track con su estilo (solid, outline, pulse).

        Args:
            frame: Frame donde dibujar
            cx: Centroide X
            cy: Centroide Y
            color: Color del track
            track_data: Datos del track
        """
        confidence = track_data.get("confidence", 0.5)

        if self.track_circle_style == "pulse":
            pulse = 0.5 + 0.5 * np.sin(time.time() * 2)
            radius = int(TRACK_CIRCLE_RADIUS * (0.8 + 0.2 * pulse))
            cv2.circle(frame, (cx, cy), radius, color, 2)
        elif self.track_circle_style == "outline":
            cv2.circle(frame, (cx, cy), TRACK_CIRCLE_RADIUS, color, 2)
        else:
            cv2.circle(frame, (cx, cy), TRACK_CIRCLE_RADIUS, color, 2)
            if self.show_track_confidence:
                inner_radius = int(
                    TRACK_CONFIDENCE_RADIUS_MIN +
                    (TRACK_CONFIDENCE_RADIUS_MAX - TRACK_CONFIDENCE_RADIUS_MIN) * confidence
                )
                cv2.circle(frame, (cx, cy), max(2, inner_radius), color, -1)

        cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)

    def _draw_track_label(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        track_id: int,
        status_icon: str,
        color: Tuple[int, int, int]
    ) -> None:
        """
        Dibuja la etiqueta del track con su ID y estado.

        Args:
            frame: Frame donde dibujar
            cx: Centroide X
            cy: Centroide Y
            track_id: ID del track
            status_icon: Icono del estado
            color: Color del track
        """

        if not self._show_track_ids:
            return

        h, w = frame.shape[:2]

        label = f"#{track_id} {status_icon}"
        label_x = min(w - 50, cx + 15)
        label_y = max(10, cy - 15)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        cv2.rectangle(
            frame,
            (label_x - 2, label_y - text_h - 2),
            (label_x + text_w + 2, label_y + 2),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    def _draw_track_trail(
        self,
        frame: np.ndarray,
        history: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        trail_length: int
    ) -> None:
        """
        Dibuja la trayectoria (trail) del track.

        Args:
            frame: Frame donde dibujar
            history: Historial de posiciones
            color: Color del track
            trail_length: Longitud máxima de la trayectoria
        """
        if not self.show_trails:
            return

        if not isinstance(history, list) or len(history) <= 1:
            return

        if len(history) > trail_length:
            history = history[-trail_length:]

        h, w = frame.shape[:2]

        for i in range(1, len(history)):
            try:
                prev = self._sanitize_point(history[i - 1], (h, w))
                curr = self._sanitize_point(history[i], (h, w))

                if prev is None or curr is None:
                    continue

                alpha = i / len(history)
                thickness = max(
                    TRACK_TRAIL_THICKNESS_MIN,
                    int(TRACK_TRAIL_THICKNESS_MAX * alpha)
                )
                color_fade = tuple(int(c * alpha) for c in color)

                cv2.line(
                    frame,
                    prev,
                    curr,
                    color_fade,
                    thickness,
                    cv2.LINE_AA
                )
            except Exception:
                continue

    def _draw_track_bbox(
        self,
        frame: np.ndarray,
        track_data: Dict[str, Any],
        color: Tuple[int, int, int]
    ) -> None:
        """
        Dibuja el bounding box del track con su confianza.

        Args:
            frame: Frame donde dibujar
            track_data: Datos del track
            color: Color del track
        """
        bbox = track_data.get("bbox")
        if not bbox or not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            return

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        confidence = track_data.get("confidence", 0.5)

        border_thickness = max(
            TRACK_BBOX_THICKNESS_MIN,
            int(TRACK_BBOX_THICKNESS_MAX * confidence)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            border_thickness
        )

        if self.show_track_confidence and confidence > 0.3:
            conf_text = f"{confidence:.0%}"
            conf_x = x1 + 2
            conf_y = y1 + 15

            (text_w, text_h), _ = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
            )

            cv2.rectangle(
                frame,
                (conf_x - 2, conf_y - text_h - 2),
                (conf_x + text_w + 2, conf_y + 2),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                conf_text,
                (conf_x, conf_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

    def _draw_track_speed(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        track_data: Dict[str, Any],
        h: int,
        w: int
    ) -> None:
        """
        Dibuja la velocidad del track.

        Args:
            frame: Frame donde dibujar
            cx: Centroide X
            cy: Centroide Y
            track_data: Datos del track
            h: Alto del frame
            w: Ancho del frame
        """
        if not self.show_track_speed:
            return

        velocity = track_data.get("velocity", (0, 0))
        if not isinstance(velocity, (tuple, list)) or len(velocity) != 2:
            return

        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)

        if speed <= 1.0:
            return

        speed_text = f"{speed:.1f} px/f"
        speed_x = min(w - 60, cx + 15)
        speed_y = min(h - 10, cy + 25)

        cv2.putText(
            frame,
            speed_text,
            (speed_x, speed_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    def _draw_track_prediction(
        self,
        frame: np.ndarray,
        track_data: Dict[str, Any],
        cx: int,
        cy: int,
        color: Tuple[int, int, int]
    ) -> None:
        """
        Dibuja la predicción de trayectoria del track.

        Args:
            frame: Frame donde dibujar
            track_data: Datos del track
            cx: Centroide X
            cy: Centroide Y
            color: Color del track
        """
        path_prediction = track_data.get("path_prediction")
        if not path_prediction or not isinstance(path_prediction, dict):
            return

        self._draw_enhanced_path_prediction(
            frame,
            path_prediction,
            cx,
            cy,
            color
        )

    def _draw_tracks(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Dibuja todos los tracks activos con información mejorada.

        Args:
            frame: Frame donde dibujar
            tracks: Diccionario de tracks activos

        Returns:
            np.ndarray: Frame con tracks dibujados
        """
        if not tracks or frame is None or not isinstance(frame, np.ndarray):
            return frame

        h, w = frame.shape[:2]

        track_ids = list(tracks.keys())
        colors = self._color_manager.get_colors_for_tracks(track_ids)

        for track_id, track_data in tracks.items():
            try:
                if not isinstance(track_data, dict):
                    continue

                centroid = track_data.get("centroid")
                if not centroid or not isinstance(centroid, (tuple, list)):
                    continue

                cx, cy = self._sanitize_centroid(centroid, h, w)
                if cx is None or cy is None:
                    continue

                color = colors.get(track_id, (0, 255, 0))

                status = track_data.get("status", "")
                status_color, status_icon, _ = self._get_enhanced_status_info(status)
                final_color = self._blend_colors(color, status_color)

                history = track_data.get("history", [])

                self._draw_track_arrow(frame, history, cx, cy, final_color, w, h)
                self._draw_track_circle(frame, cx, cy, final_color, track_data)
                self._draw_track_label(frame, cx, cy, track_id, status_icon, final_color)
                self._draw_track_trail(frame, history, final_color, self.trail_length)
                self._draw_track_bbox(frame, track_data, final_color)
                self._draw_track_speed(frame, cx, cy, track_data, h, w)
                self._draw_track_prediction(frame, track_data, cx, cy, final_color)

            except Exception as e:
                self.logger.error(
                    f"Error dibujando track {track_id}: {e}",
                    exc_info=True
                )
                continue

        return frame

    def _blend_colors(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        weight: float = 0.3
    ) -> Tuple[int, int, int]:
        """
        Mezcla dos colores.

        Args:
            color1: Color base.
            color2: Color de estado.
            weight: Peso del color de estado (0-1).

        Returns:
            Tuple[int, int, int]: Color mezclado en formato BGR.
        """
        b1, g1, r1 = color1
        b2, g2, r2 = color2

        b = int(b1 * (1 - weight) + b2 * weight)
        g = int(g1 * (1 - weight) + g2 * weight)
        r = int(r1 * (1 - weight) + r2 * weight)

        return (b, g, r)

    def _draw_enhanced_path_prediction(
        self,
        frame: np.ndarray,
        prediction: Dict[str, Any],
        cx: int,
        cy: int,
        color: tuple
    ) -> None:
        """
        Dibuja la predicción de trayectoria mejorada con efectos visuales.

        Args:
            frame: Frame donde dibujar
            prediction: Datos de predicción
            cx: Centroide X
            cy: Centroide Y
            color: Color base del track
        """
        positions = prediction.get("positions", [])
        if not positions or len(positions) < 2:
            return

        state = prediction.get("state", "unknown")
        uncertainty = prediction.get("uncertainty", 0.5)
        collision_risk = prediction.get("collision_risk", 0.0)

        h, w = frame.shape[:2]

        sanitized_positions = []
        for pos in positions:
            sp = self._sanitize_point(pos, (h, w))
            if sp is not None:
                sanitized_positions.append(sp)

        if len(sanitized_positions) < 2:
            return

        pred_color = self._get_prediction_color(state)

        for i, pos in enumerate(sanitized_positions):
            alpha = 1.0 - (i / len(sanitized_positions))
            color_pred = tuple(int(c * alpha) for c in pred_color)

            base_radius = max(
                PREDICTION_POINT_RADIUS_MIN,
                int(PREDICTION_POINT_RADIUS_MAX * (1 - uncertainty))
            )
            radius = max(1, int(base_radius * (1 + alpha * 0.5)))

            cv2.circle(frame, pos, radius, color_pred, -1)
            cv2.circle(frame, pos, radius + 1, (255, 255, 255), 1)

            if i > 0:
                prev_pos = sanitized_positions[i - 1]
                cv2.line(
                    frame,
                    prev_pos,
                    pos,
                    color_pred,
                    max(1, int(2 * alpha)),
                    cv2.LINE_AA
                )

        state_text = f"🚦 {state}"
        state_x = min(w - 60, cx + 10)
        state_y = min(h - 10, cy + 40)

        (text_w, text_h), _ = cv2.getTextSize(
            state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )

        cv2.rectangle(
            frame,
            (state_x - 2, state_y - text_h - 2),
            (state_x + text_w + 2, state_y + 2),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            state_text,
            (state_x, state_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            pred_color,
            1
        )

        if collision_risk > 0.3:
            risk_color = (0, 0, 255) if collision_risk > 0.6 else (0, 165, 255)
            risk_text = f"⚠️ Risk: {collision_risk:.0%}"

            last_pos = sanitized_positions[-1] if sanitized_positions else (cx, cy)
            if isinstance(last_pos, (tuple, list)) and len(last_pos) == 2:
                risk_x = min(w - 10, last_pos[0] + 10)
                risk_y = max(10, last_pos[1] - 10)

                (text_w, text_h), _ = cv2.getTextSize(
                    risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )

                cv2.rectangle(
                    frame,
                    (risk_x - 2, risk_y - text_h - 2),
                    (risk_x + text_w + 2, risk_y + 2),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    risk_text,
                    (risk_x, risk_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    risk_color,
                    1
                )

                if collision_risk > 0.6:
                    pulse = 0.5 + 0.5 * np.sin(time.time() * 2)
                    radius = int(15 + 10 * pulse)
                    cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 2)

    def _draw_collision_alerts(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Dibuja alertas de colisión para tracks de alto riesgo.

        Args:
            frame: Frame donde dibujar
            tracks: Diccionario de tracks activos

        Returns:
            np.ndarray: Frame con alertas de colisión dibujadas
        """
        if not tracks or frame is None or not isinstance(frame, np.ndarray):
            return frame

        high_risk = []

        for track_id, track_data in tracks.items():
            prediction = track_data.get("path_prediction")
            if prediction and prediction.get("collision_risk", 0.0) > 0.5:
                high_risk.append(track_id)

        if not high_risk:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        for track_id in high_risk[:5]:
            track_data = tracks.get(track_id)
            if not track_data:
                continue

            centroid = track_data.get("centroid")
            if not centroid or not isinstance(centroid, (tuple, list)):
                continue

            cx, cy = self._sanitize_centroid(centroid, h, w)
            if cx is None or cy is None:
                continue

            pulse = 0.5 + 0.5 * np.sin(time.time() * 2)
            radius = int(25 + 10 * pulse)

            cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), 3)
            cv2.circle(overlay, (cx, cy), radius - 5, (0, 0, 255), 2)

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

    def _draw_detections(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Dibuja las detecciones (si están disponibles en tracks).

        Args:
            frame: Frame donde dibujar
            tracks: Diccionario de tracks

        Returns:
            np.ndarray: Frame con detecciones dibujadas
        """
        return frame
