"""
Detector de cruces de líneas.

Maneja la detección de cuándo un objeto cruza una línea de conteo.
"""

from typing import Dict, Tuple, Set, Optional
from collections import defaultdict
import time

from utils.geometry import check_crossing
from core.counter.line_manager import CountingLine


class CrossingDetector:
    """
    Detector de cruces de líneas.

    Responsabilidades:
    - Detectar cruces de líneas
    - Mantener historial de cruces por objeto
    - Prevenir conteos duplicados

    Attributes:
        _crossed_lines: Historial de líneas cruzadas por objeto
        _previous_positions: Posiciones anteriores por objeto
        _cross_timestamps: Timestamps de cruces por objeto
    """

    def __init__(self):
        self._crossed_lines: Dict[int, Set[str]] = defaultdict(set)
        self._previous_positions: Dict[int, Tuple[int, int]] = {}
        self._cross_timestamps: Dict[int, float] = {}

        self._stats = {
            "total_crossings": 0,
            "unique_objects": 0,
            "active_objects": 0,
        }

    def detect_crossing(
        self,
        object_id: int,
        current_position: Tuple[int, int],
        line: CountingLine,
        height: int
    ) -> bool:
        """
        Detecta si un objeto ha cruzado una línea.

        Args:
            object_id: ID del objeto
            current_position: Posición actual (x, y)
            line: Línea de conteo
            height: Alto del frame

        Returns:
            bool: True si el objeto cruzó la línea
        """
        if line.id in self._crossed_lines[object_id]:
            return False

        prev_pos = self._previous_positions.get(object_id)
        if prev_pos is None:
            self._previous_positions[object_id] = current_position
            return False

        prev_y, current_y = prev_pos[1], current_position[1]
        line_y = line.y_position or height // 2

        crossed = check_crossing(prev_y, current_y, line_y, line.direction)

        if crossed:
            self._crossed_lines[object_id].add(line.id)
            self._cross_timestamps[object_id] = time.time()
            self._stats["total_crossings"] += 1

            if len(self._crossed_lines[object_id]) == 1:
                self._stats["unique_objects"] += 1

        self._previous_positions[object_id] = current_position

        return crossed

    def has_crossed_line(self, object_id: int, line_id: str) -> bool:
        """
        Verifica si un objeto ya cruzó una línea específica.

        Args:
            object_id: ID del objeto
            line_id: ID de la línea

        Returns:
            bool: True si ya cruzó la línea
        """
        return line_id in self._crossed_lines.get(object_id, set())

    def get_crossed_lines(self, object_id: int) -> Set[str]:
        """
        Obtiene las líneas que un objeto ha cruzado.

        Args:
            object_id: ID del objeto

        Returns:
            Set[str]: Conjunto de IDs de líneas cruzadas
        """
        return self._crossed_lines.get(object_id, set())

    def get_cross_timestamp(self, object_id: int) -> Optional[float]:
        """
        Obtiene el timestamp del último cruce de un objeto.

        Args:
            object_id: ID del objeto

        Returns:
            Optional[float]: Timestamp del último cruce o None
        """
        return self._cross_timestamps.get(object_id)

    def reset_object(self, object_id: int) -> None:
        """Reinicia el historial de un objeto."""
        self._crossed_lines.pop(object_id, None)
        self._previous_positions.pop(object_id, None)
        self._cross_timestamps.pop(object_id, None)

    def reset_line(self, line_id: str) -> None:
        """
        Reinicia el historial de una línea específica.

        Args:
            line_id: ID de la línea a reiniciar
        """
        for object_id in list(self._crossed_lines.keys()):
            if line_id in self._crossed_lines[object_id]:
                self._crossed_lines[object_id].remove(line_id)
                self._stats["total_crossings"] = max(
                    0, self._stats["total_crossings"] - 1
                )

    def clear(self) -> None:
        """Limpia todo el historial."""
        self._crossed_lines.clear()
        self._previous_positions.clear()
        self._cross_timestamps.clear()
        self._stats = {
            "total_crossings": 0,
            "unique_objects": 0,
            "active_objects": 0,
        }

    def get_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas del detector."""
        self._stats["active_objects"] = len(self._crossed_lines)
        return self._stats.copy()

    def get_active_objects(self) -> int:
        """Obtiene el número de objetos activos."""
        return len(self._crossed_lines)

    def __len__(self) -> int:
        """Retorna el número de objetos con cruces registrados."""
        return len(self._crossed_lines)
