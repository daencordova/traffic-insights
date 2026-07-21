"""
Recolector de estadísticas de conteo.

Maneja la recolección y cálculo de estadísticas de vehículos.
"""

import time
from typing import Dict, Any, List
from collections import defaultdict


class VehicleEvent:
    """Evento de conteo de un vehículo."""
    __slots__ = ('timestamp', 'object_id', 'line_id', 'line_name',
                     'label', 'class_id', 'centroid', 'velocity',
                     'confidence', 'metadata')

    def __init__(
        self,
        timestamp: str,
        object_id: int,
        line_id: str,
        line_name: str,
        label: str,
        class_id: int,
        centroid: tuple,
        velocity: tuple,
        confidence: float = 0.0,
        metadata: Dict[str, Any] = None
    ):
        self.timestamp = timestamp
        self.object_id = object_id
        self.line_id = line_id
        self.line_name = line_name
        self.label = label
        self.class_id = class_id
        self.centroid = centroid
        self.velocity = velocity
        self.confidence = confidence
        self.metadata = metadata or {}


class StatisticsCollector:
    """
    Recolector de estadísticas de conteo.

    Responsabilidades:
    - Mantener conteos por línea
    - Recolectar estadísticas de clases
    - Calcular velocidades promedio
    - Mantener historial de eventos

    Attributes:
        line_counts: Conteos por línea
        class_counts: Conteos por clase
        speed_history: Historial de velocidades por objeto
        events: Historial de eventos
    """

    def __init__(self, max_history: int = 1000):
        """
        Inicializa el recolector de estadísticas.

        Args:
            max_history: Número máximo de eventos en el historial
        """
        self.line_counts: Dict[str, int] = defaultdict(int)
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.speed_history: Dict[int, List[float]] = defaultdict(list)
        self.events: List[VehicleEvent] = []
        self.max_history = max_history

        self._start_time = time.time()
        self._counts_per_minute: List[int] = []
        self._last_count_time = time.time()
        self._last_stats_time = time.time()
        self._stats_window = 60.0

    def record_crossing(
        self,
        object_id: int,
        line_id: str,
        line_name: str,
        track_data: Dict[str, Any],
        centroid: tuple
    ) -> None:
        """
        Registra un cruce de línea.

        Args:
            object_id: ID del objeto
            line_id: ID de la línea
            line_name: Nombre de la línea
            track_data: Datos del track
            centroid: Posición del objeto
        """
        self.line_counts[line_id] += 1

        label = track_data.get("label", "vehicle")
        self.class_counts[label] += 1

        event = VehicleEvent(
            timestamp=time.strftime("%H:%M:%S"),
            object_id=object_id,
            line_id=line_id,
            line_name=line_name,
            label=label,
            class_id=track_data.get("class_id", -1),
            centroid=centroid,
            velocity=track_data.get("velocity", (0, 0)),
            confidence=track_data.get("confidence", 0.0),
            metadata=track_data.get("metadata", {}),
        )
        self.events.append(event)

        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history:]

    def record_speed(
        self,
        object_id: int,
        velocity: tuple
    ) -> None:
        """
        Registra la velocidad de un objeto.

        Args:
            object_id: ID del objeto
            velocity: Velocidad (vx, vy)
        """
        if not isinstance(velocity, (tuple, list)) or len(velocity) != 2:
            return

        speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
        self.speed_history[object_id].append(speed)

        if len(self.speed_history[object_id]) > 30:
            self.speed_history[object_id] = self.speed_history[object_id][-30:]

    def update_minute_counts(self, total: int) -> None:
        """Actualiza los conteos por minuto."""
        current_time = time.time()
        if current_time - self._last_count_time >= 60:
            self._counts_per_minute.append(total)
            self._last_count_time = current_time

            if len(self._counts_per_minute) > 60:
                self._counts_per_minute = self._counts_per_minute[-60:]

    def get_average_speed(self) -> float:
        """Calcula la velocidad promedio de todos los objetos."""
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        try:
            return float(sum(all_speeds) / len(all_speeds))
        except Exception:
            return 0.0

    def get_max_speed(self) -> float:
        """Obtiene la velocidad máxima registrada."""
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        return float(max(all_speeds))

    def get_min_speed(self) -> float:
        """Obtiene la velocidad mínima registrada."""
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        return float(min(all_speeds))

    def get_speed_percentile(self, percentile: float = 50.0) -> float:
        """
        Obtiene un percentil de las velocidades.

        Args:
            percentile: Percentil a calcular (0-100)

        Returns:
            float: Velocidad en el percentil especificado
        """
        all_speeds = []
        for speeds in self.speed_history.values():
            all_speeds.extend(speeds)

        if not all_speeds:
            return 0.0

        sorted_speeds = sorted(all_speeds)
        index = int(len(sorted_speeds) * percentile / 100.0)
        return float(sorted_speeds[min(index, len(sorted_speeds) - 1)])

    def get_average_per_minute(self) -> float:
        """Obtiene el promedio de conteos por minuto."""
        if not self._counts_per_minute:
            return 0.0
        return float(sum(self._counts_per_minute) / len(self._counts_per_minute))

    def get_total_count(self) -> int:
        """Obtiene el conteo total."""
        return sum(self.line_counts.values())

    def get_count_rate(self) -> float:
        """Obtiene la tasa de conteo por segundo."""
        runtime = time.time() - self._start_time
        if runtime <= 0:
            return 0.0
        return self.get_total_count() / runtime

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene todas las estadísticas."""
        return {
            "total": self.get_total_count(),
            "line_counts": dict(self.line_counts),
            "class_counts": dict(self.class_counts),
            "avg_speed": self.get_average_speed(),
            "max_speed": self.get_max_speed(),
            "min_speed": self.get_min_speed(),
            "avg_per_minute": self.get_average_per_minute(),
            "count_rate": self.get_count_rate(),
            "total_events": len(self.events),
            "runtime_seconds": time.time() - self._start_time,
            "active_objects": len(self.speed_history),
            "timestamp": time.time(),
            "line_count": len(self.line_counts),
            "class_count": len(self.class_counts),
        }

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Obtiene los eventos recientes.

        Args:
            limit: Número máximo de eventos

        Returns:
            List[Dict[str, Any]]: Eventos recientes
        """
        recent = self.events[-limit:] if self.events else []
        return [
            {
                "timestamp": e.timestamp,
                "object_id": e.object_id,
                "line": e.line_name,
                "label": e.label,
                "centroid": e.centroid,
                "confidence": e.confidence,
            }
            for e in recent
        ]

    def get_line_count(self, line_id: str) -> int:
        """
        Obtiene el conteo de una línea específica.

        Args:
            line_id: ID de la línea

        Returns:
            int: Conteo de la línea
        """
        return self.line_counts.get(line_id, 0)

    def get_class_count(self, class_name: str) -> int:
        """
        Obtiene el conteo de una clase específica.

        Args:
            class_name: Nombre de la clase

        Returns:
            int: Conteo de la clase
        """
        return self.class_counts.get(class_name, 0)

    def reset(self) -> None:
        """Reinicia todas las estadísticas."""
        self.line_counts.clear()
        self.class_counts.clear()
        self.speed_history.clear()
        self.events.clear()
        self._counts_per_minute.clear()
        self._start_time = time.time()
        self._last_count_time = time.time()

    def merge(self, other: 'StatisticsCollector') -> None:
        """
        Fusiona las estadísticas de otro recolector.

        Args:
            other: Otro recolector de estadísticas
        """
        for line_id, count in other.line_counts.items():
            self.line_counts[line_id] += count

        for class_name, count in other.class_counts.items():
            self.class_counts[class_name] += count

        self.events.extend(other.events)
        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history:]

        for obj_id, speeds in other.speed_history.items():
            if obj_id in self.speed_history:
                self.speed_history[obj_id].extend(speeds)
                if len(self.speed_history[obj_id]) > 30:
                    self.speed_history[obj_id] = self.speed_history[obj_id][-30:]
            else:
                self.speed_history[obj_id] = speeds.copy()
