"""
Gestor de líneas de conteo.

Maneja la configuración, validación y acceso a las líneas de conteo.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CountingLine:
    """
    Representa una línea de conteo.

    Attributes:
        id: Identificador único de la línea
        name: Nombre descriptivo
        points: Puntos que definen la línea
        color: Color en formato BGR
        direction: Dirección de conteo ('up' o 'down')
        y_position: Posición Y de la línea
        enabled: Si la línea está activa
        metadata: Metadatos adicionales
    """
    id: str
    name: str
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int]
    direction: str
    y_position: int
    enabled: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la línea a diccionario."""
        return {
            "id": self.id,
            "name": self.name,
            "points": self.points,
            "color": self.color,
            "direction": self.direction,
            "y_position": self.y_position,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


class LineManager:
    """
    Gestor de líneas de conteo.

    Responsabilidades:
    - Validar configuraciones de líneas
    - Proporcionar acceso a líneas
    - Mantener estado de líneas

    Attributes:
        lines: Lista de líneas de conteo
    """

    def __init__(self, config_lines: List[Dict[str, Any]]):
        """
        Inicializa el gestor de líneas.

        Args:
            config_lines: Lista de configuraciones de líneas
        """
        self.lines: List[CountingLine] = []
        self._initialize_lines(config_lines)

    def _initialize_lines(self, config_lines: List[Dict[str, Any]]) -> None:
        """
        Inicializa las líneas desde la configuración.

        Args:
            config_lines: Lista de configuraciones de líneas
        """
        for idx, line_config in enumerate(config_lines):
            if not self._validate_line_config(line_config):
                continue

            points = line_config.get("points", [])
            first_point = points[0] if points else (0, 0)

            line = CountingLine(
                id=line_config.get("id", f"line_{idx}"),
                name=line_config.get("name", f"Línea {idx + 1}"),
                points=points,
                color=tuple(line_config.get("color", (0, 255, 0))),
                direction=line_config.get("direction", "down"),
                y_position=first_point[1] if points else 0,
                enabled=line_config.get("enabled", True),
                metadata=line_config.get("metadata", {}),
            )
            self.lines.append(line)

    def _validate_line_config(self, config: Dict[str, Any]) -> bool:
        """
        Valida una configuración de línea.

        Args:
            config: Configuración a validar

        Returns:
            bool: True si la configuración es válida
        """
        if not isinstance(config, dict):
            return False

        points = config.get("points", [])
        if not points or len(points) < 1:
            return False

        first_point = points[0]
        if not isinstance(first_point, (list, tuple)) or len(first_point) != 2:
            return False

        x, y = first_point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False

        return True

    def get_line(self, line_id: str) -> Optional[CountingLine]:
        """
        Obtiene una línea por su ID.

        Args:
            line_id: ID de la línea

        Returns:
            Optional[CountingLine]: Línea encontrada o None
        """
        for line in self.lines:
            if line.id == line_id:
                return line
        return None

    def get_line_by_index(self, index: int) -> Optional[CountingLine]:
        """
        Obtiene una línea por su índice.

        Args:
            index: Índice de la línea

        Returns:
            Optional[CountingLine]: Línea encontrada o None
        """
        if 0 <= index < len(self.lines):
            return self.lines[index]
        return None

    def get_all_lines(self) -> List[CountingLine]:
        """Obtiene todas las líneas activas."""
        return [line for line in self.lines if line.enabled]

    def get_line_count(self) -> int:
        """Obtiene el número de líneas activas."""
        return len(self.get_all_lines())

    def get_total_lines(self) -> int:
        """Obtiene el número total de líneas (incluyendo desactivadas)."""
        return len(self.lines)

    def is_empty(self) -> bool:
        """Verifica si no hay líneas configuradas."""
        return len(self.lines) == 0

    def has_active_lines(self) -> bool:
        """Verifica si hay líneas activas."""
        return self.get_line_count() > 0

    def add_line(self, line: CountingLine) -> None:
        """
        Añade una nueva línea.

        Args:
            line: Línea a añadir
        """
        self.lines.append(line)

    def remove_line(self, line_id: str) -> bool:
        """
        Elimina una línea por su ID.

        Args:
            line_id: ID de la línea a eliminar

        Returns:
            bool: True si se eliminó correctamente
        """
        for i, line in enumerate(self.lines):
            if line.id == line_id:
                self.lines.pop(i)
                return True
        return False

    def toggle_line(self, line_id: str) -> bool:
        """
        Activa o desactiva una línea.

        Args:
            line_id: ID de la línea

        Returns:
            bool: True si se encontró la línea
        """
        line = self.get_line(line_id)
        if line:
            line.enabled = not line.enabled
            return True
        return False

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convierte todas las líneas a diccionario."""
        return [line.to_dict() for line in self.lines]

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor."""
        return {
            "total_lines": self.get_total_lines(),
            "active_lines": self.get_line_count(),
            "disabled_lines": self.get_total_lines() - self.get_line_count(),
            "line_ids": [line.id for line in self.lines],
            "line_names": [line.name for line in self.lines],
        }

    def __len__(self) -> int:
        """Retorna el número total de líneas."""
        return len(self.lines)

    def __iter__(self):
        """Iterador sobre las líneas."""
        return iter(self.lines)
