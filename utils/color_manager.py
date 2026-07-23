"""
Gestor de colores para tracks y visualizaciones.

Proporciona una asignación consistente y eficiente de colores
para elementos visuales en el sistema.
"""

from typing import Dict, Tuple, List, Optional, Any
import threading


class ColorManager:
    """
    Gestor de colores con caché para tracks.

    Características:
    - Asignación consistente de colores por ID
    - Pool de colores predefinidos
    - Thread-safe para uso en pipelines asíncronos
    - Soporte para colores personalizados
    - Estadísticas de uso

    Attributes:
        _color_pool: Lista de colores predefinidos en formato BGR
        _color_cache: Caché de colores asignados por ID
        _lock: Lock para operaciones thread-safe
        _stats: Estadísticas de uso del gestor
    """

    DEFAULT_COLORS = [
        (0, 255, 0),
        (255, 165, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 128, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 255, 128),
        (255, 0, 128),
        (128, 255, 0),
        (0, 128, 128),
        (128, 128, 0),
        (128, 0, 128),
        (0, 0, 255),
    ]

    HIGH_CONTRAST_COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    def __init__(self, colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        Inicializa el gestor de colores.

        Args:
            colors: Lista de colores personalizada (opcional).
                   Si no se proporciona, usa DEFAULT_COLORS.
        """
        self._color_pool = colors or self.DEFAULT_COLORS
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}
        self._lock = threading.RLock()
        self._next_color_index = 0

        self._stats = {
            "total_assignments": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "active_colors": 0,
            "pool_size": len(self._color_pool),
        }

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Obtiene un color para un track.

        Args:
            track_id: ID del track.

        Returns:
            Tuple[int, int, int]: Color en formato BGR.

        Example:
            >>> color_manager = ColorManager()
            >>> color = color_manager.get_color(42)
            >>> print(color)  # (255, 0, 0)
        """
        with self._lock:
            if track_id in self._color_cache:
                self._stats["cache_hits"] += 1
                return self._color_cache[track_id]

            self._stats["cache_misses"] += 1

            color = self._assign_color(track_id)
            self._color_cache[track_id] = color
            self._stats["total_assignments"] += 1
            self._stats["active_colors"] = len(self._color_cache)

            return color

    def _assign_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Asigna un color a un track.

        Args:
            track_id: ID del track.

        Returns:
            Tuple[int, int, int]: Color asignado en formato BGR.
        """
        index = self._next_color_index % len(self._color_pool)
        self._next_color_index += 1

        if self._next_color_index >= len(self._color_pool) * 2:
            return self._generate_color_from_id(track_id)

        return self._color_pool[index]

    def _generate_color_from_id(self, track_id: int) -> Tuple[int, int, int]:
        """
        Genera un color a partir del ID del track.

        Args:
            track_id: ID del track.

        Returns:
            Tuple[int, int, int]: Color generado en formato BGR.
        """
        hue = (track_id * 137) % 360
        saturation = 200
        value = 200

        h = hue / 60.0
        c = value * saturation / 255.0
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        b = int((b + m) * 255)
        g = int((g + m) * 255)
        r = int((r + m) * 255)

        return (b, g, r)

    def get_color_palette(self, count: int) -> List[Tuple[int, int, int]]:
        """
        Obtiene una paleta de colores para múltiples elementos.

        Args:
            count: Número de colores a obtener.

        Returns:
            List[Tuple[int, int, int]]: Lista de colores en formato BGR.

        Example:
            >>> colors = color_manager.get_color_palette(5)
            >>> for i, color in enumerate(colors):
            ...     print(f"Elemento {i}: {color}")
        """
        with self._lock:
            palette = []
            for i in range(count):
                temp_id = i
                palette.append(self.get_color(temp_id))
            return palette

    def get_colors_for_tracks(
        self,
        track_ids: List[int]
    ) -> Dict[int, Tuple[int, int, int]]:
        """
        Obtiene colores para una lista de tracks de forma eficiente.

        Args:
            track_ids: Lista de IDs de tracks.

        Returns:
            Dict[int, Tuple[int, int, int]]: Mapeo de ID a color.

        Example:
            >>> active_tracks = [1, 5, 10, 15]
            >>> colors = color_manager.get_colors_for_tracks(active_tracks)
            >>> for track_id, color in colors.items():
            ...     print(f"Track {track_id}: {color}")
        """
        result = {}
        for track_id in track_ids:
            result[track_id] = self.get_color(track_id)
        return result

    def get_color_with_alpha(
        self,
        track_id: int,
        alpha: float = 0.5
    ) -> Tuple[int, int, int, float]:
        """
        Obtiene un color con transparencia para un track.

        Args:
            track_id: ID del track.
            alpha: Nivel de transparencia (0-1).

        Returns:
            Tuple[int, int, int, float]: Color en formato BGRA.

        Example:
            >>> color = color_manager.get_color_with_alpha(42, 0.7)
            >>> # Útil para overlays semitransparentes
        """
        b, g, r = self.get_color(track_id)
        return (b, g, r, alpha)

    def get_brighter_color(self, track_id: int, factor: float = 1.3) -> Tuple[int, int, int]:
        """
        Obtiene una versión más brillante del color de un track.

        Args:
            track_id: ID del track.
            factor: Factor de brillo (>1 para más brillante).

        Returns:
            Tuple[int, int, int]: Color más brillante en formato BGR.
        """
        b, g, r = self.get_color(track_id)

        b = min(255, int(b * factor))
        g = min(255, int(g * factor))
        r = min(255, int(r * factor))

        return (b, g, r)

    def get_darker_color(self, track_id: int, factor: float = 0.7) -> Tuple[int, int, int]:
        """
        Obtiene una versión más oscura del color de un track.

        Args:
            track_id: ID del track.
            factor: Factor de oscuridad (0-1).

        Returns:
            Tuple[int, int, int]: Color más oscuro en formato BGR.
        """
        b, g, r = self.get_color(track_id)

        b = max(0, int(b * factor))
        g = max(0, int(g * factor))
        r = max(0, int(r * factor))

        return (b, g, r)

    def clear_cache(self) -> None:
        """Limpia el caché de colores."""
        with self._lock:
            count = len(self._color_cache)
            self._color_cache.clear()
            self._next_color_index = 0
            self._stats["active_colors"] = 0
            self._stats["cache_hits"] = 0
            self._stats["cache_misses"] = 0
            self._stats["total_assignments"] = 0
            print(f"🧹 Caché de colores limpiado: {count} colores eliminados")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del gestor de colores.

        Returns:
            Dict[str, any]: Estadísticas de uso.

        Example:
            >>> stats = color_manager.get_stats()
            >>> print(f"Colores activos: {stats['active_colors']}")
            >>> print(f"Tasa de aciertos: {stats['hit_rate']:.2%}")
        """
        with self._lock:
            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]

            return {
                **self._stats,
                "hit_rate": self._stats["cache_hits"] / max(1, total_requests),
                "miss_rate": self._stats["cache_misses"] / max(1, total_requests),
                "color_pool_size": len(self._color_pool),
            }

    def __len__(self) -> int:
        """Retorna el número de colores en caché."""
        with self._lock:
            return len(self._color_cache)


_default_color_manager = None


def get_color_manager() -> ColorManager:
    """
    Obtiene la instancia global del gestor de colores.

    Returns:
        ColorManager: Instancia global del gestor.

    Example:
        >>> color_manager = get_color_manager()
        >>> color = color_manager.get_color(42)
    """
    global _default_color_manager
    if _default_color_manager is None:
        _default_color_manager = ColorManager()
    return _default_color_manager


def get_color(index: int) -> Tuple[int, int, int]:
    """
    Función de compatibilidad para el código existente.

    Args:
        index: Índice o ID para obtener color.

    Returns:
        Tuple[int, int, int]: Color en formato BGR.

    Note:
        Esta función mantiene compatibilidad con la función anterior
        get_color en utils/geometry.py.
    """
    return get_color_manager().get_color(index)
