"""Funciones utilitarias generales."""

import gc
from pathlib import Path
import time
from typing import Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def ensure_directory_exists(path: str) -> None:
    """Asegura que un directorio existe, creándolo si es necesario.

    Args:
        path: Ruta del directorio.

    Example:
        >>> ensure_directory_exists("data/screenshots/")
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp_filename(prefix: str = "", extension: str = "jpg") -> str:
    """Genera un nombre de archivo con timestamp.

    Args:
        prefix: Prefijo para el nombre.
        extension: Extensión del archivo.

    Returns:
        str: Nombre de archivo con timestamp.

    Example:
        >>> filename = get_timestamp_filename("capture", "jpg")
        >>> print(filename)  # "capture_20240101_120000.jpg"
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}" if prefix else f"{timestamp}.{extension}"


def format_time(seconds: float) -> str:
    """Formatea segundos en formato HH:MM:SS.

    Args:
        seconds: Segundos a formatear.

    Returns:
        str: String formateado en formato HH:MM:SS o MM:SS.

    Example:
        >>> format_time(3661)
        "01:01:01"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_memory_usage() -> dict[str, float]:
    """Obtiene información de uso de memoria del proceso actual.

    Returns:
        Dict[str, float]: Diccionario con información de memoria incluyendo:
            - rss_mb: Memoria residente en MB
            - vms_mb: Memoria virtual en MB
            - percent: Porcentaje de memoria del proceso
            - system_percent: Porcentaje de memoria del sistema
            - system_available_mb: Memoria disponible del sistema en MB

    Example:
        >>> mem = get_memory_usage()
        >>> print(f"Memory: {mem['rss_mb']:.2f} MB")
    """
    if not PSUTIL_AVAILABLE:
        return {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "percent": 0.0,
            "system_percent": 0.0,
            "system_available_mb": 0.0,
        }

    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "system_percent": psutil.virtual_memory().percent,
            "system_available_mb": psutil.virtual_memory().available / (1024 * 1024),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "percent": 0.0,
            "system_percent": 0.0,
            "system_available_mb": 0.0,
        }


def force_garbage_collection() -> dict[str, int | bool]:
    """Fuerza la recolección de basura y retorna estadísticas.

    Returns:
        Dict[str, Union[int, bool]]: Estadísticas de GC incluyendo:
            - collected_objects: Objetos recolectados
            - gc_enabled: Si GC está habilitado
            - garbage_count: Número de objetos en garbage

    Example:
        >>> stats = force_garbage_collection()
        >>> print(f"Recolectados: {stats['collected_objects']}")
    """
    collected = gc.collect()
    return {
        "collected_objects": collected,
        "gc_enabled": gc.isenabled(),
        "garbage_count": len(gc.garbage),
    }


class MemoryTracker:
    """Tracker simple de uso de memoria."""

    def __init__(self, name: str = "memory_tracker") -> None:
        """Inicializa el tracker de memoria.

        Args:
            name: Nombre identificador del tracker.
        """
        self.name: str = name
        self._snapshots: list[dict[str, Any]] = []
        self._max_snapshots: int = 100
        self._start_memory: float | None = None

    def snapshot(self, label: str = "") -> dict[str, Any]:
        """Toma una instantánea del uso de memoria.

        Args:
            label: Etiqueta para identificar la instantánea.

        Returns:
            Diccionario con información de memoria (timestamp, label, delta_mb, etc.).
        """
        memory = get_memory_usage()
        memory["timestamp"] = time.time()
        memory["label"] = label

        if self._start_memory is None:
            self._start_memory = memory["rss_mb"]

        memory["delta_mb"] = memory["rss_mb"] - self._start_memory

        self._snapshots.append(memory)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

        return memory

    def get_stats(self) -> dict[str, float]:
        """Obtiene estadísticas de las instantáneas.

        Returns:
            Diccionario con estadísticas (count, current_mb, peak_mb, delta_mb, start_mb).
        """
        if not self._snapshots:
            return {"count": 0.0}

        current = self._snapshots[-1]
        peak = max(s["rss_mb"] for s in self._snapshots)

        return {
            "count": float(len(self._snapshots)),
            "current_mb": current["rss_mb"],
            "peak_mb": peak,
            "delta_mb": current["delta_mb"],
            "start_mb": self._start_memory or 0.0,
        }

    def clear(self) -> None:
        """Limpia las instantáneas almacenadas."""
        self._snapshots.clear()
        self._start_memory = None
