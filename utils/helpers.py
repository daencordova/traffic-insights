"""
Funciones utilitarias generales
"""

import time
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def ensure_directory_exists(path: str) -> None:
    """
    Asegura que un directorio existe, creándolo si es necesario

    Args:
        path: Ruta del directorio
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp_filename(prefix: str = "", extension: str = "jpg") -> str:
    """
    Genera un nombre de archivo con timestamp

    Args:
        prefix: Prefijo para el nombre
        extension: Extensión del archivo

    Returns:
        Nombre de archivo con timestamp
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}" if prefix else f"{timestamp}.{extension}"


def format_time(seconds: float) -> str:
    """
    Formatea segundos en formato HH:MM:SS

    Args:
        seconds: Segundos a formatear

    Returns:
        String formateado
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def get_memory_usage() -> Dict[str, float]:
    """
    Obtiene información de uso de memoria del proceso actual

    Returns:
        Diccionario con información de memoria
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


def force_garbage_collection() -> Dict[str, Union[int, bool]]:
    """
    Fuerza la recolección de basura y retorna estadísticas

    Returns:
        Diccionario con estadísticas de GC
    """
    collected = gc.collect()
    return {
        "collected_objects": collected,
        "gc_enabled": gc.isenabled(),
        "garbage_count": len(gc.garbage),
    }


class MemoryTracker:
    """Tracker simple de uso de memoria"""

    def __init__(self, name: str = "memory_tracker") -> None:
        self.name: str = name
        self._snapshots: List[Dict[str, Any]] = []
        self._max_snapshots: int = 100
        self._start_memory: Optional[float] = None

    def snapshot(self, label: str = "") -> Dict[str, Any]:
        """
        Toma una instantánea del uso de memoria

        Args:
            label: Etiqueta para identificar la instantánea

        Returns:
            Diccionario con información de memoria
        """
        memory = get_memory_usage()
        memory["timestamp"] = time.time()
        memory["label"] = label

        if self._start_memory is None:
            self._start_memory = memory["rss_mb"]

        memory["delta_mb"] = memory["rss_mb"] - self._start_memory

        self._snapshots.append(memory)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

        return memory

    def get_stats(self) -> Dict[str, float]:
        """Obtiene estadísticas de las instantáneas"""
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
        """Limpia las instantáneas"""
        self._snapshots.clear()
        self._start_memory = None
