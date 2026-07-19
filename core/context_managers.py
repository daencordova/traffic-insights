"""
Context managers especializados para gestión de recursos
"""

import time
import gc
import threading
from contextlib import contextmanager
from typing import Optional, Generator, Any, Dict, Callable, TypeVar, Union
from pathlib import Path
import cv2
import numpy as np

from utils.logger import LoggerMixin
from utils.helpers import get_memory_usage, force_garbage_collection
from core.constants import MEMORY_CHECK_INTERVAL, GC_INTERVAL

T = TypeVar('T')


@contextmanager
def timer_context(name: str = "operation") -> Generator[float, None, None]:
    """
    Context manager para medir tiempo de ejecución

    Args:
        name: Nombre de la operación a medir

    Yields:
        Tiempo transcurrido en segundos
    """
    start_time = time.perf_counter()
    try:
        yield 0.0
    finally:
        elapsed = time.perf_counter() - start_time
        pass


@contextmanager
def memory_tracker_context(name: str = "memory") -> Generator[Dict[str, float], None, None]:
    """
    Context manager para monitorear uso de memoria

    Args:
        name: Nombre del contexto

    Yields:
        Diccionario con estadísticas de memoria
    """
    start_memory = get_memory_usage()
    start_time = time.time()

    try:
        yield {
            "start_memory_mb": start_memory.get("rss_mb", 0),
            "start_time": start_time,
            "name": name,
        }
    finally:
        end_memory = get_memory_usage()
        end_time = time.time()

        stats = {
            "name": name,
            "duration_seconds": end_time - start_time,
            "start_memory_mb": start_memory.get("rss_mb", 0),
            "end_memory_mb": end_memory.get("rss_mb", 0),
            "memory_delta_mb": end_memory.get("rss_mb", 0) - start_memory.get("rss_mb", 0),
            "system_percent": end_memory.get("system_percent", 0),
        }


@contextmanager
def video_capture_context(source: Union[str, int]) -> Generator[cv2.VideoCapture, None, None]:
    """
    Context manager para captura de video con manejo automático de recursos

    Args:
        source: Fuente de video (número de dispositivo o ruta)

    Yields:
        Objeto VideoCapture configurado

    Raises:
        RuntimeError: Si no se puede abrir la fuente
    """
    cap = None
    try:
        if isinstance(source, str) and source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source}")

        yield cap
    finally:
        if cap is not None:
            cap.release()


@contextmanager
def image_window_context(window_name: str) -> Generator[None, None, None]:
    """
    Context manager para ventanas de imagen con limpieza automática

    Args:
        window_name: Nombre de la ventana
    """
    try:
        yield
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


@contextmanager
def lock_context(lock: threading.Lock, timeout: Optional[float] = None) -> Generator[bool, None, None]:
    """
    Context manager para locks con timeout opcional

    Args:
        lock: Objeto Lock a adquirir
        timeout: Timeout en segundos (opcional)

    Yields:
        True si se adquirió el lock, False si timeout
    """
    acquired = False
    try:
        if timeout is not None:
            acquired = lock.acquire(timeout=timeout)
        else:
            lock.acquire()
            acquired = True

        yield acquired
    finally:
        if acquired:
            lock.release()


@contextmanager
def file_context(filepath: str, mode: str = "r", encoding: str = "utf-8") -> Generator[Any, None, None]:
    """
    Context manager para archivos con manejo automático

    Args:
        filepath: Ruta del archivo
        mode: Modo de apertura
        encoding: Codificación

    Yields:
        Objeto archivo abierto
    """
    path = Path(filepath)
    if "w" in mode or "a" in mode:
        path.parent.mkdir(parents=True, exist_ok=True)

    f = None
    try:
        f = open(filepath, mode, encoding=encoding)
        yield f
    finally:
        if f is not None:
            f.close()


@contextmanager
def gc_context(aggressive: bool = False) -> Generator[Dict[str, int], None, None]:
    """
    Context manager para control de garbage collection

    Args:
        aggressive: Si usar limpieza agresiva

    Yields:
        Estadísticas de GC
    """
    gc.disable()
    stats_before = {
        "garbage_count": len(gc.garbage),
        "gc_enabled": gc.isenabled(),
    }

    try:
        yield stats_before
    finally:
        gc.enable()
        collected = gc.collect()
        if aggressive:
            for _ in range(3):
                gc.collect()

        stats_after = {
            "collected_objects": collected,
            "garbage_count": len(gc.garbage),
            "gc_enabled": gc.isenabled(),
        }


@contextmanager
def performance_context(name: str = "operation") -> Generator[Dict[str, Any], None, None]:
    """
    Context manager para medir rendimiento completo (tiempo + memoria)

    Args:
        name: Nombre de la operación

    Yields:
        Diccionario con estadísticas de rendimiento
    """
    start_time = time.perf_counter()
    start_memory = get_memory_usage()

    try:
        yield {
            "name": name,
            "start_time": start_time,
            "start_memory_mb": start_memory.get("rss_mb", 0),
        }
    finally:
        end_time = time.perf_counter()
        end_memory = get_memory_usage()

        stats = {
            "name": name,
            "duration_ms": (end_time - start_time) * 1000,
            "duration_seconds": end_time - start_time,
            "memory_delta_mb": end_memory.get("rss_mb", 0) - start_memory.get("rss_mb", 0),
            "start_memory_mb": start_memory.get("rss_mb", 0),
            "end_memory_mb": end_memory.get("rss_mb", 0),
        }


class VideoCaptureContext(LoggerMixin):
    """
    Context manager avanzado para captura de video con reconexión automática
    """

    def __init__(
        self,
        source: Union[str, int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

        self.logger.info(
            "Inicializando VideoCaptureContext",
            source=source,
            width=width,
            height=height
        )

    def __enter__(self) -> "VideoCaptureContext":
        """Abre la captura de video"""
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cierra la captura de video"""
        self.close()

    def _open(self) -> None:
        """Abre la captura con reintentos"""
        for attempt in range(self.reconnect_attempts):
            try:
                if isinstance(self.source, str) and self.source.isdigit():
                    self.cap = cv2.VideoCapture(int(self.source))
                else:
                    self.cap = cv2.VideoCapture(self.source)

                if self.cap.isOpened():
                    self._configure_capture()
                    self._is_open = True
                    self.logger.info("Captura abierta exitosamente", attempt=attempt + 1)
                    return

                self.logger.warning("Intento de apertura fallido", attempt=attempt + 1)
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)

            except Exception as e:
                self.logger.warning("Error abriendo captura", attempt=attempt + 1, error=str(e))
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)

        raise RuntimeError(f"No se pudo abrir la fuente después de {self.reconnect_attempts} intentos")

    def _configure_capture(self) -> None:
        """Configura la captura"""
        if self.cap is None:
            return

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> tuple:
        """Lee un frame de la captura"""
        if not self._is_open or self.cap is None:
            self.logger.warning("Intento de lectura con captura cerrada")
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.debug("No se pudo leer frame")
            return ret, frame
        except Exception as e:
            self.logger.error("Error leyendo frame", error=str(e))
            return False, None

    def get_fps(self) -> float:
        """Obtiene el FPS de la captura"""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_size(self) -> tuple:
        """Obtiene el tamaño del frame"""
        if self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def is_opened(self) -> bool:
        """Verifica si la captura está abierta"""
        return self._is_open and self.cap is not None and self.cap.isOpened()

    def close(self) -> None:
        """Cierra la captura"""
        if self.cap is not None:
            try:
                self.cap.release()
                self.logger.debug("Captura liberada")
            except Exception as e:
                self.logger.warning("Error liberando captura", error=str(e))
            finally:
                self.cap = None
                self._is_open = False

    def __del__(self) -> None:
        """Limpieza al destruir el objeto"""
        self.close()


class ResourcePool(LoggerMixin):
    """
    Pool de recursos para reutilización de objetos costosos
    """

    def __init__(self, max_size: int = 10, timeout: float = 30.0) -> None:
        self.max_size = max_size
        self.timeout = timeout
        self._pool: Dict[str, list] = {}
        self._lock = threading.Lock()

        self.logger.info("ResourcePool inicializado", max_size=max_size, timeout=timeout)

    def get(self, resource_type: str, creator: Callable[[], T]) -> Optional[T]:
        """
        Obtiene un recurso del pool o crea uno nuevo

        Args:
            resource_type: Tipo de recurso
            creator: Función que crea el recurso

        Returns:
            Recurso o None si no se puede crear
        """
        with lock_context(self._lock, timeout=self.timeout) as acquired:
            if not acquired:
                self.logger.warning("Timeout adquiriendo lock", resource_type=resource_type)
                return None

            if resource_type not in self._pool:
                self._pool[resource_type] = []

            pool = self._pool[resource_type]

            while pool:
                resource = pool.pop()
                if self._is_valid(resource):
                    self.logger.debug("Recurso reutilizado", resource_type=resource_type)
                    return resource

            try:
                resource = creator()
                self.logger.debug("Recurso creado", resource_type=resource_type)
                return resource
            except Exception as e:
                self.logger.error("Error creando recurso", resource_type=resource_type, error=str(e))
                return None

    def release(self, resource_type: str, resource: T) -> bool:
        """
        Libera un recurso de vuelta al pool

        Args:
            resource_type: Tipo de recurso
            resource: Recurso a liberar

        Returns:
            True si se liberó correctamente
        """
        with lock_context(self._lock, timeout=self.timeout) as acquired:
            if not acquired:
                self.logger.warning("Timeout adquiriendo lock para release", resource_type=resource_type)
                return False

            if resource_type not in self._pool:
                self._pool[resource_type] = []

            pool = self._pool[resource_type]

            if len(pool) < self.max_size and self._is_valid(resource):
                pool.append(resource)
                self.logger.debug("Recurso liberado al pool", resource_type=resource_type)
                return True

            self._cleanup_resource(resource)
            return False

    def _is_valid(self, resource: Any) -> bool:
        """Verifica si un recurso es válido para reutilización"""
        if resource is None:
            return False

        if isinstance(resource, cv2.VideoCapture):
            return resource.isOpened()

        if isinstance(resource, np.ndarray):
            return resource.size > 0

        return True

    def _cleanup_resource(self, resource: Any) -> None:
        """Limpia un recurso antes de descartarlo"""
        try:
            if isinstance(resource, cv2.VideoCapture):
                resource.release()
            elif isinstance(resource, np.ndarray):
                pass
        except Exception as e:
            self.logger.debug("Error limpiando recurso", error=str(e))

    def clear(self, resource_type: Optional[str] = None) -> None:
        """
        Limpia el pool

        Args:
            resource_type: Tipo específico a limpiar, o None para todos
        """
        with lock_context(self._lock) as acquired:
            if not acquired:
                return

            if resource_type is None:
                for rt in list(self._pool.keys()):
                    self._clear_pool(rt)
            elif resource_type in self._pool:
                self._clear_pool(resource_type)

    def _clear_pool(self, resource_type: str) -> None:
        """Limpia un pool específico"""
        pool = self._pool.get(resource_type, [])
        for resource in pool:
            self._cleanup_resource(resource)
        pool.clear()
        self.logger.debug("Pool limpiado", resource_type=resource_type)

    def stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del pool"""
        with lock_context(self._lock) as acquired:
            if not acquired:
                return {}

            return {
                "total_types": len(self._pool),
                "total_resources": sum(len(pool) for pool in self._pool.values()),
                "resources_by_type": {
                    rt: len(pool) for rt, pool in self._pool.items()
                },
                "max_size": self.max_size,
            }
