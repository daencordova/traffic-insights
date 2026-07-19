"""
Gestión de recursos con context managers y pool
"""

import threading
import time
from typing import Optional, Dict, Any, Callable, TypeVar, Generic, Generator
from contextlib import contextmanager
from collections import deque

T = TypeVar('T')


class ResourceManager(Generic[T]):
    """
    Gestor genérico de recursos con límite de tiempo de vida
    """

    def __init__(self, max_lifetime: float = 60.0, cleanup_interval: float = 10.0):
        self.max_lifetime = max_lifetime
        self.cleanup_interval = cleanup_interval
        self._resources: Dict[str, T] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def register(self, key: str, resource: T) -> None:
        """
        Registra un recurso con su timestamp

        Args:
            key: Clave única del recurso
            resource: Recurso a registrar
        """
        with self._lock:
            self._resources[key] = resource
            self._timestamps[key] = time.time()

    def get(self, key: str) -> Optional[T]:
        """
        Obtiene un recurso por su clave

        Args:
            key: Clave del recurso

        Returns:
            Recurso o None si no existe o ha expirado
        """
        with self._lock:
            if key not in self._resources:
                return None

            if self._is_expired(key):
                self._remove(key)
                return None

            return self._resources[key]

    def remove(self, key: str) -> bool:
        """
        Elimina un recurso

        Args:
            key: Clave del recurso

        Returns:
            True si se eliminó correctamente
        """
        with self._lock:
            return self._remove(key)

    def _remove(self, key: str) -> bool:
        """Método interno para eliminar recurso"""
        if key in self._resources:
            resource = self._resources.pop(key)
            self._timestamps.pop(key, None)
            self._cleanup_resource(resource)
            return True
        return False

    def _is_expired(self, key: str) -> bool:
        """Verifica si un recurso ha expirado"""
        if key not in self._timestamps:
            return True

        age = time.time() - self._timestamps[key]
        return age > self.max_lifetime

    def _cleanup_resource(self, resource: T) -> None:
        """Limpia un recurso específico"""
        try:
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'release'):
                resource.release()
            elif hasattr(resource, 'clear'):
                resource.clear()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Limpia recursos expirados"""
        current_time = time.time()
        if current_time - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            expired_keys = [
                key for key in self._resources.keys()
                if self._is_expired(key)
            ]

            for key in expired_keys:
                self._remove(key)

            self._last_cleanup = current_time

    def stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor"""
        with self._lock:
            return {
                "total_resources": len(self._resources),
                "keys": list(self._resources.keys()),
                "max_lifetime_seconds": self.max_lifetime,
            }


@contextmanager
def managed_resource(manager: ResourceManager, key: str, resource: T) -> Generator[T, None, None]:
    """
    Context manager para recursos gestionados

    Args:
        manager: Gestor de recursos
        key: Clave del recurso
        resource: Recurso a gestionar

    Yields:
        Recurso gestionado
    """
    try:
        manager.register(key, resource)
        yield resource
    finally:
        manager.remove(key)
