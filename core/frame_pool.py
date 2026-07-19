"""
Pool de frames para preasignación de memoria optimizado.
"""

import threading
import numpy as np
from typing import Optional, Tuple, List
from utils.logger import LoggerMixin


class FramePool(LoggerMixin):
    """
    Pool de frames para reutilización de memoria optimizado.

    Características:
    - Memoria preasignada para evitar reallocaciones
    - Thread-safe para uso en pipelines asíncronos
    - Estadísticas de uso
    - Limpieza automática
    - Liberación inmediata de memoria
    """

    def __init__(
        self,
        pool_size: int = 3,
        frame_shape: Tuple[int, int, int] = (480, 640, 3),
        dtype: np.dtype = np.uint8
    ) -> None:
        """
        Inicializa el pool de frames.

        Args:
            pool_size: Número de frames en el pool (reducido para ahorrar memoria).
            frame_shape: Shape de los frames (height, width, channels).
            dtype: Tipo de datos de los frames.
        """
        self.pool_size = pool_size
        self.frame_shape = frame_shape
        self.dtype = dtype

        self._pool: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._idx = 0

        self._stats = {
            "total_allocated": pool_size,
            "total_acquired": 0,
            "total_released": 0,
            "current_used": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "memory_used_mb": 0,
        }

        self._preallocate()

        self.logger.info(
            "FramePool inicializado",
            pool_size=pool_size,
            frame_shape=frame_shape,
            dtype=str(dtype)
        )

    def _preallocate(self) -> None:
        """Preasigna todos los frames del pool."""
        for _ in range(self.pool_size):
            frame = np.zeros(self.frame_shape, dtype=self.dtype)
            self._pool.append(frame)
            self._stats["memory_used_mb"] += frame.nbytes / (1024 * 1024)

    def acquire(self) -> np.ndarray:
        """
        Adquiere un frame del pool.

        Returns:
            np.ndarray: Frame del pool.
        """
        with self._lock:
            if not self._pool:
                self.logger.warning("Pool vacío, creando nuevo frame")
                frame = np.zeros(self.frame_shape, dtype=self.dtype)
                self._stats["total_allocated"] += 1
                self._stats["memory_used_mb"] += frame.nbytes / (1024 * 1024)
                return frame

            frame = self._pool.pop(0)
            self._stats["total_acquired"] += 1
            self._stats["current_used"] += 1
            self._stats["pool_hits"] += 1

            return frame

    def release(self, frame: np.ndarray) -> bool:
        """
        Libera un frame de vuelta al pool.

        Args:
            frame: Frame a liberar.

        Returns:
            bool: True si se liberó correctamente.
        """
        with self._lock:
            if frame is None or frame.size == 0:
                return False

            if len(self._pool) >= self.pool_size:
                self.logger.debug("Pool lleno, descartando frame")
                frame.fill(0)
                return False

            frame.fill(0)
            self._pool.append(frame)
            self._stats["total_released"] += 1
            self._stats["current_used"] -= 1

            return True

    def get_stats(self) -> dict:
        """Obtiene estadísticas del pool."""
        with self._lock:
            return {
                **self._stats,
                "pool_size": self.pool_size,
                "available": len(self._pool),
                "frame_shape": self.frame_shape,
                "memory_used_mb": self._stats["memory_used_mb"],
            }

    def clear(self) -> None:
        """Limpia el pool y libera memoria."""
        with self._lock:
            for frame in self._pool:
                frame.fill(0)
            self._pool.clear()
            self._stats["total_allocated"] = 0
            self._stats["current_used"] = 0
            self._stats["memory_used_mb"] = 0

        import gc
        gc.collect()
        self.logger.info("FramePool limpiado y memoria liberada")

    def resize(self, new_size: int) -> None:
        """
        Cambia el tamaño del pool.

        Args:
            new_size: Nuevo tamaño del pool.
        """
        with self._lock:
            current_size = len(self._pool)

            if new_size > current_size:
                for _ in range(new_size - current_size):
                    frame = np.zeros(self.frame_shape, dtype=self.dtype)
                    self._pool.append(frame)
                    self._stats["memory_used_mb"] += frame.nbytes / (1024 * 1024)
            elif new_size < current_size:
                for _ in range(current_size - new_size):
                    if self._pool:
                        frame = self._pool.pop()
                        frame.fill(0)
                        self._stats["memory_used_mb"] -= frame.nbytes / (1024 * 1024)

            self.pool_size = new_size
            self._stats["total_allocated"] = new_size

        import gc
        gc.collect()
        self.logger.info(
            "FramePool redimensionado",
            new_size=new_size,
            current_used=self._stats["current_used"]
        )

    def __len__(self) -> int:
        """Retorna el número de frames disponibles en el pool."""
        with self._lock:
            return len(self._pool)
