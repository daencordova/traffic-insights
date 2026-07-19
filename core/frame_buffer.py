"""
Buffer circular optimizado para almacenamiento de frames con bajo overhead
"""

from collections import deque
from typing import Optional, Deque, Tuple
import threading
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


class BufferStatus(Enum):
    """Estados del buffer"""
    EMPTY = auto()
    PARTIAL = auto()
    FULL = auto()
    OVERFLOW = auto()
    DRAINING = auto()


@dataclass
class FrameMetadata:
    """Metadatos asociados a un frame"""
    timestamp: float
    frame_number: int
    source_fps: float
    capture_time_ms: float
    processing_time_ms: float = 0.0
    dropped: bool = False

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class CircularFrameBuffer:
    """
    Buffer circular optimizado con memoria preasignada para frames

    Características:
    - Memoria preasignada para evitar reallocaciones
    - Soporte para drops selectivos
    - Estadísticas de rendimiento
    - Thread-safe
    - Liberación inmediata de frames para optimización de memoria
    """

    def __init__(
        self,
        max_size: int = 30,
        frame_shape: Optional[Tuple[int, int, int]] = None,
        dtype: np.dtype = np.uint8,
        drop_policy: str = "oldest"
    ):
        """
        Args:
            max_size: Tamaño máximo del buffer
            frame_shape: Shape predefinido para preasignar memoria
            dtype: Tipo de datos de los frames
            drop_policy: Política de drop cuando está lleno
        """
        self.max_size = max_size
        self.drop_policy = drop_policy
        self.dtype = dtype

        self._preallocated = frame_shape is not None
        if self._preallocated:
            self._buffer = np.zeros((max_size, *frame_shape), dtype=dtype)
        else:
            self._buffer: Deque[np.ndarray] = deque(maxlen=max_size)

        self._metadata: Deque[FrameMetadata] = deque(maxlen=max_size)
        self._lock = threading.RLock()

        self._total_frames_received = 0
        self._total_frames_dropped = 0
        self._total_frames_processed = 0
        self._buffer_overflow_count = 0

        self._head = 0
        self._tail = 0
        self._count = 0

        self._status = BufferStatus.EMPTY

        self._last_watermark_time = time.time()
        self._watermark_history: Deque[float] = deque(maxlen=60)

        self._memory_freed = 0
        self._total_memory_allocated = 0

    def put(self, frame: np.ndarray, metadata: Optional[FrameMetadata] = None) -> bool:
        """
        Inserta un frame en el buffer

        Returns:
            True si se insertó correctamente, False si fue descartado
        """
        if frame is None or frame.size == 0:
            return False

        with self._lock:
            self._total_frames_received += 1
            frame_size = frame.nbytes

            if metadata is None:
                metadata = FrameMetadata(
                    timestamp=time.time(),
                    frame_number=self._total_frames_received,
                    source_fps=0.0
                )

            if self._is_full():
                self._handle_overflow()
                self._total_frames_dropped += 1
                self._buffer_overflow_count += 1
                self._memory_freed += frame_size
                return False

            if self._preallocated:
                self._buffer[self._tail] = frame.copy()
            else:
                self._buffer.append(frame.copy())

            self._metadata.append(metadata)
            self._count += 1
            self._tail = (self._tail + 1) % self.max_size
            self._total_memory_allocated += frame_size

            self._update_status()

            return True

    def get(self, block: bool = True, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """
        Obtiene el siguiente frame del buffer y libera la memoria inmediatamente.

        Args:
            block: Si debe bloquear esperando un frame
            timeout: Timeout en segundos si block=True

        Returns:
            Tupla (frame, metadata) o None si no hay frames
        """
        start_time = time.time()

        while True:
            with self._lock:
                if self._count > 0:
                    if self._preallocated:
                        frame = self._buffer[self._head].copy()
                        self._buffer[self._head].fill(0)
                    else:
                        frame = self._buffer.popleft()

                    metadata = self._metadata.popleft()

                    self._count -= 1
                    self._head = (self._head + 1) % self.max_size
                    self._total_frames_processed += 1

                    if not self._preallocated and frame is not None:
                        pass

                    self._update_status()

                    return frame, metadata

                if not block or time.time() - start_time >= timeout:
                    return None

            time.sleep(0.001)

    def get_batch(self, batch_size: int, timeout: float = 0.05) -> list:
        """
        Obtiene un lote de frames liberando memoria inmediatamente.

        Args:
            batch_size: Tamaño máximo del lote
            timeout: Timeout para esperar frames

        Returns:
            Lista de tuplas (frame, metadata)
        """
        batch = []
        start_time = time.time()

        while len(batch) < batch_size and (time.time() - start_time) < timeout:
            result = self.get(block=False)
            if result is None:
                time.sleep(0.001)
                continue
            batch.append(result)

        return batch

    def peek(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Mira el siguiente frame sin removerlo"""
        with self._lock:
            if self._count == 0:
                return None

            if self._preallocated:
                frame = self._buffer[self._head].copy()
            else:
                frame = self._buffer[0].copy()

            metadata = self._metadata[0]
            return frame, metadata

    def clear(self) -> int:
        """Limpia el buffer y retorna el número de frames eliminados"""
        with self._lock:
            count = self._count
            if self._preallocated:
                for i in range(self.max_size):
                    self._buffer[i].fill(0)
            else:
                self._buffer.clear()
            self._metadata.clear()
            self._head = 0
            self._tail = 0
            self._count = 0
            self._update_status()
            return count

    def _is_full(self) -> bool:
        """Verifica si el buffer está lleno"""
        return self._count >= self.max_size

    def _handle_overflow(self):
        """Maneja el overflow según la política configurada"""
        if self.drop_policy == "oldest":
            if self._preallocated:
                old_frame = self._buffer[self._head]
                self._memory_freed += old_frame.nbytes
                old_frame.fill(0)
                self._head = (self._head + 1) % self.max_size
            else:
                old_frame = self._buffer.popleft()
                if old_frame is not None:
                    self._memory_freed += old_frame.nbytes
            self._metadata.popleft()
            self._count -= 1

        elif self.drop_policy == "newest":
            pass

        elif self.drop_policy == "lowest_priority":
            if self._preallocated:
                old_frame = self._buffer[self._head]
                self._memory_freed += old_frame.nbytes
                old_frame.fill(0)
                self._head = (self._head + 1) % self.max_size
            else:
                old_frame = self._buffer.popleft()
                if old_frame is not None:
                    self._memory_freed += old_frame.nbytes
            self._metadata.popleft()
            self._count -= 1

    def _update_status(self):
        """Actualiza el estado del buffer"""
        ratio = self._count / self.max_size

        if self._count == 0:
            self._status = BufferStatus.EMPTY
        elif ratio >= 0.9:
            self._status = BufferStatus.OVERFLOW
        elif ratio >= 0.7:
            self._status = BufferStatus.FULL
        else:
            self._status = BufferStatus.PARTIAL

        current_time = time.time()
        if current_time - self._last_watermark_time >= 1.0:
            self._watermark_history.append(ratio)
            self._last_watermark_time = current_time

    def get_stats(self) -> dict:
        """Obtiene estadísticas del buffer"""
        with self._lock:
            return {
                "size": self._count,
                "max_size": self.max_size,
                "capacity_ratio": self._count / self.max_size if self.max_size > 0 else 0,
                "status": self._status.name,
                "total_frames_received": self._total_frames_received,
                "total_frames_dropped": self._total_frames_dropped,
                "total_frames_processed": self._total_frames_processed,
                "drop_rate": self._total_frames_dropped / max(1, self._total_frames_received),
                "overflow_count": self._buffer_overflow_count,
                "avg_watermark": sum(self._watermark_history) / max(1, len(self._watermark_history)),
                "preallocated": self._preallocated,
                "drop_policy": self.drop_policy,
                "memory_freed_mb": self._memory_freed / (1024 * 1024),
                "total_memory_allocated_mb": self._total_memory_allocated / (1024 * 1024),
            }

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count >= self.max_size

    @property
    def status(self) -> BufferStatus:
        with self._lock:
            return self._status
