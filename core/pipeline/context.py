"""
Context manager para captura de video con manejo automático de recursos.

Este módulo proporciona context managers para manejar capturas de video
de forma segura, con reconexión automática y manejo de errores.
"""

import time
from typing import Optional, Union

import cv2

from utils.logger import LoggerMixin


class VideoCaptureContext(LoggerMixin):
    """
    Context manager para captura de video con reconexión automática.

    Responsabilidades:
    - Abrir y cerrar automáticamente la captura
    - Reintentar conexiones fallidas
    - Manejar errores de captura
    - Proporcionar estadísticas de la fuente

    Attributes:
        source: Fuente de video (número de dispositivo o ruta)
        width: Ancho deseado del frame
        height: Alto deseado del frame
        reconnect_attempts: Número de intentos de reconexión
        reconnect_delay: Delay entre intentos (segundos)
    """

    def __init__(
        self,
        source: Union[str, int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self._fps = 0.0
        self._frame_count = 0

        self.logger.info(
            "VideoCaptureContext inicializado",
            source=source,
            width=width,
            height=height
        )

    def __enter__(self) -> "VideoCaptureContext":
        """Abre la captura de video."""
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cierra la captura de video."""
        self.close()

    def _open(self) -> None:
        """Abre la captura con reintentos."""
        for attempt in range(self.reconnect_attempts):
            try:
                if isinstance(self.source, str) and self.source.isdigit():
                    self.cap = cv2.VideoCapture(int(self.source))
                else:
                    self.cap = cv2.VideoCapture(self.source)

                if self.cap.isOpened():
                    self._configure_capture()
                    self._is_open = True
                    self._fps = self.cap.get(cv2.CAP_PROP_FPS)

                    self.logger.info(
                        "Captura abierta exitosamente",
                        attempt=attempt + 1,
                        fps=self._fps
                    )
                    return

                self.logger.warning(
                    "Intento de apertura fallido",
                    attempt=attempt + 1
                )

                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)

            except Exception as e:
                self.logger.warning(
                    "Error abriendo captura",
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)

        raise RuntimeError(
            f"No se pudo abrir la fuente después de {self.reconnect_attempts} intentos: {self.source}"
        )

    def _configure_capture(self) -> None:
        """Configura la captura con los parámetros deseados."""
        if self.cap is None:
            return

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> tuple:
        """
        Lee un frame de la captura.

        Returns:
            tuple: (ret, frame) donde ret es booleano y frame es la imagen
        """
        if not self._is_open or self.cap is None:
            self.logger.warning("Intento de lectura con captura cerrada")
            return False, None

        try:
            ret, frame = self.cap.read()
            if ret:
                self._frame_count += 1
            return ret, frame
        except Exception as e:
            self.logger.error("Error leyendo frame", error=str(e))
            return False, None

    def get_fps(self) -> float:
        """Obtiene el FPS de la captura."""
        return self._fps

    def get_frame_size(self) -> tuple:
        """
        Obtiene el tamaño del frame.

        Returns:
            tuple: (width, height)
        """
        if self.cap is None:
            return (0, 0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def is_opened(self) -> bool:
        """Verifica si la captura está abierta."""
        return self._is_open and self.cap is not None and self.cap.isOpened()

    def get_stats(self) -> dict:
        """Obtiene estadísticas de la captura."""
        return {
            "source": self.source,
            "is_open": self.is_opened(),
            "fps": self._fps,
            "frame_count": self._frame_count,
            "width": self.width,
            "height": self.height,
        }

    def reconnect(self) -> bool:
        """
        Reconecta la captura.

        Returns:
            bool: True si la reconexión fue exitosa
        """
        self.close()
        time.sleep(0.5)

        try:
            self._open()
            return self.is_opened()
        except Exception as e:
            self.logger.error("Error en reconexión", error=str(e))
            return False

    def close(self) -> None:
        """Cierra la captura y libera recursos."""
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
        """Limpieza al destruir el objeto."""
        self.close()
