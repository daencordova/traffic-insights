"""
Manejador de reconexión para captura de video.

Proporciona lógica de reconexión automática con reintentos y delays.
"""

from typing import Optional, Union

import cv2

from utils.logger import LoggerMixin


class Reconnector(LoggerMixin):
    """
    Gestiona la reconexión a fuentes de video.

    Attributes:
        max_attempts: Número máximo de intentos de reconexión.
        delay: Delay entre intentos en segundos.
    """

    def __init__(self, max_attempts: int = 5, delay: float = 1.0):
        """
        Inicializa el reconector.

        Args:
            max_attempts: Número máximo de intentos.
            delay: Delay entre intentos en segundos.
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self._attempts = 0

        self.logger.info(
            "Reconnector inicializado",
            max_attempts=max_attempts,
            delay=delay
        )

    def connect(
        self,
        source: Union[str, int],
        config = None
    ) -> Optional[cv2.VideoCapture]:
        """
        Intenta conectar a la fuente con reintentos.

        Args:
            source: Fuente de video (número o ruta).
            config: Configuración de la cámara (opcional).

        Returns:
            Optional[cv2.VideoCapture]: Captura conectada o None.
        """
        self._attempts = 0

        while self._attempts < self.max_attempts:
            try:
                self._attempts += 1

                if isinstance(source, str) and source.isdigit():
                    cap = cv2.VideoCapture(int(source))
                else:
                    cap = cv2.VideoCapture(source)

                if cap.isOpened():
                    self._configure_capture(cap, config)
                    self.logger.info(
                        "Conectado exitosamente",
                        source=source,
                        attempts=self._attempts
                    )
                    self._attempts = 0
                    return cap

                self.logger.warning(
                    "Intento de conexión fallido",
                    source=source,
                    attempt=self._attempts,
                    max_attempts=self.max_attempts
                )

                if self._attempts < self.max_attempts:
                    import time
                    time.sleep(self.delay)

            except Exception as e:
                self.logger.warning(
                    "Error en conexión",
                    source=source,
                    attempt=self._attempts,
                    error=str(e)
                )
                if self._attempts < self.max_attempts:
                    import time
                    time.sleep(self.delay)

        self.logger.error(
            "No se pudo conectar después de múltiples intentos",
            source=source,
            attempts=self.max_attempts
        )
        return None

    def _configure_capture(self, cap: cv2.VideoCapture, config) -> None:
        """
        Configura la captura con los parámetros deseados.

        Args:
            cap: Captura a configurar.
            config: Configuración de la cámara.
        """
        if config is None:
            return

        if hasattr(config, "width") and config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)

        if hasattr(config, "height") and config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    def reset(self) -> None:
        """Reinicia el contador de intentos."""
        self._attempts = 0
        self.logger.debug("Reconnector reiniciado")

    def get_stats(self) -> dict:
        """Obtiene estadísticas del reconector."""
        return {
            "max_attempts": self.max_attempts,
            "delay": self.delay,
            "current_attempts": self._attempts,
        }
