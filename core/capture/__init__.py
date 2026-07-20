"""
Módulo de captura de video.

Proporciona componentes para la captura y gestión de flujos de video.
"""

from core.capture.manager import CaptureManager
from core.capture.reconnector import Reconnector

__all__ = [
    "CaptureManager",
    "Reconnector",
]
