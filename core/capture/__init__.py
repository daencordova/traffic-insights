"""
Módulo de captura de video.

Proporciona componentes para la captura y gestión de flujos de video.
"""

from .manager import CaptureManager
from .reconnector import Reconnector

__all__ = [
    "CaptureManager",
    "Reconnector",
]
