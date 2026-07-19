"""
Definición de interfaces abstractas para el sistema usando Protocol
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class IDetector(Protocol):
    """Interface para detectores de objetos"""

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta objetos en un frame"""
        ...

    def get_classes(self) -> List[int]:
        """Retorna las clases que detecta"""
        ...


@runtime_checkable
class ITracker(Protocol):
    """Interface para trackers de objetos"""

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Actualiza el tracker con nuevas detecciones"""
        ...

    def get_tracking_info(self) -> Dict[int, Dict[str, Any]]:
        """Retorna información de tracking actual"""
        ...

    def reset(self) -> None:
        """Reinicia el tracker"""
        ...


@runtime_checkable
class ICounter(Protocol):
    """Interface para contadores de objetos"""

    def process(self, tracks: Dict[int, Dict[str, Any]], frame: np.ndarray) -> Dict[str, Any]:
        """Procesa los tracks y actualiza los conteos"""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas actuales"""
        ...

    def reset(self) -> None:
        """Reinicia los contadores"""
        ...


@runtime_checkable
class IPipeline(Protocol):
    """Interface para el pipeline principal"""

    def run(self) -> None:
        """Ejecuta el pipeline principal"""
        ...

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame individual"""
        ...

    def pause(self) -> None:
        """Pausa la ejecución"""
        ...

    def resume(self) -> None:
        """Reanuda la ejecución"""
        ...

    def stop(self) -> None:
        """Detiene la ejecución"""
        ...
