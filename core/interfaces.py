"""
Definición de interfaces abstractas para el sistema usando Protocol.

Este módulo define los contratos que deben cumplir los componentes
principales del sistema: detector, tracker, counter y pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IDetector(Protocol):
    """
    Interfaz para detectores de objetos.

    Define el contrato que deben cumplir todos los detectores
    de objetos en el sistema.
    """

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta objetos en un frame.

        Args:
            frame: Imagen en formato numpy array (H, W, C) en BGR.

        Returns:
            List[Dict[str, Any]]: Lista de detecciones, donde cada detección
                contiene al menos: 'box', 'centroid', 'confidence', 'class_id'.

        Raises:
            DetectionError: Si ocurre un error durante la detección.
        """
        ...

    def get_classes(self) -> List[int]:
        """
        Retorna las clases que el detector está configurado para detectar.

        Returns:
            List[int]: Lista de IDs de clases.
        """
        ...


@runtime_checkable
class ITracker(Protocol):
    """
    Interfaz para trackers de objetos.

    Define el contrato que deben cumplir todos los sistemas
    de seguimiento de objetos.
    """

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Actualiza el tracker con nuevas detecciones.

        Args:
            detections: Lista de detecciones del frame actual.
            frame: Imagen actual para extraer features visuales.

        Returns:
            Dict[int, Dict[str, Any]]: Diccionario de tracks activos,
                donde la clave es el track_id y el valor contiene
                información como centroid, bbox, estado, etc.

        Raises:
            TrackingError: Si ocurre un error durante el tracking.
        """
        ...

    def get_tracking_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Retorna información de tracking actual.

        Returns:
            Dict[int, Dict[str, Any]]: Estado actual de todos los tracks activos.
        """
        ...

    def reset(self) -> None:
        """
        Reinicia el tracker completamente.

        Elimina todos los tracks, limpia cachés y reinicia contadores.
        """
        ...


@runtime_checkable
class ICounter(Protocol):
    """
    Interfaz para contadores de objetos.

    Define el contrato que deben cumplir todos los sistemas
    de conteo de objetos.
    """

    def process(self, tracks: Dict[int, Dict[str, Any]], frame: np.ndarray) -> Dict[str, Any]:
        """
        Procesa los tracks y actualiza los conteos.

        Args:
            tracks: Diccionario de tracks activos.
            frame: Imagen actual para referencias espaciales.

        Returns:
            Dict[str, Any]: Estadísticas actualizadas del conteo.
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas actuales.

        Returns:
            Dict[str, Any]: Estadísticas detalladas del conteo.
        """
        ...

    def reset(self) -> None:
        """
        Reinicia los contadores.

        Limpia todos los conteos y estadísticas acumuladas.
        """
        ...


@runtime_checkable
class IPipeline(Protocol):
    """
    Interfaz para el pipeline principal.

    Define el contrato que deben cumplir todos los pipelines
    de procesamiento de video.
    """

    def run(self) -> None:
        """
        Ejecuta el pipeline principal.

        Inicia el procesamiento continuo de video hasta que
        se detenga explícitamente.
        """
        ...

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame individual.

        Args:
            frame: Imagen a procesar.

        Returns:
            np.ndarray: Frame procesado con visualizaciones.
        """
        ...

    def pause(self) -> None:
        """
        Pausa la ejecución del pipeline.

        Detiene temporalmente el procesamiento de nuevos frames.
        """
        ...

    def resume(self) -> None:
        """
        Reanuda la ejecución del pipeline.

        Continúa el procesamiento después de una pausa.
        """
        ...

    def stop(self) -> None:
        """
        Detiene la ejecución del pipeline.

        Termina el procesamiento y libera recursos.
        """
        ...
