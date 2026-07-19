"""
Pipeline de renderizado por capas.
"""

from enum import Enum, auto
from typing import Any, Dict, List

import numpy as np


class RenderLayer(Enum):
    """Capas de renderizado en orden de ejecución."""
    OVERLAY = auto()
    SYSTEM_INFO = auto()
    CONTROLS = auto()
    ERROR = auto()


class RenderPipeline:
    """
    Pipeline de renderizado por capas.
    Permite añadir/eliminar capas fácilmente.
    """

    __slots__ = ("_layers", "_renderers")

    def __init__(self):
        self._layers: List[RenderLayer] = []
        self._renderers: Dict[RenderLayer, Any] = {}

    def add_layer(self, layer: RenderLayer, renderer: Any) -> None:
        """Añade una capa de renderizado."""
        if layer not in self._layers:
            self._layers.append(layer)
            self._renderers[layer] = renderer

    def remove_layer(self, layer: RenderLayer) -> bool:
        """Elimina una capa de renderizado."""
        if layer in self._layers:
            self._layers.remove(layer)
            self._renderers.pop(layer, None)
            return True
        return False

    def render(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Ejecuta todas las capas de renderizado en orden.

        Args:
            frame: Frame a renderizar
            **kwargs: Argumentos para las capas

        Returns:
            np.ndarray: Frame renderizado
        """
        result = frame

        for layer in self._layers:
            renderer = self._renderers.get(layer)
            if renderer is not None:
                try:
                    result = renderer(result, **kwargs)
                except Exception:
                    continue

        return result
