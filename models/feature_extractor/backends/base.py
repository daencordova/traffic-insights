"""
Interfaz abstracta para backends de extracción de features.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class FeatureBackend(ABC):
    """
    Interfaz abstracta para backends de extracción de features.

    Todos los backends deben implementar esta interfaz para ser
    utilizados por el FeatureExtractor.

    Attributes:
        feature_dim: Dimensión del vector de features
        is_available: Si el backend está disponible
    """

    @abstractmethod
    def extract(self, region: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae features de una región de imagen.

        Args:
            region: Región de imagen (recorte del objeto)

        Returns:
            Optional[np.ndarray]: Vector de features o None si falla
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """
        Calienta el backend para reducir latencia inicial.
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensión del vector de features."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el backend está disponible."""
        pass

    @property
    def name(self) -> str:
        """Nombre del backend."""
        return self.__class__.__name__.replace("Backend", "").lower()
