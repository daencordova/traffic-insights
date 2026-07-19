"""
Estrategias de aprendizaje para features en línea.

Implementa diferentes estrategias de aprendizaje: incremental,
adaptativo, batch y híbrido.
"""

from abc import ABC, abstractmethod

import numpy as np

from core.tracker.learning.statistics import FeatureStatistics


class LearningStrategy(ABC):
    """Interfaz abstracta para estrategias de aprendizaje."""

    @abstractmethod
    def update(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        confidence: float,
        learning_rate: float
    ) -> np.ndarray:
        """
        Actualiza el feature promedio.

        Args:
            stats: Estadísticas del track
            features: Nuevos features
            confidence: Confianza de la observación
            learning_rate: Tasa de aprendizaje

        Returns:
            np.ndarray: Features actualizados
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre de la estrategia."""
        pass


class IncrementalStrategy(LearningStrategy):
    """
    Estrategia incremental simple.

    Actualiza la media con un factor de aprendizaje fijo.
    """

    def update(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        confidence: float,
        learning_rate: float
    ) -> np.ndarray:
        alpha = min(learning_rate, 1.0 / (stats.n_samples + 1))
        stats.mean_features = (1 - alpha) * stats.mean_features + alpha * features
        return stats.mean_features

    @property
    def name(self) -> str:
        return "incremental"


class AdaptiveStrategy(LearningStrategy):
    """
    Estrategia adaptativa con factor de confianza.

    Ajusta la tasa de aprendizaje basada en la confianza
    y el número de muestras.
    """

    def update(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        confidence: float,
        learning_rate: float
    ) -> np.ndarray:
        confidence_factor = 0.5 + 0.5 * confidence

        sample_factor = min(1.0, 5 / max(1, stats.n_samples))

        final_lr = learning_rate * confidence_factor * (0.7 + 0.3 * sample_factor)
        alpha = min(final_lr, 0.3)

        stats.mean_features = (1 - alpha) * stats.mean_features + alpha * features
        return stats.mean_features

    @property
    def name(self) -> str:
        return "adaptive"


class BatchStrategy(LearningStrategy):
    """
    Estrategia por lotes.

    Actualiza usando el promedio del historial completo.
    """

    def update(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        confidence: float,
        learning_rate: float
    ) -> np.ndarray:
        history = list(stats.feature_history)
        if len(history) < 5:
            history.append(features)

        if history:
            batch_mean = np.mean(history, axis=0)
            norm = np.linalg.norm(batch_mean)
            if norm > 0:
                batch_mean = batch_mean / norm

            alpha = 0.3 if len(history) > 5 else 0.1
            stats.mean_features = (1 - alpha) * stats.mean_features + alpha * batch_mean

        return stats.mean_features

    @property
    def name(self) -> str:
        return "batch"


class HybridStrategy(LearningStrategy):
    """
    Estrategia híbrida: incremental + batch periódico.

    Combina actualización incremental con re-cálculo batch periódico.
    """

    def update(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        confidence: float,
        learning_rate: float
    ) -> np.ndarray:
        alpha = min(learning_rate, 1.0 / (stats.n_samples + 1))
        stats.mean_features = (1 - alpha) * stats.mean_features + alpha * features

        if stats.total_updates % 10 == 0 and stats.n_samples > 5:
            history = list(stats.feature_history)
            if len(history) >= 5:
                batch_mean = np.mean(history, axis=0)
                norm = np.linalg.norm(batch_mean)
                if norm > 0:
                    batch_mean = batch_mean / norm
                    stats.mean_features = 0.7 * stats.mean_features + 0.3 * batch_mean

        return stats.mean_features

    @property
    def name(self) -> str:
        return "hybrid"


class LearningStrategyFactory:
    """Fábrica de estrategias de aprendizaje."""

    _strategies = {
        "incremental": IncrementalStrategy,
        "adaptive": AdaptiveStrategy,
        "batch": BatchStrategy,
        "hybrid": HybridStrategy,
    }

    @classmethod
    def create(cls, strategy_type: str) -> LearningStrategy:
        """
        Crea una estrategia de aprendizaje.

        Args:
            strategy_type: Tipo de estrategia

        Returns:
            LearningStrategy: Estrategia de aprendizaje

        Raises:
            ValueError: Si la estrategia no es soportada
        """
        strategy_class = cls._strategies.get(strategy_type)
        if strategy_class is None:
            raise ValueError(f"Estrategia no soportada: {strategy_type}")
        return strategy_class()

    @classmethod
    def get_available_strategies(cls) -> list:
        """Obtiene la lista de estrategias disponibles."""
        return list(cls._strategies.keys())
