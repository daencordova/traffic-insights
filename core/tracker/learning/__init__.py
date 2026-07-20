"""
Módulo de aprendizaje en línea para tracking.

Proporciona componentes para el aprendizaje en línea de features:
- Estadísticas de features
- Estrategias de aprendizaje
- Detección de concept drift
- Agregación de features
"""

from core.tracker.learning.statistics import FeatureStatistics
from core.tracker.learning.strategies import (
    LearningStrategy,
    IncrementalStrategy,
    AdaptiveStrategy,
    BatchStrategy,
    HybridStrategy,
    LearningStrategyFactory,
)
from core.tracker.learning.drift_detector import ConceptDriftDetector
from core.tracker.learning.aggregator import FeatureAggregator

__all__ = [
    "FeatureStatistics",
    "LearningStrategy",
    "IncrementalStrategy",
    "AdaptiveStrategy",
    "BatchStrategy",
    "HybridStrategy",
    "LearningStrategyFactory",
    "ConceptDriftDetector",
    "FeatureAggregator",
]
