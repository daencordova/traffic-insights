"""
Módulo de aprendizaje en línea para tracking.

Proporciona componentes para el aprendizaje en línea de features:
- Estadísticas de features
- Estrategias de aprendizaje
- Detección de concept drift
- Agregación de features
"""

from .statistics import FeatureStatistics
from .strategies import (
    LearningStrategy,
    IncrementalStrategy,
    AdaptiveStrategy,
    BatchStrategy,
    HybridStrategy,
    LearningStrategyFactory,
)
from .drift_detector import ConceptDriftDetector
from .aggregator import FeatureAggregator

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
