"""
Módulo de tracking avanzado con re-identificación robusta.
"""

from core.tracker.base import AdvancedTracker
from core.tracker.matcher import HierarchicalMatcher, MatchResult, MatchLevel
from core.tracker.reidentifier import ReIdentificationSystem, ReIdentificationCandidate
from core.tracker.feature_cache import FeatureCacheManager, FeatureEntry
from core.tracker.validator import TrackValidator, ValidationRule
from core.tracker.mht_integration import MHTIntegration
from core.tracker.online_learner import OnlineFeatureLearner
from core.tracker.sensor_fusion import SensorFusionTracker, SensorObservation, SensorType
from core.tracker.path_predictor import PathPredictor, TrajectoryPrediction
from core.tracker.managers.track_manager import TrackManager
from core.tracker.state.state_machine import TrackStateMachine
from core.tracker.state.track_updater import TrackUpdater
from core.tracker.managers.feature_manager import FeatureManager

from core.tracker.prediction import (
    TrajectoryHistory,
    TrajectorySample,
    MotionModel,
    LinearModel,
    CurvedModel,
    CyclicModel,
    PolynomialModel,
    AdaptiveModel,
    MotionModelFactory,
    ModelSelector,
    StateDetector,
    TrajectoryState,
    CollisionDetector,
)

from core.tracker.learning import (
    FeatureStatistics,
    LearningStrategy,
    IncrementalStrategy,
    AdaptiveStrategy,
    BatchStrategy,
    HybridStrategy,
    LearningStrategyFactory,
    ConceptDriftDetector,
    FeatureAggregator,
)

__all__ = [
    "AdvancedTracker",
    "HierarchicalMatcher",
    "MatchResult",
    "MatchLevel",
    "ReIdentificationSystem",
    "ReIdentificationCandidate",
    "FeatureCacheManager",
    "FeatureEntry",
    "TrackValidator",
    "ValidationRule",
    "MHTIntegration",
    "OnlineFeatureLearner",
    "SensorFusionTracker",
    "SensorObservation",
    "SensorType",
    "PathPredictor",
    "TrajectoryPrediction",
    "TrackManager",
    "TrackStateMachine",
    "TrackUpdater",
    "FeatureManager",
    "TrajectoryHistory",
    "TrajectorySample",
    "MotionModel",
    "LinearModel",
    "CurvedModel",
    "CyclicModel",
    "PolynomialModel",
    "AdaptiveModel",
    "MotionModelFactory",
    "ModelSelector",
    "StateDetector",
    "TrajectoryState",
    "CollisionDetector",
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
