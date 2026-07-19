"""
Módulo de tracking avanzado con re-identificación robusta.
"""

from .base import AdvancedTracker
from .matcher import HierarchicalMatcher, MatchResult, MatchLevel
from .reidentifier import ReIdentificationSystem, ReIdentificationCandidate
from .feature_cache import FeatureCacheManager, FeatureEntry
from .validator import TrackValidator, ValidationRule
from .mht_integration import MHTIntegration
from .online_learner import OnlineFeatureLearner
from .sensor_fusion import SensorFusionTracker, SensorObservation, SensorType
from .path_predictor import PathPredictor, TrajectoryPrediction
from .managers.track_manager import TrackManager
from .state.state_machine import TrackStateMachine
from .state.track_updater import TrackUpdater
from .managers.feature_manager import FeatureManager

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
]
