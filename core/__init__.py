"""
Módulo core del sistema
"""

from core.interfaces import IDetector, ITracker, ICounter, IPipeline
from core.detector import YOLODetector, OptimizedYOLODetector, DetectorFactory
from core.pipeline.sync_pipeline import SyncPipeline
from core.pipeline.async_pipeline import AsyncPipeline

from core.tracker import (
    TrackMatcher,
    ReIDSystem,
    TrackValidator,
    FeatureCacheManager,
    MultiObjectTracker
)

from core.counter import (
    LineManager,
    CountingLine,
    CrossingDetector,
    StatisticsCollector,
    VehicleEvent,
    VehicleCounter
)

__all__ = [
    "YOLODetector",
    "OptimizedYOLODetector",
    "DetectorFactory",
    "MultiObjectTracker",
    "VehicleCounter",
    "SyncPipeline",
    "AsyncPipeline",
    "IDetector",
    "ITracker",
    "ICounter",
    "IPipeline",
    "TrackMatcher",
    "ReIDSystem",
    "TrackValidator",
    "FeatureCacheManager",
    "LineManager",
    "CountingLine",
    "CrossingDetector",
    "StatisticsCollector",
    "VehicleEvent",
]
