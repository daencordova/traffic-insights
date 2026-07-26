"""
Módulo core del sistema
"""

from core.counter import (
    CountingLine,
    CrossingDetector,
    LineManager,
    StatisticsCollector,
    VehicleCounter,
    VehicleEvent,
)
from core.detector import DetectorFactory, OptimizedYOLODetector, YOLODetector
from core.interfaces import ICounter, IDetector, IPipeline, ITracker
from core.pipeline.async_pipeline import AsyncPipeline
from core.pipeline.sync_pipeline import SyncPipeline
from core.tracker import (
    FeatureCacheManager,
    MultiObjectTracker,
    ReIDSystem,
    TrackMatcher,
    TrackValidator,
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
