"""
Módulo core del sistema
"""

from core.interfaces import IDetector, ITracker, ICounter, IPipeline
from core.detector import YOLODetector, OptimizedYOLODetector, DetectorFactory
from core.pipeline.sync_pipeline import VehicleCountingPipeline
from core.pipeline.async_pipeline import AsyncVehicleCountingPipeline

from core.tracker import (
    HierarchicalMatcher,
    ReIdentificationSystem,
    TrackValidator,
    FeatureCacheManager,
    AdvancedTracker
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
    "AdvancedTracker",
    "VehicleCounter",
    "VehicleCountingPipeline",
    "AsyncVehicleCountingPipeline",
    "IDetector",
    "ITracker",
    "ICounter",
    "IPipeline",
    "HierarchicalMatcher",
    "ReIdentificationSystem",
    "TrackValidator",
    "FeatureCacheManager",
    "LineManager",
    "CountingLine",
    "CrossingDetector",
    "StatisticsCollector",
    "VehicleEvent",
]
