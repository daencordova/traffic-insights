"""
Módulo core del sistema
"""

from .detector import YOLODetector, OptimizedYOLODetector, DetectorFactory
from .tracker import AdvancedTracker
from .counter import VehicleCounter
from .pipeline.sync_pipeline import VehicleCountingPipeline
from .pipeline.async_pipeline import AsyncVehicleCountingPipeline
from .interfaces import IDetector, ITracker, ICounter, IPipeline

from .tracker import (
    HierarchicalMatcher,
    ReIdentificationSystem,
    TrackValidator,
    FeatureCacheManager,
)

from .counter import (
    LineManager,
    CountingLine,
    CrossingDetector,
    StatisticsCollector,
    VehicleEvent,
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
