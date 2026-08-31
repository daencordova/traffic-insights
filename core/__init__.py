"""
Core module of the system.

This module provides the main components of the computer vision pipeline,
including detection, tracking, counting, and pipeline orchestration.

Submodules:
    - core.detector: Object detection using YOLO models
    - core.tracker: Multi-object tracking with re-identification
    - core.counter: Vehicle counting and line crossing detection
    - core.pipeline: Synchronous and asynchronous processing pipelines
    - core.interfaces: Abstract interfaces for system components

The module exports all major components for easy import:
    - Detection: YOLODetector, OptimizedYOLODetector, DetectorFactory
    - Tracking: MultiObjectTracker, ReIDSystem, TrackMatcher
    - Counting: VehicleCounter, CountingLine, LineManager
    - Pipeline: SyncPipeline, AsyncPipeline for processing
    - Interfaces: IDetector, ITracker, ICounter, IPipeline

Example:
    >>> from core import YOLODetector, MultiObjectTracker, SyncPipeline
    >>> detector = YOLODetector(model_path="yolov8n.pt")
    >>> tracker = MultiObjectTracker()
    >>> pipeline = SyncPipeline(detector, tracker)
    >>> results = pipeline.process_frame(frame)
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
    "TrackMatcher",
    "ReIDSystem",
    "TrackValidator",
    "FeatureCacheManager",
    "VehicleCounter",
    "CountingLine",
    "LineManager",
    "CrossingDetector",
    "StatisticsCollector",
    "VehicleEvent",
    "SyncPipeline",
    "AsyncPipeline",
    "IDetector",
    "ITracker",
    "ICounter",
    "IPipeline",
]
