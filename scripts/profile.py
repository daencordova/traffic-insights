#!/usr/bin/env python
"""Script de perfilamiento para identificar cuellos de botella en CPU."""

import cProfile
import io
from pathlib import Path
import pstats
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import config_manager
from core.detector.optimized import OptimizedYOLODetector
from core.tracker.base import MultiObjectTracker
from utils.logger import setup_logger

logger = setup_logger("profile", level="INFO")


def run_profile():
    """Ejecuta perfilamiento del sistema."""
    logger.info("=" * 60)
    logger.info("🔍 PERFILAMIENTO DEL SISTEMA (CPU)")
    logger.info("=" * 60)

    config_manager.load_from_file("config.yaml")

    detector = OptimizedYOLODetector()
    tracker = MultiObjectTracker()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(frame, (300, 150), (400, 250), (255, 255, 255), -1)

    logger.info(f"📹 Frame size: {frame.shape}")
    logger.info(f"📦 Detector: {type(detector).__name__}")
    logger.info(f"📦 Tracker: {type(tracker).__name__}")

    logger.info("\n⏳ Ejecutando profiling...")

    profiler = cProfile.Profile()
    profiler.enable()

    num_iterations = 50
    for _ in range(num_iterations):
        detections = detector.detect(frame)
        _ = tracker.update(detections, frame)

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats(20)

    logger.info("\n" + "=" * 60)
    logger.info("📊 ESTADÍSTICAS DE RENDIMIENTO")
    logger.info("=" * 60)
    logger.info(stream.getvalue())

    det_stats = detector.get_performance_stats()
    logger.info("\n📊 DETECTOR:")
    logger.info(f"  Avg inference time: {det_stats.get('avg_inference_time_ms', 0):.2f}ms")
    logger.info(f"  Cache hit ratio: {det_stats.get('cache_hit_ratio', 0):.2%}")
    logger.info(f"  ONNX available: {det_stats.get('onnx_available', False)}")
    logger.info(f"  Numba available: {det_stats.get('numba_available', False)}")

    trk_stats = tracker.get_stats()
    logger.info("\n📊 TRACKER:")
    logger.info(f"  Active tracks: {trk_stats.get('active_tracks', 0)}")
    logger.info(f"  Tracking time: {trk_stats.get('tracking_time_ms', 0):.2f}ms")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    run_profile()
