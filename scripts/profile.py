"""
Script de perfilamiento para identificar cuellos de botella en CPU.
"""

import cProfile
import pstats
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import config_manager
from core.detector.optimized import OptimizedYOLODetector
from core.tracker.base import MultiObjectTracker


def run_profile():
    """Ejecuta perfilamiento del sistema."""
    print("=" * 60)
    print("🔍 PERFILAMIENTO DEL SISTEMA (CPU)")
    print("=" * 60)

    config_manager.load_from_file("config.yaml")

    detector = OptimizedYOLODetector()
    tracker = MultiObjectTracker()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(frame, (300, 150), (400, 250), (255, 255, 255), -1)

    print(f"📹 Frame size: {frame.shape}")
    print(f"📦 Detector: {type(detector).__name__}")
    print(f"📦 Tracker: {type(tracker).__name__}")

    print("\n⏳ Ejecutando profiling...")

    profiler = cProfile.Profile()
    profiler.enable()

    num_iterations = 50
    for i in range(num_iterations):
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats(20)

    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DE RENDIMIENTO")
    print("=" * 60)
    print(stream.getvalue())

    det_stats = detector.get_performance_stats()
    print("\n📊 DETECTOR:")
    print(f"  Avg inference time: {det_stats.get('avg_inference_time_ms', 0):.2f}ms")
    print(f"  Cache hit ratio: {det_stats.get('cache_hit_ratio', 0):.2%}")
    print(f"  ONNX available: {det_stats.get('onnx_available', False)}")
    print(f"  Numba available: {det_stats.get('numba_available', False)}")

    trk_stats = tracker.get_stats()
    print("\n📊 TRACKER:")
    print(f"  Active tracks: {trk_stats.get('active_tracks', 0)}")
    print(f"  Tracking time: {trk_stats.get('tracking_time_ms', 0):.2f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_profile()
