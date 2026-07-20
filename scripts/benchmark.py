"""
Benchmark para comparar rendimiento entre versiones optimizadas y no optimizadas.
"""

import time
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import config_manager
from core.detector import YOLODetector
from core.detector.optimized import OptimizedYOLODetector


def benchmark_detector(detector_class, name: str, frames: list) -> dict:
    """Benchmark de un detector."""
    detector = detector_class()

    total_time = 0
    total_detections = 0

    start_total = time.perf_counter()

    for frame in frames:
        start = time.perf_counter()
        detections = detector.detect(frame)
        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_detections += len(detections)

    elapsed_total = time.perf_counter() - start_total

    return {
        "name": name,
        "total_time": elapsed_total,
        "avg_time": elapsed_total / len(frames) * 1000,
        "total_detections": total_detections,
        "fps": len(frames) / elapsed_total,
    }


def run_benchmark():
    """Ejecuta benchmark del sistema."""
    print("=" * 60)
    print("⚡ BENCHMARK DE RENDIMIENTO (CPU)")
    print("=" * 60)

    config_manager.load_from_file("config.yaml")

    frames = []
    for i in range(50):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        x1 = 50 + i * 5
        y1 = 50 + i * 3
        x2 = x1 + 100
        y2 = y1 + 80

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        frames.append(frame)

    print(f"📹 Frames de prueba: {len(frames)}")

    results = []

    print("\n📊 DETECTORES:")
    results.append(benchmark_detector(YOLODetector, "YOLO (PyTorch)", frames))
    results.append(benchmark_detector(OptimizedYOLODetector, "YOLO Optimizado (ONNX+Numba)", frames))

    print("\n" + "=" * 60)
    print("📊 RESULTADOS")
    print("=" * 60)

    for result in results:
        print(f"\n{result['name']}:")
        print(f"  ⏱️  Tiempo total: {result['total_time']:.2f}s")
        print(f"  ⏱️  Tiempo promedio: {result['avg_time']:.2f}ms")
        print(f"  📦 Detecciones: {result['total_detections']}")
        print(f"  🚀 FPS: {result['fps']:.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_benchmark()
