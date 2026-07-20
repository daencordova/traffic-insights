"""
Módulo de conteo de vehículos.

Proporciona componentes para el conteo de vehículos a través de líneas virtuales.

Componentes principales:
- VehicleCounter: Orquestador principal del conteo
- LineManager: Gestión de líneas de conteo
- CrossingDetector: Detección de cruces de líneas
- StatisticsCollector: Recolección de estadísticas
"""

from core.counter.base import VehicleCounter
from core.counter.line_manager import LineManager, CountingLine
from core.counter.crossing_detector import CrossingDetector
from core.counter.statistics_collector import StatisticsCollector, VehicleEvent

__all__ = [
    "VehicleCounter",
    "LineManager",
    "CountingLine",
    "CrossingDetector",
    "StatisticsCollector",
    "VehicleEvent",
]
