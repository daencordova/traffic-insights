"""
Módulo de conteo de vehículos.

Proporciona componentes para el conteo de vehículos a través de líneas virtuales.

Componentes principales:
- VehicleCounter: Orquestador principal del conteo
  └── Coordina LineManager, CrossingDetector y StatisticsCollector

- LineManager: Gestión de líneas de conteo
  └── Validación, acceso y estado de líneas

- CrossingDetector: Detección de cruces de líneas
  └── Detección de dirección, prevención de duplicados

- StatisticsCollector: Recolección de estadísticas
  └── Conteos por línea/clase, velocidades, eventos

Ejemplo de uso:
    >>> counter = VehicleCounter()
    >>> stats = counter.process(tracks, frame)
    >>> print(f"Total: {stats['total']}")
"""

from core.counter.base import VehicleCounter
from core.counter.crossing_detector import CrossingDetector
from core.counter.line_manager import CountingLine, LineManager
from core.counter.statistics_collector import StatisticsCollector, VehicleEvent

__all__ = [
    "VehicleCounter",
    "LineManager",
    "CountingLine",
    "CrossingDetector",
    "StatisticsCollector",
    "VehicleEvent",
]
