#!/usr/bin/env python
"""
Scripts de utilidad para el sistema de seguimiento de tráfico.

Este módulo contiene scripts para:
- benchmark.py: Pruebas de rendimiento entre versiones optimizadas y no optimizadas
- profile.py: Perfilamiento para identificar cuellos de botella en CPU
- check_structure.py: Verificación de estructura del proyecto (__init__.py en todos los módulos)

Uso:
    python scripts/benchmark.py
    python scripts/profile.py
"""

from scripts import benchmark, profile

__all__ = [
    "benchmark",
    "profile",
]
