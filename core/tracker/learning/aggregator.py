"""
Agregador de features para aprendizaje en línea.

Maneja la agregación y fusión de features de diferentes fuentes.
"""

from typing import Optional, Dict, Any, List
import numpy as np

from core.tracker.learning.statistics import FeatureStatistics


class FeatureAggregator:
    """
    Agregador de features.

    Responsabilidades:
    - Agregar features de diferentes fuentes
    - Fusionar estadísticas de tracks
    - Calcular features promedio ponderados

    Attributes:
        _stats: Estadísticas del agregador
    """

    def __init__(self):
        self._stats = {
            "total_aggregations": 0,
            "total_merges": 0,
            "avg_merge_size": 0.0,
        }

    def aggregate_features(
        self,
        features_list: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """
        Agrega múltiples features en uno solo.

        Args:
            features_list: Lista de features a agregar
            weights: Pesos para cada feature (opcional)

        Returns:
            Optional[np.ndarray]: Feature agregado o None
        """
        if not features_list:
            return None

        if weights is None:
            weights = [1.0] * len(features_list)

        total_weight = sum(weights)
        if total_weight <= 0:
            return None

        normalized_weights = [w / total_weight for w in weights]

        aggregated = np.zeros_like(features_list[0])
        for feature, weight in zip(features_list, normalized_weights):
            aggregated += weight * feature

        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm

        self._stats["total_aggregations"] += 1
        return aggregated

    def merge_statistics(
        self,
        target_stats: FeatureStatistics,
        source_stats: FeatureStatistics
    ) -> FeatureStatistics:
        """
        Fusiona dos conjuntos de estadísticas.

        Args:
            target_stats: Estadísticas destino
            source_stats: Estadísticas origen

        Returns:
            FeatureStatistics: Estadísticas fusionadas
        """
        total_samples = target_stats.n_samples + source_stats.n_samples
        if total_samples == 0:
            return target_stats

        weight_target = target_stats.n_samples / total_samples
        weight_source = source_stats.n_samples / total_samples

        target_stats.mean_features = (
            weight_target * target_stats.mean_features +
            weight_source * source_stats.mean_features
        )
        target_stats.n_samples = total_samples
        target_stats.total_updates += source_stats.total_updates

        for feat in source_stats.feature_history:
            target_stats.feature_history.append(feat)

        for conf in source_stats.confidence_history:
            target_stats.confidence_history.append(conf)

        for ts in source_stats.timestamps:
            target_stats.timestamps.append(ts)

        target_stats.quality_score = (
            weight_target * target_stats.quality_score +
            weight_source * source_stats.quality_score
        )

        self._stats["total_merges"] += 1
        self._stats["avg_merge_size"] = (
            (self._stats["avg_merge_size"] * (self._stats["total_merges"] - 1) +
             total_samples) / self._stats["total_merges"]
        )

        return target_stats

    def compute_quality_score(self, stats: FeatureStatistics) -> float:
        """
        Calcula una puntuación de calidad para las estadísticas.

        Args:
            stats: Estadísticas a evaluar

        Returns:
            float: Puntuación de calidad (0-1)
        """
        if stats.n_samples < 5:
            return 0.0

        sample_score = min(1.0, stats.n_samples / 20.0)

        confidence_mean = stats.get_average_confidence()
        confidence_score = min(1.0, confidence_mean / 0.7)

        stability_score = 1.0
        if len(stats.feature_history) > 5:
            history = list(stats.feature_history)
            similarities = []
            for i in range(1, len(history)):
                norm1 = np.linalg.norm(history[i-1])
                norm2 = np.linalg.norm(history[i])
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(history[i-1], history[i]) / (norm1 * norm2)
                    similarities.append(max(0.0, sim))
            if similarities:
                stability_score = float(np.mean(similarities))

        drift_score = 0.0 if stats.concept_drift_detected else 1.0

        quality = (
            0.30 * sample_score +
            0.25 * confidence_score +
            0.25 * stability_score +
            0.20 * drift_score
        )

        return max(0.0, min(1.0, quality))

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del agregador."""
        return self._stats

    def reset(self) -> None:
        """Reinicia las estadísticas."""
        self._stats = {
            "total_aggregations": 0,
            "total_merges": 0,
            "avg_merge_size": 0.0,
        }
