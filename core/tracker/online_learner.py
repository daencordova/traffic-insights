"""
Sistema de aprendizaje en línea para adaptación de features en tracking.

Orquesta los componentes de estadísticas, estrategias de aprendizaje,
detector de concept drift y agregador de features.
"""

from typing import Dict, Optional, List, Any
import time
from collections import deque

import numpy as np

from utils.logger import LoggerMixin
from core.tracker.learning.statistics import FeatureStatistics
from core.tracker.learning.strategies import LearningStrategyFactory
from core.tracker.learning.drift_detector import ConceptDriftDetector
from core.tracker.learning.aggregator import FeatureAggregator


class OnlineFeatureLearner(LoggerMixin):
    """
    Sistema de aprendizaje en línea para features de tracking.

    Orquesta los componentes de estadísticas, estrategias de aprendizaje,
    detector de concept drift y agregador de features.

    Características:
    - Actualización incremental de features
    - Múltiples estrategias de aprendizaje
    - Detección de concept drift
    - Fusión de tracks
    - Curvas de aprendizaje
    - Estadísticas de rendimiento
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        learning_rate: float = 0.05,
        min_samples: int = 5,
        drift_threshold: float = 0.35,
        max_history: int = 50,
        strategy: str = "adaptive"
    ) -> None:
        """
        Inicializa el sistema de aprendizaje en línea.

        Args:
            feature_dim: Dimensión de los features
            learning_rate: Tasa de aprendizaje base
            min_samples: Mínimo de muestras para estadísticas robustas
            drift_threshold: Umbral para detección de cambio de concepto
            max_history: Tamaño máximo del historial por track
            strategy: Estrategia de aprendizaje
        """
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.max_history = max_history

        self._stats: Dict[int, FeatureStatistics] = {}
        self._strategy = LearningStrategyFactory.create(strategy)
        self._drift_detector = ConceptDriftDetector(
            threshold=drift_threshold,
            min_samples=min_samples,
            history_size=max_history
        )
        self._aggregator = FeatureAggregator()

        self._global_stats = {
            "total_tracks": 0,
            "total_updates": 0,
            "total_drifts_detected": 0,
            "active_learners": 0,
            "avg_learning_rate": learning_rate,
            "strategy": strategy,
            "start_time": time.time(),
        }

        self.logger.info(
            "OnlineFeatureLearner inicializado",
            feature_dim=feature_dim,
            learning_rate=learning_rate,
            strategy=strategy,
            drift_threshold=drift_threshold
        )

    def update(
        self,
        track_id: int,
        features: np.ndarray,
        confidence: float = 1.0,
        force: bool = False
    ) -> np.ndarray:
        """
        Actualiza el feature promedio con la nueva observación.

        Args:
            track_id: ID del track a actualizar
            features: Nuevo vector de features
            confidence: Confianza de la observación (0-1)
            force: Forzar actualización incluso si baja confianza

        Returns:
            np.ndarray: Features actualizados (promedio)
        """
        if features is None or len(features) == 0:
            return features

        if len(features) != self.feature_dim:
            self.logger.warning(
                "Dimensión de features incorrecta",
                expected=self.feature_dim,
                actual=len(features),
                track_id=track_id
            )
            return features

        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        else:
            return features

        if track_id not in self._stats:
            self._stats[track_id] = FeatureStatistics(
                mean_features=features.copy(),
                n_samples=1,
                confidence_history=deque(maxlen=self.max_history),
                timestamps=deque(maxlen=self.max_history),
                feature_history=deque(maxlen=self.max_history)
            )
            self._stats[track_id].add_sample(features, confidence)
            self._global_stats["total_tracks"] += 1
            self._global_stats["active_learners"] += 1

            self.logger.debug(
                "Nuevo learner creado",
                track_id=track_id,
                n_samples=1
            )
            return features

        stats = self._stats[track_id]

        if confidence < 0.2 and not force:
            self.logger.debug(
                "Confianza insuficiente para actualizar",
                track_id=track_id,
                confidence=confidence,
                threshold=0.2
            )
            return stats.mean_features

        if self._drift_detector.detect_drift(track_id, stats, features):
            self._handle_concept_drift(track_id, features, confidence)
            self._global_stats["total_drifts_detected"] += 1
            return stats.mean_features

        current_lr = self._compute_learning_rate(stats, confidence)

        updated_features = self._strategy.update(
            stats,
            features,
            confidence,
            current_lr
        )

        stats.add_sample(features, confidence)

        stats.quality_score = self._aggregator.compute_quality_score(stats)

        self._global_stats["total_updates"] += 1

        self.logger.debug(
            "Features actualizados",
            track_id=track_id,
            n_samples=stats.n_samples,
            confidence=confidence,
            learning_rate=current_lr
        )

        return updated_features

    def _handle_concept_drift(
        self,
        track_id: int,
        features: np.ndarray,
        confidence: float
    ) -> None:
        """
        Maneja un cambio de concepto (concept drift).

        Args:
            track_id: ID del track
            features: Nuevos features
            confidence: Confianza de la observación
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return

        stats.mean_features = features.copy()
        stats.n_samples = 1
        stats.total_updates += 1
        stats.feature_history.clear()
        stats.feature_history.append(features.copy())
        stats.confidence_history.clear()
        stats.confidence_history.append(confidence)
        stats.timestamps.clear()
        stats.timestamps.append(time.time())
        stats.concept_drift_detected = True
        stats.covariance = None
        stats.quality_score = 0.0

        self.logger.info(
            "Concept drift manejado",
            track_id=track_id,
            confidence=confidence
        )

    def _compute_learning_rate(
        self,
        stats: FeatureStatistics,
        confidence: float
    ) -> float:
        """
        Calcula la tasa de aprendizaje adaptativa.

        Args:
            stats: Estadísticas del track
            confidence: Confianza de la observación

        Returns:
            float: Tasa de aprendizaje calculada
        """
        base_lr = self.learning_rate

        sample_decay = max(0.01, 1.0 / (1.0 + 0.1 * stats.n_samples))
        decayed_lr = base_lr * sample_decay

        confidence_factor = 0.5 + 0.5 * confidence

        adjusted_lr = decayed_lr * confidence_factor

        return max(0.001, min(0.2, adjusted_lr))

    def get_feature(self, track_id: int) -> Optional[np.ndarray]:
        """
        Obtiene el feature promedio actual de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[np.ndarray]: Feature promedio o None
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return None
        return stats.mean_features

    def get_confidence(self, track_id: int, features: np.ndarray) -> float:
        """
        Calcula la confianza de un feature respecto al promedio del track.

        Args:
            track_id: ID del track
            features: Features a evaluar

        Returns:
            float: Confianza (0-1)
        """
        mean_feat = self.get_feature(track_id)
        if mean_feat is None:
            return 0.0

        norm_mean = np.linalg.norm(mean_feat)
        norm_feat = np.linalg.norm(features)

        if norm_mean < 1e-8 or norm_feat < 1e-8:
            return 0.0

        similarity = np.dot(mean_feat, features) / (norm_mean * norm_feat)
        return max(0.0, min(1.0, similarity))

    def get_stats(self, track_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema o de un track específico.

        Args:
            track_id: ID del track (opcional)

        Returns:
            Dict[str, Any]: Estadísticas del sistema
        """
        if track_id is not None:
            stats = self._stats.get(track_id)
            if stats is None:
                return {}
            return stats.to_dict()

        return {
            **self._global_stats,
            "active_tracks": len(self._stats),
            "total_samples": sum(s.n_samples for s in self._stats.values()),
            "total_updates": sum(s.total_updates for s in self._stats.values()),
            "avg_samples_per_track": sum(s.n_samples for s in self._stats.values()) / max(1, len(self._stats)),
            "drift_rate": self._drift_detector.get_drift_rate(),
            "strategy": self._strategy.name,
            "drift_detector": self._drift_detector.get_stats(),
            "aggregator": self._aggregator.get_stats(),
        }

    def get_covariance(self, track_id: int) -> Optional[np.ndarray]:
        """
        Obtiene la matriz de covarianza de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[np.ndarray]: Matriz de covarianza o None
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return None
        return stats.covariance

    def get_uncertainty(self, track_id: int) -> float:
        """
        Calcula la incertidumbre del modelo para un track.

        Args:
            track_id: ID del track

        Returns:
            float: Medida de incertidumbre (0-1)
        """
        stats = self._stats.get(track_id)
        if stats is None or stats.n_samples < self.min_samples:
            return 1.0

        if stats.covariance is not None:
            trace = np.trace(stats.covariance)
            max_trace = self.feature_dim * 0.1
            return min(1.0, trace / max_trace)

        return 1.0 / (1.0 + stats.n_samples / self.min_samples)

    def get_quality(self, track_id: int) -> float:
        """
        Obtiene la calidad del modelo para un track.

        Args:
            track_id: ID del track

        Returns:
            float: Calidad del modelo (0-1)
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return 0.0
        return stats.quality_score

    def merge_tracks(self, target_id: int, source_id: int) -> bool:
        """
        Fusiona las estadísticas de dos tracks.

        Args:
            target_id: ID del track destino
            source_id: ID del track origen

        Returns:
            bool: True si la fusión fue exitosa
        """
        if target_id not in self._stats or source_id not in self._stats:
            return False

        target_stats = self._stats[target_id]
        source_stats = self._stats[source_id]

        self._aggregator.merge_statistics(target_stats, source_stats)

        del self._stats[source_id]
        self._drift_detector.clear_track(source_id)
        self._global_stats["active_learners"] -= 1

        self.logger.info(
            "Tracks fusionados",
            target_id=target_id,
            source_id=source_id,
            total_samples=target_stats.n_samples
        )

        return True

    def clear_track(self, track_id: int) -> bool:
        """
        Elimina todas las estadísticas de un track.

        Args:
            track_id: ID del track a eliminar

        Returns:
            bool: True si se eliminó correctamente
        """
        if track_id in self._stats:
            del self._stats[track_id]
            self._drift_detector.clear_track(track_id)
            self._global_stats["active_learners"] -= 1
            self.logger.debug("Track eliminado del learner", track_id=track_id)
            return True
        return False

    def clear_all(self) -> None:
        """Limpia todas las estadísticas del sistema."""
        count = len(self._stats)
        self._stats.clear()
        self._drift_detector.clear_all()
        self._global_stats["active_learners"] = 0
        self._global_stats["total_tracks"] = 0
        self._global_stats["total_updates"] = 0
        self._global_stats["total_drifts_detected"] = 0
        self.logger.info("Todos los learners eliminados", count=count)

    def get_learning_curve(self, track_id: int) -> Dict[str, List[float]]:
        """
        Obtiene la curva de aprendizaje de un track.

        Args:
            track_id: ID del track

        Returns:
            Dict[str, List[float]]: Curva de aprendizaje
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return {}

        return {
            "confidences": list(stats.confidence_history),
            "timestamps": list(stats.timestamps),
            "sample_count": stats.n_samples,
            "total_updates": stats.total_updates,
            "quality": [stats.quality_score] if stats.quality_score > 0 else [],
        }

    def set_strategy(self, strategy_type: str) -> None:
        """
        Cambia la estrategia de aprendizaje.

        Args:
            strategy_type: Tipo de estrategia
        """
        try:
            self._strategy = LearningStrategyFactory.create(strategy_type)
            self._global_stats["strategy"] = strategy_type
            self.logger.info(f"Estrategia cambiada a: {strategy_type}")
        except ValueError as e:
            self.logger.error(f"Error cambiando estrategia: {e}")

    def reset(self) -> None:
        """Reinicia completamente el sistema de aprendizaje."""
        self.clear_all()
        self._global_stats["start_time"] = time.time()
        self._global_stats["total_drifts_detected"] = 0
        self._aggregator.reset()
        self.logger.info("Sistema de aprendizaje reiniciado")

    def __len__(self) -> int:
        """Retorna el número de tracks activos."""
        return len(self._stats)

    def __contains__(self, track_id: int) -> bool:
        """Verifica si un track existe en el sistema."""
        return track_id in self._stats
