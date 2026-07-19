"""
Sistema de aprendizaje en línea para adaptación de features en tracking.

Este módulo implementa un sistema de aprendizaje incremental que permite
actualizar los features de los tracks en tiempo real, adaptándose a
cambios de apariencia y condiciones de iluminación.
"""

import time
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from utils.logger import LoggerMixin


class LearningStrategy(Enum):
    """
    Estrategias de aprendizaje disponibles.
    """
    INCREMENTAL = "incremental"
    ADAPTIVE = "adaptive"
    BATCH = "batch"
    HYBRID = "hybrid"


@dataclass
class FeatureStatistics:
    """
    Estadísticas de features para aprendizaje en línea.

    Attributes:
        mean_features: Vector de features promedio.
        covariance: Matriz de covarianza incremental.
        n_samples: Número total de muestras procesadas.
        feature_history: Historial de features recientes.
        timestamps: Timestamps de las actualizaciones.
        confidence_history: Historial de confianzas.
        last_update_time: Timestamp de la última actualización.
        total_updates: Número total de actualizaciones.
        concept_drift_detected: Flag de detección de cambio de concepto.
    """
    mean_features: np.ndarray
    covariance: Optional[np.ndarray] = None
    n_samples: int = 0
    feature_history: deque = field(default_factory=lambda: deque(maxlen=50))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_update_time: float = field(default_factory=time.time)
    total_updates: int = 0
    concept_drift_detected: bool = False


class OnlineFeatureLearner(LoggerMixin):
    """
    Sistema de aprendizaje en línea para features de tracking.

    Este sistema permite actualizar incrementalmente los features
    de los tracks, adaptándose a cambios de apariencia y condiciones
    ambientales.

    Características:
    - Actualización incremental de la media con factor de aprendizaje
    - Covarianza para medir incertidumbre
    - Detección de cambios de apariencia (concept drift)
    - Estrategias de aprendizaje configurables
    - Estadísticas de rendimiento y métricas
    - Historial de actualizaciones para análisis

    Attributes:
        feature_dim: Dimensión de los features.
        learning_rate: Tasa de aprendizaje base.
        min_samples: Mínimo de muestras para estadísticas robustas.
        drift_threshold: Umbral para detección de cambio de concepto.
        max_history: Tamaño máximo del historial.
        strategy: Estrategia de aprendizaje activa.
        _stats: Diccionario de estadísticas por track_id.
        _global_stats: Estadísticas globales del sistema.
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
            feature_dim: Dimensión de los features (por defecto 2048).
            learning_rate: Tasa de aprendizaje base (0.01-0.2).
            min_samples: Mínimo de muestras para estadísticas robustas.
            drift_threshold: Umbral para detección de cambio de concepto (0.2-0.5).
            max_history: Tamaño máximo del historial por track.
            strategy: Estrategia de aprendizaje ('incremental', 'adaptive', 'batch', 'hybrid').
        """
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.drift_threshold = drift_threshold
        self.max_history = max_history

        if isinstance(strategy, str):
            self.strategy = LearningStrategy(strategy)
        else:
            self.strategy = strategy

        self._stats: Dict[int, FeatureStatistics] = {}
        self._global_stats: Dict[str, Any] = {
            "total_tracks": 0,
            "total_updates": 0,
            "total_drifts_detected": 0,
            "avg_learning_rate": learning_rate,
            "active_learners": 0,
            "memory_usage_mb": 0.0,
            "start_time": time.time(),
        }

        self._lock = None

        self.logger.info(
            "OnlineFeatureLearner inicializado",
            feature_dim=feature_dim,
            learning_rate=learning_rate,
            min_samples=min_samples,
            drift_threshold=drift_threshold,
            strategy=self.strategy.value
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

        Este método implementa el núcleo del aprendizaje en línea,
        actualizando la media de features y detectando cambios de
        apariencia.

        Args:
            track_id: ID del track a actualizar.
            features: Nuevo vector de features.
            confidence: Confianza de la observación (0-1).
            force: Forzar actualización incluso si baja confianza.

        Returns:
            np.ndarray: Features actualizados (promedio).
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
            self._stats[track_id].confidence_history.append(confidence)
            self._stats[track_id].timestamps.append(time.time())
            self._stats[track_id].feature_history.append(features.copy())
            self._stats[track_id].total_updates = 1
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

        if self._detect_appearance_change(stats, features):
            self._handle_concept_drift(track_id, features, confidence)
            return stats.mean_features

        current_lr = self._compute_learning_rate(stats, confidence)

        if self.strategy == LearningStrategy.INCREMENTAL:
            updated_features = self._update_incremental(stats, features, current_lr)
        elif self.strategy == LearningStrategy.ADAPTIVE:
            updated_features = self._update_adaptive(stats, features, current_lr, confidence)
        elif self.strategy == LearningStrategy.BATCH:
            updated_features = self._update_batch(stats, features)
        else:
            updated_features = self._update_hybrid(stats, features, current_lr, confidence)

        stats.n_samples += 1
        stats.total_updates += 1
        stats.last_update_time = time.time()
        stats.confidence_history.append(confidence)
        stats.timestamps.append(time.time())
        stats.feature_history.append(features.copy())

        if stats.n_samples > self.min_samples:
            self._update_covariance(stats, features)

        self._global_stats["total_updates"] += 1

        self.logger.debug(
            "Features actualizados",
            track_id=track_id,
            n_samples=stats.n_samples,
            confidence=confidence,
            learning_rate=current_lr
        )

        return updated_features

    def _update_incremental(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        learning_rate: float
    ) -> np.ndarray:
        """
        Actualización incremental simple de la media.

        Args:
            stats: Estadísticas del track.
            features: Nuevos features.
            learning_rate: Tasa de aprendizaje.

        Returns:
            np.ndarray: Features actualizados.
        """
        alpha = min(learning_rate, 1.0 / (stats.n_samples + 1))
        stats.mean_features = (1 - alpha) * stats.mean_features + alpha * features
        return stats.mean_features

    def _update_adaptive(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        learning_rate: float,
        confidence: float
    ) -> np.ndarray:
        """
        Actualización adaptativa con factor de confianza.

        Args:
            stats: Estadísticas del track.
            features: Nuevos features.
            learning_rate: Tasa de aprendizaje base.
            confidence: Confianza de la observación.

        Returns:
            np.ndarray: Features actualizados.
        """
        confidence_factor = 0.5 + 0.5 * confidence
        adjusted_lr = learning_rate * confidence_factor

        sample_factor = min(1.0, self.min_samples / max(1, stats.n_samples))
        final_lr = adjusted_lr * (0.7 + 0.3 * sample_factor)

        alpha = min(final_lr, 0.3)
        stats.mean_features = (1 - alpha) * stats.mean_features + alpha * features

        return stats.mean_features

    def _update_batch(
        self,
        stats: FeatureStatistics,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Actualización por lotes usando historial completo.

        Args:
            stats: Estadísticas del track.
            features: Nuevos features.

        Returns:
            np.ndarray: Features actualizados.
        """
        history = list(stats.feature_history)
        if len(history) < self.min_samples:
            history.append(features)

        if history:
            batch_mean = np.mean(history, axis=0)
            norm = np.linalg.norm(batch_mean)
            if norm > 0:
                batch_mean = batch_mean / norm

            alpha = 0.3 if len(history) > self.min_samples else 0.1
            stats.mean_features = (1 - alpha) * stats.mean_features + alpha * batch_mean

        return stats.mean_features

    def _update_hybrid(
        self,
        stats: FeatureStatistics,
        features: np.ndarray,
        learning_rate: float,
        confidence: float
    ) -> np.ndarray:
        """
        Actualización híbrida: incremental + batch periódico.

        Args:
            stats: Estadísticas del track.
            features: Nuevos features.
            learning_rate: Tasa de aprendizaje.
            confidence: Confianza de la observación.

        Returns:
            np.ndarray: Features actualizados.
        """
        alpha = min(learning_rate, 1.0 / (stats.n_samples + 1))
        stats.mean_features = (1 - alpha) * stats.mean_features + alpha * features

        if stats.total_updates % 10 == 0 and stats.n_samples > self.min_samples:
            history = list(stats.feature_history)
            if len(history) >= self.min_samples:
                batch_mean = np.mean(history, axis=0)
                norm = np.linalg.norm(batch_mean)
                if norm > 0:
                    batch_mean = batch_mean / norm
                    stats.mean_features = 0.7 * stats.mean_features + 0.3 * batch_mean

        return stats.mean_features

    def _detect_appearance_change(
        self,
        stats: FeatureStatistics,
        features: np.ndarray
    ) -> bool:
        """
        Detecta si hay un cambio significativo en la apariencia.

        Args:
            stats: Estadísticas del track.
            features: Nuevos features.

        Returns:
            bool: True si se detectó cambio de apariencia.
        """
        if stats.n_samples < self.min_samples:
            return False

        norm_mean = np.linalg.norm(stats.mean_features)
        norm_feat = np.linalg.norm(features)

        if norm_mean < 1e-8 or norm_feat < 1e-8:
            return False

        similarity = np.dot(stats.mean_features, features) / (norm_mean * norm_feat)
        similarity = max(0.0, min(1.0, similarity))

        if similarity < self.drift_threshold and stats.n_samples > self.min_samples * 2:
            self.logger.debug(
                "Cambio de apariencia detectado",
                similarity=similarity,
                threshold=self.drift_threshold,
                n_samples=stats.n_samples
            )
            return True

        return False

    def _handle_concept_drift(
        self,
        track_id: int,
        features: np.ndarray,
        confidence: float
    ) -> None:
        """
        Maneja un cambio de concepto (concept drift).

        Args:
            track_id: ID del track.
            features: Nuevos features.
            confidence: Confianza de la observación.
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

        self._global_stats["total_drifts_detected"] += 1

        self.logger.info(
            "Concept drift manejado",
            track_id=track_id,
            confidence=confidence
        )

    def _update_covariance(
        self,
        stats: FeatureStatistics,
        features: np.ndarray
    ) -> None:
        """
        Actualiza la matriz de covarianza incrementalmente.

        Args:
            stats: Estadísticas del track.
            features: Nuevos features.
        """
        if stats.covariance is None:
            stats.covariance = np.eye(self.feature_dim) * 0.01

        diff = features - stats.mean_features
        if stats.n_samples > 1:
            update = np.outer(diff, diff) / (stats.n_samples - 1)
            stats.covariance = (stats.covariance * (stats.n_samples - 2) + update) / (stats.n_samples - 1)
        else:
            stats.covariance = np.eye(self.feature_dim) * 0.01

    def _compute_learning_rate(
        self,
        stats: FeatureStatistics,
        confidence: float
    ) -> float:
        """
        Calcula la tasa de aprendizaje adaptativa.

        Args:
            stats: Estadísticas del track.
            confidence: Confianza de la observación.

        Returns:
            float: Tasa de aprendizaje calculada.
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
            track_id: ID del track.

        Returns:
            Optional[np.ndarray]: Feature promedio o None.
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return None
        return stats.mean_features

    def get_confidence(self, track_id: int, features: np.ndarray) -> float:
        """
        Calcula la confianza de un feature respecto al promedio del track.

        Args:
            track_id: ID del track.
            features: Features a evaluar.

        Returns:
            float: Confianza (0-1).
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
            track_id: ID del track (opcional).

        Returns:
            Dict[str, Any]: Estadísticas del sistema.
        """
        if track_id is not None:
            stats = self._stats.get(track_id)
            if stats is None:
                return {}
            return {
                "n_samples": stats.n_samples,
                "total_updates": stats.total_updates,
                "confidence_mean": np.mean(stats.confidence_history) if stats.confidence_history else 0.0,
                "concept_drift_detected": stats.concept_drift_detected,
                "last_update_time": stats.last_update_time,
                "history_size": len(stats.feature_history),
                "covariance_trace": np.trace(stats.covariance) if stats.covariance is not None else 0.0,
            }

        return {
            **self._global_stats,
            "active_tracks": len(self._stats),
            "total_samples": sum(s.n_samples for s in self._stats.values()),
            "total_updates": sum(s.total_updates for s in self._stats.values()),
            "avg_samples_per_track": sum(s.n_samples for s in self._stats.values()) / max(1, len(self._stats)),
            "drift_rate": self._global_stats["total_drifts_detected"] / max(1, self._global_stats["total_updates"]),
            "strategy": self.strategy.value,
        }

    def get_covariance(self, track_id: int) -> Optional[np.ndarray]:
        """
        Obtiene la matriz de covarianza de un track.

        Args:
            track_id: ID del track.

        Returns:
            Optional[np.ndarray]: Matriz de covarianza o None.
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return None
        return stats.covariance

    def get_uncertainty(self, track_id: int) -> float:
        """
        Calcula la incertidumbre del modelo para un track.

        Args:
            track_id: ID del track.

        Returns:
            float: Medida de incertidumbre (0-1).
        """
        stats = self._stats.get(track_id)
        if stats is None or stats.n_samples < self.min_samples:
            return 1.0

        if stats.covariance is not None:
            trace = np.trace(stats.covariance)
            max_trace = self.feature_dim * 0.1
            return min(1.0, trace / max_trace)

        return 1.0 / (1.0 + stats.n_samples / self.min_samples)

    def merge_tracks(self, target_id: int, source_id: int) -> bool:
        """
        Fusiona las estadísticas de dos tracks.

        Args:
            target_id: ID del track destino.
            source_id: ID del track origen.

        Returns:
            bool: True si la fusión fue exitosa.
        """
        if target_id not in self._stats or source_id not in self._stats:
            return False

        target_stats = self._stats[target_id]
        source_stats = self._stats[source_id]

        total_samples = target_stats.n_samples + source_stats.n_samples
        if total_samples > 0:
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

        del self._stats[source_id]
        self._global_stats["active_learners"] -= 1

        self.logger.info(
            "Tracks fusionados",
            target_id=target_id,
            source_id=source_id,
            total_samples=total_samples
        )

        return True

    def clear_track(self, track_id: int) -> bool:
        """
        Elimina todas las estadísticas de un track.

        Args:
            track_id: ID del track a eliminar.

        Returns:
            bool: True si se eliminó correctamente.
        """
        if track_id in self._stats:
            del self._stats[track_id]
            self._global_stats["active_learners"] -= 1
            self.logger.debug("Track eliminado del learner", track_id=track_id)
            return True
        return False

    def clear_all(self) -> None:
        """
        Limpia todas las estadísticas del sistema.
        """
        count = len(self._stats)
        self._stats.clear()
        self._global_stats["active_learners"] = 0
        self._global_stats["total_tracks"] = 0
        self._global_stats["total_updates"] = 0
        self._global_stats["total_drifts_detected"] = 0
        self.logger.info("Todos los learners eliminados", count=count)

    def get_learning_curve(self, track_id: int) -> Dict[str, List[float]]:
        """
        Obtiene la curva de aprendizaje de un track.

        Args:
            track_id: ID del track.

        Returns:
            Dict[str, List[float]]: Curva de aprendizaje con confianzas y timestamps.
        """
        stats = self._stats.get(track_id)
        if stats is None:
            return {}

        return {
            "confidences": list(stats.confidence_history),
            "timestamps": list(stats.timestamps),
            "sample_count": stats.n_samples,
            "total_updates": stats.total_updates,
        }

    def get_model_quality(self, track_id: int) -> float:
        """
        Evalúa la calidad del modelo para un track.

        Args:
            track_id: ID del track.

        Returns:
            float: Calidad del modelo (0-1).
        """
        stats = self._stats.get(track_id)
        if stats is None or stats.n_samples < self.min_samples:
            return 0.0

        sample_score = min(1.0, stats.n_samples / (self.min_samples * 2))
        confidence_mean = np.mean(stats.confidence_history) if stats.confidence_history else 0.0
        drift_score = 0.0 if stats.concept_drift_detected else 1.0

        if len(stats.feature_history) > self.min_samples:
            history = list(stats.feature_history)
            similarities = []
            for i in range(1, len(history)):
                sim = np.dot(history[i-1], history[i])
                similarities.append(max(0.0, sim))
            stability = np.mean(similarities) if similarities else 0.5
        else:
            stability = 0.5

        quality = (
            0.30 * sample_score +
            0.25 * confidence_mean +
            0.25 * stability +
            0.20 * drift_score
        )

        return max(0.0, min(1.0, quality))

    def reset(self) -> None:
        """
        Reinicia completamente el sistema de aprendizaje.
        """
        self.clear_all()
        self._global_stats["start_time"] = time.time()
        self.logger.info("Sistema de aprendizaje reiniciado")

    def __len__(self) -> int:
        """
        Retorna el número de tracks activos.

        Returns:
            int: Número de tracks activos.
        """
        return len(self._stats)

    def __contains__(self, track_id: int) -> bool:
        """
        Verifica si un track existe en el sistema.

        Args:
            track_id: ID del track.

        Returns:
            bool: True si el track existe.
        """
        return track_id in self._stats
