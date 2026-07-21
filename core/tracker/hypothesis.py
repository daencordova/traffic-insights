"""
Sistema de seguimiento multi-hipótesis (MHT) para tracking robusto.

Este módulo implementa un sistema de hipótesis múltiples que permite mantener
varias posibles trayectorias para un mismo objeto, manejando eficazmente
occlusiones y ambigüedades en la asociación de datos.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple

import numpy as np

from core.constants import MAX_TRACK_HISTORY


class HypothesisStatus(Enum):
    """Estados posibles de una hipótesis en el sistema MHT."""
    ACTIVE = "active"
    PRUNED = "pruned"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class TrackHypothesis:
    """
    Representa una hipótesis de trayectoria para un objeto en seguimiento.

    Attributes:
        track_id: Identificador único del track al que pertenece la hipótesis.
        positions: Historial de posiciones de la hipótesis.
        features: Historial de features visuales asociados.
        confidence: Confianza actual de la hipótesis.
        probability: Probabilidad de que esta hipótesis sea la correcta.
        last_update: Timestamp de la última actualización.
        active: Indica si la hipótesis está activa.
        parent_id: ID de la hipótesis padre (para árbol de hipótesis).
        status: Estado actual de la hipótesis.
        creation_time: Timestamp de creación de la hipótesis.
        bbox_history: Historial de bounding boxes.
        velocity: Velocidad estimada de la hipótesis.
        acceleration: Aceleración estimada de la hipótesis.
    """
    __slots__ = (
        'track_id', 'positions', 'features', 'confidence', 'probability',
        'last_update', 'active', 'parent_id', 'status', 'creation_time',
        'bbox_history', 'velocity', 'acceleration', '_lock'
    )

    track_id: int
    positions: List[Tuple[int, int]] = field(default_factory=list)
    features: List[np.ndarray] = field(default_factory=list)
    confidence: float = 0.0
    probability: float = 0.5
    last_update: float = field(default_factory=time.time)
    active: bool = True
    parent_id: Optional[int] = None
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    creation_time: float = field(default_factory=time.time)
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_position(self, position: Tuple[int, int], bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        Actualiza la posición de la hipótesis con nueva observación.

        Args:
            position: Nueva posición (x, y) del objeto.
            bbox: Nuevo bounding box (x1, y1, x2, y2) opcional.
        """
        with self._lock:
            self.positions.append(position)
            if len(self.positions) > MAX_TRACK_HISTORY:
                self.positions = self.positions[-MAX_TRACK_HISTORY:]

            if bbox is not None:
                self.bbox_history.append(bbox)
                if len(self.bbox_history) > MAX_TRACK_HISTORY:
                    self.bbox_history = self.bbox_history[-MAX_TRACK_HISTORY:]

            self.last_update = time.time()

            if len(self.positions) >= 2:
                prev = self.positions[-2]
                curr = self.positions[-1]
                self.velocity = (curr[0] - prev[0], curr[1] - prev[1])

                if len(self.positions) >= 3:
                    prev_prev = self.positions[-3]
                    prev_vel = (prev[0] - prev_prev[0], prev[1] - prev_prev[1])
                    self.acceleration = (
                        self.velocity[0] - prev_vel[0],
                        self.velocity[1] - prev_vel[1],
                    )

    def add_feature(self, feature: np.ndarray) -> None:
        """
        Añade un feature visual a la hipótesis.

        Args:
            feature: Vector de features a añadir.
        """
        with self._lock:
            if feature is not None:
                self.features.append(feature.copy())
                if len(self.features) > 20:
                    self.features = self.features[-20:]

    def compute_bayesian_probability(
        self,
        observation: Dict[str, Any],
        similarity: float,
        spatial_distance: float,
        max_distance: float = 100.0,
        temporal_weight: float = 0.6,
        spatial_weight: float = 0.4
    ) -> float:
        """
        Actualiza la probabilidad usando filtro Bayesiano con normalización.

        Args:
            observation: Observación actual del objeto.
            similarity: Similitud coseno entre features (0-1).
            spatial_distance: Distancia espacial entre observación y predicción.
            max_distance: Distancia máxima considerada para matching.
            temporal_weight: Peso para la similitud temporal.
            spatial_weight: Peso para la similitud espacial.

        Returns:
            float: Probabilidad actualizada de la hipótesis (entre 0 y 1).
        """
        with self._lock:
            spatial_score = 1.0 - min(1.0, spatial_distance / max_distance)
            likelihood = (temporal_weight * max(0.0, similarity)) + (spatial_weight * spatial_score)

            det_confidence = observation.get('confidence', 0.5)
            confidence_factor = 0.7 + 0.3 * det_confidence
            likelihood *= confidence_factor

            likelihood = max(0.0, min(1.0, likelihood))

            learning_rate = 0.3
            prior = self.probability
            posterior = prior * likelihood

            self.probability = (1.0 - learning_rate) * prior + learning_rate * posterior

            self.probability = max(0.01, min(1.0, self.probability))

            return self.probability

    def get_average_feature(self) -> Optional[np.ndarray]:
        """
        Obtiene el feature promedio de la hipótesis.

        Returns:
            Optional[np.ndarray]: Feature promedio o None si no hay features.
        """
        with self._lock:
            if not self.features:
                return None

            avg_feature = np.mean(self.features, axis=0)
            norm = np.linalg.norm(avg_feature)
            if norm > 0:
                avg_feature = avg_feature / norm
            return avg_feature

    def get_recent_velocity(self, num_samples: int = 5) -> Tuple[float, float]:
        """
        Calcula la velocidad promedio reciente.

        Args:
            num_samples: Número de posiciones a considerar.

        Returns:
            Tuple[float, float]: Velocidad promedio en x e y.
        """
        with self._lock:
            if len(self.positions) < 2:
                return (0.0, 0.0)

            start_idx = max(0, len(self.positions) - num_samples - 1)
            positions = self.positions[start_idx:]

            if len(positions) < 2:
                return self.velocity

            total_dx = positions[-1][0] - positions[0][0]
            total_dy = positions[-1][1] - positions[0][1]
            num_steps = len(positions) - 1

            if num_steps == 0:
                return (0.0, 0.0)

            return (total_dx / num_steps, total_dy / num_steps)

    def is_expired(self, max_age_seconds: float = 30.0) -> bool:
        """
        Verifica si la hipótesis ha expirado por antigüedad.

        Args:
            max_age_seconds: Tiempo máximo de vida en segundos.

        Returns:
            bool: True si la hipótesis ha expirado.
        """
        return time.time() - self.last_update > max_age_seconds

    def get_position_at_time(self, t: float) -> Optional[Tuple[float, float]]:
        """
        Interpola la posición en un momento específico.

        Args:
            t: Timestamp para el cual se desea la posición.

        Returns:
            Optional[Tuple[float, float]]: Posición interpolada o None.
        """
        with self._lock:
            if len(self.positions) < 2:
                return None

            times = [self.creation_time + i * 0.1 for i in range(len(self.positions))]
            if t < times[0] or t > times[-1]:
                return None

            for i in range(len(times) - 1):
                if times[i] <= t <= times[i + 1]:
                    alpha = (t - times[i]) / (times[i + 1] - times[i] + 1e-8)
                    x = self.positions[i][0] + alpha * (self.positions[i + 1][0] - self.positions[i][0])
                    y = self.positions[i][1] + alpha * (self.positions[i + 1][1] - self.positions[i][1])
                    return (x, y)

            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la hipótesis a diccionario para serialización.

        Returns:
            Dict[str, Any]: Representación en diccionario de la hipótesis.
        """
        return {
            "track_id": self.track_id,
            "positions": self.positions[-10:],
            "confidence": self.confidence,
            "probability": self.probability,
            "last_update": self.last_update,
            "active": self.active,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "position_count": len(self.positions),
            "feature_count": len(self.features),
            "age_seconds": time.time() - self.creation_time,
        }


class HypothesisTree:
    """
    Árbol de hipótesis para el sistema MHT.

    Gestiona múltiples hipótesis por track, manteniendo las más probables
    y podando las de baja probabilidad.

    Attributes:
        max_depth: Profundidad máxima del árbol de hipótesis.
        pruning_threshold: Umbral de probabilidad para podar hipótesis.
        max_hypotheses_per_track: Número máximo de hipótesis por track.
        _hypotheses: Diccionario de hipótesis agrupadas por track_id.
        _active_hyps: Conjunto de IDs de hipótesis activas.
        _lock: Lock para operaciones thread-safe.
        _stats: Estadísticas del sistema MHT.
    """

    def __init__(
        self,
        max_depth: int = 10,
        pruning_threshold: float = 0.05,
        max_hypotheses_per_track: int = 5
    ) -> None:
        """
        Inicializa el árbol de hipótesis.

        Args:
            max_depth: Profundidad máxima del árbol.
            pruning_threshold: Umbral de probabilidad para poda.
            max_hypotheses_per_track: Máximo de hipótesis por track.
        """
        self.max_depth = max_depth
        self.pruning_threshold = pruning_threshold
        self.max_hypotheses_per_track = max_hypotheses_per_track

        self._hypotheses: Dict[int, List[TrackHypothesis]] = {}
        self._active_hyps: Set[int] = set()
        self._lock = threading.Lock()

        self._stats = {
            "total_hypotheses_created": 0,
            "total_hypotheses_pruned": 0,
            "total_hypotheses_confirmed": 0,
            "total_hypotheses_rejected": 0,
            "avg_hypotheses_per_track": 0.0,
            "track_distribution": {},
            "last_prune_time": time.time(),
            "total_prunes": 0,
        }

    def add_hypothesis(
        self,
        track_id: int,
        hypothesis: TrackHypothesis,
        parent_id: Optional[int] = None
    ) -> None:
        """
        Añade una nueva hipótesis al árbol.

        Args:
            track_id: ID del track al que pertenece la hipótesis.
            hypothesis: Hipótesis a añadir.
            parent_id: ID de la hipótesis padre (opcional).
        """
        with self._lock:
            if track_id not in self._hypotheses:
                self._hypotheses[track_id] = []

            hypothesis.parent_id = parent_id

            if hypothesis.probability > 0.5:
                hypothesis.probability = 0.3

            self._hypotheses[track_id].append(hypothesis)
            self._active_hyps.add(track_id)
            self._stats["total_hypotheses_created"] += 1

            self._normalize_probabilities(track_id)
            self._prune_track(track_id)

    def get_best_hypothesis(self, track_id: int) -> Optional[TrackHypothesis]:
        """
        Obtiene la mejor hipótesis para un track según probabilidad.

        Args:
            track_id: ID del track.

        Returns:
            Optional[TrackHypothesis]: Mejor hipótesis o None si no existe.
        """
        with self._lock:
            hyps = self._hypotheses.get(track_id, [])
            active_hyps = [h for h in hyps if h.active and h.status == HypothesisStatus.ACTIVE]

            if not active_hyps:
                return None

            return max(active_hyps, key=lambda h: h.probability)

    def get_top_k_hypotheses(self, track_id: int, k: int = 3) -> List[TrackHypothesis]:
        """
        Obtiene las k mejores hipótesis para un track.

        Args:
            track_id: ID del track.
            k: Número de hipótesis a retornar.

        Returns:
            List[TrackHypothesis]: Lista de hipótesis ordenadas por probabilidad.
        """
        with self._lock:
            hyps = self._hypotheses.get(track_id, [])
            active_hyps = [h for h in hyps if h.active and h.status == HypothesisStatus.ACTIVE]

            if not active_hyps:
                return []

            sorted_hyps = sorted(active_hyps, key=lambda h: h.probability, reverse=True)
            return sorted_hyps[:k]

    def _normalize_probabilities(self, track_id: int) -> None:
        """
        Normaliza las probabilidades de todas las hipótesis de un track.

        Args:
            track_id: ID del track.
        """
        hyps = self._hypotheses.get(track_id, [])
        active_hyps = [h for h in hyps if h.active and h.status == HypothesisStatus.ACTIVE]

        if not active_hyps:
            return

        total_prob = sum(h.probability for h in active_hyps)

        if total_prob > 0:
            for hyp in active_hyps:
                hyp.probability /= total_prob
        else:
            prob = 1.0 / len(active_hyps)
            for hyp in active_hyps:
                hyp.probability = prob

    def update_hypothesis_probabilities(
        self,
        track_id: int,
        observation: Dict[str, Any],
        similarities: Dict[int, float],
        spatial_distances: Dict[int, float]
    ) -> None:
        """
        Actualiza las probabilidades de todas las hipótesis de un track.

        Args:
            track_id: ID del track.
            observation: Observación actual.
            similarities: Diccionario de similitudes por ID de hipótesis.
            spatial_distances: Diccionario de distancias espaciales por ID de hipótesis.
        """
        with self._lock:
            hyps = self._hypotheses.get(track_id, [])
            active_hyps = [h for h in hyps if h.active and h.status == HypothesisStatus.ACTIVE]

            if not active_hyps:
                return

            for hyp in active_hyps:
                hyp_id = id(hyp)
                similarity = similarities.get(hyp_id, 0.3)
                spatial_dist = spatial_distances.get(hyp_id, 50.0)

                hyp.compute_bayesian_probability(
                    observation,
                    similarity,
                    spatial_dist
                )

            self._normalize_probabilities(track_id)

    def confirm_hypothesis(self, track_id: int, hypothesis_id: int) -> bool:
        """
        Confirma una hipótesis como la correcta para un track.

        Args:
            track_id: ID del track.
            hypothesis_id: ID de la hipótesis a confirmar.

        Returns:
            bool: True si la hipótesis fue confirmada exitosamente.
        """
        with self._lock:
            hyps = self._hypotheses.get(track_id, [])

            for hyp in hyps:
                if id(hyp) == hypothesis_id:
                    hyp.status = HypothesisStatus.CONFIRMED
                    hyp.probability = 1.0
                    self._stats["total_hypotheses_confirmed"] += 1
                    self._reject_other_hypotheses(track_id, hypothesis_id)
                    return True

            return False

    def _reject_other_hypotheses(self, track_id: int, confirmed_id: int) -> None:
        """
        Rechaza todas las hipótesis excepto la confirmada.

        Args:
            track_id: ID del track.
            confirmed_id: ID de la hipótesis confirmada.
        """
        hyps = self._hypotheses.get(track_id, [])
        for hyp in hyps:
            if id(hyp) != confirmed_id and hyp.active:
                hyp.status = HypothesisStatus.REJECTED
                hyp.active = False
                hyp.probability = 0.0
                self._stats["total_hypotheses_rejected"] += 1

    def _prune_track(self, track_id: int) -> None:
        """
        Poda las hipótesis de baja probabilidad para un track específico.

        Args:
            track_id: ID del track a podar.
        """
        hyps = self._hypotheses.get(track_id, [])

        if not hyps:
            return

        self._normalize_probabilities(track_id)

        hyps = [h for h in hyps if h.probability > self.pruning_threshold]

        if len(hyps) > self.max_hypotheses_per_track:
            sorted_hyps = sorted(hyps, key=lambda h: h.probability, reverse=True)
            to_keep = sorted_hyps[:self.max_hypotheses_per_track]

            for hyp in hyps:
                if hyp not in to_keep:
                    hyp.status = HypothesisStatus.PRUNED
                    hyp.active = False
                    self._stats["total_hypotheses_pruned"] += 1

            self._hypotheses[track_id] = to_keep
        else:
            self._hypotheses[track_id] = hyps

        active_hyps = [h for h in self._hypotheses.get(track_id, []) if h.active]
        if not active_hyps:
            del self._hypotheses[track_id]
            self._active_hyps.discard(track_id)

        self._stats["total_prunes"] += 1
        self._stats["last_prune_time"] = time.time()

    def prune_all(self) -> int:
        """
        Poda todas las hipótesis en el árbol.

        Returns:
            int: Número total de hipótesis podadas.
        """
        with self._lock:
            pruned_count = 0

            for track_id in list(self._hypotheses.keys()):
                before = len(self._hypotheses[track_id])
                self._prune_track(track_id)
                after = len(self._hypotheses.get(track_id, []))
                pruned_count += (before - after)

            total_hyps = sum(len(hyps) for hyps in self._hypotheses.values())
            num_tracks = len(self._hypotheses)

            self._stats["avg_hypotheses_per_track"] = (
                total_hyps / num_tracks if num_tracks > 0 else 0.0
            )
            self._stats["track_distribution"] = {
                str(tid): len(hyps) for tid, hyps in self._hypotheses.items()
            }

            return pruned_count

    def get_most_likely_positions(self, track_id: int, horizon: int = 10) -> List[Tuple[int, int]]:
        """
        Obtiene las posiciones más probables futuras para un track.

        Args:
            track_id: ID del track.
            horizon: Número de pasos a predecir.

        Returns:
            List[Tuple[int, int]]: Posiciones predichas más probables.
        """
        best_hyp = self.get_best_hypothesis(track_id)
        if best_hyp is None or not best_hyp.positions:
            return []

        last_pos = best_hyp.positions[-1]
        velocity = best_hyp.get_recent_velocity()

        predictions = []
        for i in range(1, horizon + 1):
            pred_x = int(last_pos[0] + velocity[0] * i)
            pred_y = int(last_pos[1] + velocity[1] * i)
            predictions.append((pred_x, pred_y))

        return predictions

    def get_track_hypotheses_history(self, track_id: int) -> List[Dict[str, Any]]:
        """
        Obtiene el historial completo de hipótesis para un track.

        Args:
            track_id: ID del track.

        Returns:
            List[Dict[str, Any]]: Historial de hipótesis en formato diccionario.
        """
        with self._lock:
            hyps = self._hypotheses.get(track_id, [])
            return [hyp.to_dict() for hyp in hyps]

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema MHT.

        Returns:
            Dict[str, Any]: Estadísticas del árbol de hipótesis.
        """
        with self._lock:
            total_hyps = sum(len(hyps) for hyps in self._hypotheses.values())
            active_hyps = sum(
                1 for hyps in self._hypotheses.values()
                for h in hyps if h.active and h.status == HypothesisStatus.ACTIVE
            )

            return {
                **self._stats,
                "active_tracks": len(self._active_hyps),
                "total_tracks_in_tree": len(self._hypotheses),
                "total_hypotheses": total_hyps,
                "active_hypotheses": active_hyps,
                "avg_hypotheses_per_track": (
                    total_hyps / len(self._hypotheses) if self._hypotheses else 0.0
                ),
                "current_prune_threshold": self.pruning_threshold,
                "max_depth": self.max_depth,
                "memory_usage_estimate": total_hyps * 1024,
                "hypothesis_status_distribution": {
                    status.value: sum(
                        1 for h_list in self._hypotheses.values()
                        for h in h_list if h.status == status
                    )
                    for status in HypothesisStatus
                }
            }

    def clear(self) -> None:
        """Limpia todas las hipótesis del árbol."""
        with self._lock:
            self._hypotheses.clear()
            self._active_hyps.clear()
            self._stats = {
                **self._stats,
                "total_hypotheses_created": 0,
                "total_hypotheses_pruned": 0,
                "total_hypotheses_confirmed": 0,
                "total_hypotheses_rejected": 0,
            }

    def __len__(self) -> int:
        """
        Retorna el número total de hipótesis activas.

        Returns:
            int: Número total de hipótesis.
        """
        with self._lock:
            return sum(len(hyps) for hyps in self._hypotheses.values())
