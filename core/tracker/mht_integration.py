"""
Integración del sistema MHT (Multi-Hypothesis Tracking) con el tracker principal.

Este módulo proporciona la integración del sistema de hipótesis múltiples
con el tracker existente, añadiendo capacidades avanzadas de seguimiento.

El sistema MHT permite:
- Mantener múltiples hipótesis de trayectoria por objeto
- Manejar eficazmente occlusiones y ambigüedades
- Recuperar objetos perdidos mediante re-identificación
- Mejorar la robustez en escenarios complejos
- Reducir falsos positivos mediante poda de hipótesis

El sistema MHT es especialmente útil en:
- Escenas con múltiples objetos similares
- Occlusiones frecuentes
- Movimientos erráticos o impredecibles
- Condiciones de iluminación variables
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import LoggerMixin
from models.track_state import TrackState
from core.tracker.hypothesis import (
    TrackHypothesis,
    HypothesisTree,
    HypothesisStatus
)
from core.constants import (
    MHT_MAX_DEPTH,
    MHT_PRUNING_THRESHOLD,
    MHT_MAX_HYPOTHESES,
)

class MHTIntegration(LoggerMixin):
    """
    Integración del sistema MHT con el tracker principal.

    Esta clase orquesta el sistema de hipótesis múltiples y lo integra
    con el flujo principal de tracking, proporcionando capacidades
    avanzadas de seguimiento.

    Características:
        - Mantenimiento de árbol de hipótesis por track
        - Poda automática de hipótesis de baja probabilidad
        - Recuperación de tracks perdidos
        - Predicción de posiciones usando MHT
        - Estadísticas de rendimiento

    Attributes:
        hypothesis_tree: Árbol de hipótesis para MHT.
        enable_mht: Si el sistema MHT está habilitado.
        _track_hypothesis_map: Mapeo de tracks a sus hipótesis.
        _confirmed_hypotheses: Hipótesis confirmadas por track.
        _recently_recovered: Tracks recuperados recientemente.
        _recovery_cooldown: Cooldown para recuperación.

    Example:
        >>> mht = MHTIntegration(
        ...     max_depth=10,
        ...     pruning_threshold=0.01,
        ...     max_hypotheses_per_track=5
        ... )
        >>> mht.update_with_observations(tracks, detections, matches, unmatched)
        >>> predictions = mht.get_track_predictions(track_id, horizon=10)
        >>> confidence = mht.get_hypothesis_confidence(track_id)
    """

    def __init__(
        self,
        max_depth: int = MHT_MAX_DEPTH,
        pruning_threshold: float = MHT_PRUNING_THRESHOLD,
        max_hypotheses_per_track: int = MHT_MAX_HYPOTHESES,
        enable_mht: bool = True
    ) -> None:
        """
        Inicializa la integración MHT.

        Args:
            max_depth: Profundidad máxima del árbol de hipótesis.
            pruning_threshold: Umbral de probabilidad para poda.
            max_hypotheses_per_track: Máximo de hipótesis por track.
            enable_mht: Si el sistema MHT está habilitado.

        Note:
            Un mayor max_depth permite más hipótesis pero consume más memoria.
            pruning_threshold controla la agresividad de la poda.
        """
        self.hypothesis_tree = HypothesisTree(
            max_depth=max_depth,
            pruning_threshold=pruning_threshold,
            max_hypotheses_per_track=max_hypotheses_per_track
        )
        self.enable_mht = enable_mht

        self._track_hypothesis_map: Dict[int, List[int]] = {}
        self._confirmed_hypotheses: Dict[int, int] = {}
        self._recently_recovered: Dict[int, float] = {}
        self._recovery_cooldown: float = 3.0
        self._lock = threading.Lock()

        self._stats = {
            "mht_enabled": enable_mht,
            "total_hypotheses_integrated": 0,
            "total_tracks_with_hyps": 0,
            "confirmed_tracks": 0,
            "tracks_recovered_by_mht": 0,
            "last_recovery_time": None,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "false_positives_filtered": 0,
        }

        self.logger.info(
            "MHTIntegration inicializado",
            enabled=enable_mht,
            max_depth=max_depth,
            pruning_threshold=pruning_threshold
        )

    def create_hypothesis_from_track(
        self,
        track: TrackState,
        observation: Dict[str, Any],
        confidence: float = 0.5
    ) -> TrackHypothesis:
        """
        Crea una nueva hipótesis a partir de un track existente.

        Args:
            track: Track existente.
            observation: Observación actual.
            confidence: Confianza de la hipótesis.

        Returns:
            TrackHypothesis: Nueva hipótesis creada.

        Note:
            La hipótesis hereda el historial y features del track.
        """
        hypothesis = TrackHypothesis(
            track_id=track.track_id,
            confidence=confidence,
            probability=0.3,
            last_update=time.time(),
            active=True,
            parent_id=track.track_id,
            status=HypothesisStatus.ACTIVE,
        )

        if track.history:
            hypothesis.positions = list(track.history)

        if hasattr(track, 'bbox_history') and track.bbox_history:
            hypothesis.bbox_history = list(track.bbox_history)

        if track.features is not None:
            hypothesis.add_feature(track.features)

        if hasattr(track, 'velocity') and track.velocity:
            hypothesis.velocity = track.velocity

        return hypothesis

    def create_hypothesis_from_detection(
        self,
        detection: Dict[str, Any],
        track_id: Optional[int] = None,
        confidence: float = 0.5
    ) -> TrackHypothesis:
        """
        Crea una nueva hipótesis a partir de una detección.

        Args:
            detection: Detección actual.
            track_id: ID del track (opcional).
            confidence: Confianza de la hipótesis.

        Returns:
            TrackHypothesis: Nueva hipótesis creada.

        Note:
            Útil para crear hipótesis iniciales para nuevas detecciones.
        """
        if track_id is None:
            track_id = -1

        hypothesis = TrackHypothesis(
            track_id=track_id,
            confidence=confidence,
            probability=0.05,
            last_update=time.time(),
            active=True,
            status=HypothesisStatus.ACTIVE,
        )

        centroid = detection.get('centroid')
        if centroid is not None:
            hypothesis.positions.append(centroid)

        bbox = detection.get('box')
        if bbox is not None:
            hypothesis.bbox_history.append(bbox)

        features = detection.get('features')
        if features is not None:
            hypothesis.add_feature(features)

        return hypothesis

    def update_with_observations(
        self,
        tracks: Dict[int, TrackState],
        detections: List[Dict[str, Any]],
        matches: List[Tuple[int, int]],
        unmatched_tracks: List[int],
        unmatched_detections: List[int]
    ) -> Dict[int, Optional[int]]:
        """
        Actualiza el árbol de hipótesis con nuevas observaciones.

        Args:
            tracks: Diccionario de tracks activos.
            detections: Lista de detecciones actuales.
            matches: Lista de matches (detection_idx, track_idx).
            unmatched_tracks: Índices de tracks no asociados.
            unmatched_detections: Índices de detecciones no asociadas.

        Returns:
            Dict[int, Optional[int]]: Hipótesis confirmadas por track.

        Note:
            Este es el método principal que integra MHT con el tracker.
            Procesa matches, tracks perdidos y detecciones no asociadas.
        """
        if not self.enable_mht:
            return {}

        start_time = time.perf_counter()

        if not detections:
            return {}

        self._cleanup_recently_recovered()

        track_ids = list(tracks.keys())

        self._process_matches(tracks, detections, matches, track_ids)

        self._handle_unmatched_tracks(tracks, matches, unmatched_tracks, track_ids)

        confirmed = self._attempt_recoveries(detections, unmatched_detections, tracks)

        self._create_temporary_hypotheses(detections, unmatched_detections)

        self._prune_tree()

        self._update_stats(tracks)

        self._log_performance(start_time, matches)

        return confirmed

    def _cleanup_recently_recovered(self) -> None:
        """Limpia los tracks recuperados que han expirado."""
        current_time = time.time()
        expired = [
            tid for tid, ts in self._recently_recovered.items()
            if current_time - ts > self._recovery_cooldown
        ]
        for tid in expired:
            del self._recently_recovered[tid]

    def _process_matches(
        self,
        tracks: Dict[int, TrackState],
        detections: List[Dict[str, Any]],
        matches: List[Tuple[int, int]],
        track_ids: List[int]
    ) -> None:
        """
        Procesa los matches entre detecciones y tracks.

        Args:
            tracks: Diccionario de tracks activos.
            detections: Lista de detecciones actuales.
            matches: Lista de matches (detection_idx, track_idx).
            track_ids: Lista de IDs de tracks.
        """
        for det_idx, track_idx in matches:
            if track_idx >= len(track_ids) or det_idx >= len(detections):
                continue

            track_id = track_ids[track_idx]
            detection = detections[det_idx]

            if detection is None:
                continue

            self._process_single_match(track_id, detection, tracks)

    def _process_single_match(
        self,
        track_id: int,
        detection: Dict[str, Any],
        tracks: Dict[int, TrackState]
    ) -> None:
        """
        Procesa un match individual entre una detección y un track.

        Args:
            track_id: ID del track.
            detection: Detección asociada.
            tracks: Diccionario de tracks activos.
        """
        best_hyp = self.hypothesis_tree.get_best_hypothesis(track_id)

        try:
            if best_hyp is not None:
                self._update_existing_hypothesis(best_hyp, detection, track_id)
            else:
                self._create_new_hypothesis_from_track(track_id, detection, tracks)

        except Exception as e:
            self.logger.warning(
                f"Error procesando match para track {track_id}: {e}",
                exc_info=True
            )

    def _update_existing_hypothesis(
        self,
        hypothesis: TrackHypothesis,
        detection: Dict[str, Any],
        track_id: int
    ) -> None:
        """
        Actualiza una hipótesis existente con nueva detección.

        Args:
            hypothesis: Hipótesis a actualizar.
            detection: Detección actual.
            track_id: ID del track.
        """
        centroid = detection.get('centroid')
        if centroid is not None:
            hypothesis.update_position(
                centroid,
                detection.get('box')
            )

        features = detection.get('features')
        if features is not None:
            hypothesis.add_feature(features)

        hypothesis.confidence = detection.get('confidence', 0.5)
        hypothesis.last_update = time.time()

        if len(self.hypothesis_tree._hypotheses.get(track_id, [])) < 10:
            self._create_alternative_hypothesis(track_id, detection)

    def _create_alternative_hypothesis(
        self,
        track_id: int,
        detection: Dict[str, Any]
    ) -> None:
        """
        Crea una hipótesis alternativa para un track.

        Args:
            track_id: ID del track.
            detection: Detección actual.
        """
        hyps = self.hypothesis_tree._hypotheses.get(track_id, [])
        active_hyps = [
            h for h in hyps
            if h.active and h.status == HypothesisStatus.ACTIVE
        ]

        if len(active_hyps) < self.hypothesis_tree.max_hypotheses_per_track:
            new_hyp = self.create_hypothesis_from_detection(
                detection,
                track_id,
                confidence=detection.get('confidence', 0.5) * 0.3
            )
            new_hyp.probability = 0.02
            self.hypothesis_tree.add_hypothesis(track_id, new_hyp)

    def _create_new_hypothesis_from_track(
        self,
        track_id: int,
        detection: Dict[str, Any],
        tracks: Dict[int, TrackState]
    ) -> None:
        """
        Crea una nueva hipótesis a partir de un track existente.

        Args:
            track_id: ID del track.
            detection: Detección actual.
            tracks: Diccionario de tracks activos.
        """
        track = tracks.get(track_id)
        if track is not None:
            new_hyp = self.create_hypothesis_from_track(
                track,
                detection,
                confidence=detection.get('confidence', 0.5)
            )
            new_hyp.probability = 0.3
            self.hypothesis_tree.add_hypothesis(track_id, new_hyp)

    def _handle_unmatched_tracks(
        self,
        tracks: Dict[int, TrackState],
        matches: List[Tuple[int, int]],
        unmatched_tracks: List[int],
        track_ids: List[int]
    ) -> None:
        """
        Maneja tracks no asociados (pérdidas).

        Args:
            tracks: Diccionario de tracks activos.
            matches: Lista de matches.
            unmatched_tracks: Índices de tracks no asociados.
            track_ids: Lista de IDs de tracks.
        """
        for track_idx in unmatched_tracks:
            if track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            if track_id not in tracks:
                continue

            self._handle_single_unmatched_track(track_id, tracks)

    def _handle_single_unmatched_track(
        self,
        track_id: int,
        tracks: Dict[int, TrackState]
    ) -> None:
        """
        Maneja un track individual no asociado.

        Args:
            track_id: ID del track.
            tracks: Diccionario de tracks activos.
        """
        best_hyp = self.hypothesis_tree.get_best_hypothesis(track_id)

        if best_hyp is not None and len(best_hyp.positions) > 3:
            self._create_predicted_hypothesis(track_id, best_hyp, tracks)

    def _create_predicted_hypothesis(
        self,
        track_id: int,
        best_hyp: TrackHypothesis,
        tracks: Dict[int, TrackState]
    ) -> None:
        """
        Crea una hipótesis basada en predicción para un track perdido.

        Args:
            track_id: ID del track.
            best_hyp: Mejor hipótesis actual.
            tracks: Diccionario de tracks activos.
        """
        velocity = best_hyp.get_recent_velocity()
        last_pos = best_hyp.positions[-1]
        predicted_pos = (
            int(last_pos[0] + velocity[0]),
            int(last_pos[1] + velocity[1])
        )

        new_hyp = TrackHypothesis(
            track_id=track_id,
            positions=best_hyp.positions + [predicted_pos],
            confidence=best_hyp.confidence * 0.4,
            probability=best_hyp.probability * 0.2,
            last_update=time.time(),
            active=True,
            parent_id=track_id,
            status=HypothesisStatus.ACTIVE,
            velocity=best_hyp.velocity,
        )

        track = tracks.get(track_id)
        if track is not None and hasattr(track, 'features') and track.features is not None:
            new_hyp.add_feature(track.features)

        self.hypothesis_tree.add_hypothesis(track_id, new_hyp)

    def _attempt_recoveries(
        self,
        detections: List[Dict[str, Any]],
        unmatched_detections: List[int],
        tracks: Dict[int, TrackState]
    ) -> Dict[int, Optional[int]]:
        """
        Intenta recuperar tracks perdidos usando el árbol de hipótesis.

        Args:
            detections: Lista de detecciones actuales.
            unmatched_detections: Índices de detecciones no asociadas.
            tracks: Diccionario de tracks activos.

        Returns:
            Dict[int, Optional[int]]: Hipótesis confirmadas por track.
        """
        confirmed = {}
        recovered_count = 0

        for det_idx in unmatched_detections:
            if det_idx >= len(detections):
                continue

            detection = detections[det_idx]

            if not self._is_valid_recovery_candidate(detection):
                continue

            recovered_track_id = self._attempt_single_recovery(detection)

            if recovered_track_id is not None:
                if self._confirm_recovered_track(
                    recovered_track_id,
                    detection,
                    tracks,
                    confirmed
                ):
                    recovered_count += 1

            if detection.get('confidence', 0) > 0.5 and recovered_count < 2:
                self._create_temporary_hypothesis(detection)

        return confirmed

    def _is_valid_recovery_candidate(self, detection: Dict[str, Any]) -> bool:
        """
        Verifica si una detección es válida para intentar recuperación.

        Args:
            detection: Detección a verificar.

        Returns:
            bool: True si es válida.
        """
        if detection is None or detection.get('confidence', 0) < 0.3:
            return False

        bbox = detection.get('box')
        if bbox:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < 20 or height < 20 or width > 300 or height > 300:
                return False

        return True

    def _attempt_single_recovery(self, detection: Dict[str, Any]) -> Optional[int]:
        """
        Intenta recuperar un track perdido con una detección.

        Args:
            detection: Detección actual.

        Returns:
            Optional[int]: ID del track recuperado o None.
        """
        self._stats["recovery_attempts"] += 1

        centroid = detection.get('centroid')
        if centroid is None:
            return None

        features = detection.get('features')

        current_time = time.time()
        active_recoveries = {
            tid: ts for tid, ts in self._recently_recovered.items()
            if current_time - ts < self._recovery_cooldown
        }

        best_match = None
        best_score = 0.0
        best_track_id = None

        with self.hypothesis_tree._lock:
            for track_id, hyps in list(self.hypothesis_tree._hypotheses.items()):
                if track_id < 0:
                    continue

                if track_id in active_recoveries:
                    continue

                if track_id in self._recently_recovered:
                    continue

                for hyp in hyps:
                    if not hyp.active or not hyp.positions:
                        continue

                    if len(hyp.positions) < 3:
                        continue

                    score, track_id_match = self._evaluate_recovery_candidate(
                        hyp, centroid, features, track_id
                    )

                    if score > best_score:
                        best_score = score
                        best_match = hyp
                        best_track_id = track_id_match

        return self._finalize_recovery(best_track_id, best_match, current_time)

    def _evaluate_recovery_candidate(
        self,
        hyp: TrackHypothesis,
        centroid: Tuple[int, int],
        features: Optional[np.ndarray],
        track_id: int
    ) -> Tuple[float, Optional[int]]:
        """
        Evalúa un candidato para recuperación.

        Args:
            hyp: Hipótesis a evaluar.
            centroid: Centroide de la detección.
            features: Features de la detección.
            track_id: ID del track.

        Returns:
            Tuple[float, Optional[int]]: (puntuación, track_id)
        """
        last_pos = hyp.positions[-1]
        spatial_dist = np.linalg.norm(
            np.array(centroid) - np.array(last_pos)
        )

        if spatial_dist > 80.0:
            return 0.0, None

        feature_similarity = self._compute_feature_similarity(features, hyp)

        spatial_score = 1.0 - min(1.0, spatial_dist / 80.0)
        combined_score = (
            0.5 * spatial_score +
            0.3 * feature_similarity +
            0.2 * hyp.probability
        )

        recovery_threshold = 0.35
        if combined_score > recovery_threshold and track_id < 10000:
            return combined_score, track_id

        return 0.0, None

    def _compute_feature_similarity(
        self,
        features: Optional[np.ndarray],
        hyp: TrackHypothesis
    ) -> float:
        """
        Calcula la similitud de features entre detección e hipótesis.

        Args:
            features: Features de la detección.
            hyp: Hipótesis a comparar.

        Returns:
            float: Similitud (0-1).
        """
        if features is None or not hyp.features:
            return 0.2

        avg_feature = hyp.get_average_feature()
        if avg_feature is None:
            return 0.2

        norm_feat = np.linalg.norm(features)
        norm_avg = np.linalg.norm(avg_feature)

        if norm_feat > 0 and norm_avg > 0:
            similarity = np.dot(features, avg_feature) / (
                norm_feat * norm_avg + 1e-8
            )
            return max(0.0, min(0.5, similarity))

        return 0.2

    def _finalize_recovery(
        self,
        best_track_id: Optional[int],
        best_match: Optional[TrackHypothesis],
        current_time: float
    ) -> Optional[int]:
        """
        Finaliza el proceso de recuperación.

        Args:
            best_track_id: ID del mejor track.
            best_match: Mejor hipótesis.
            current_time: Timestamp actual.

        Returns:
            Optional[int]: ID del track recuperado o None.
        """
        if best_track_id is not None and best_match is not None:
            self._recently_recovered[best_track_id] = current_time

            if best_match.probability > 0.5:
                best_match.probability = 0.5

            return best_track_id

        return None

    def _confirm_recovered_track(
        self,
        recovered_track_id: int,
        detection: Dict[str, Any],
        tracks: Dict[int, TrackState],
        confirmed: Dict[int, Optional[int]]
    ) -> bool:
        """
        Confirma la recuperación de un track.

        Args:
            recovered_track_id: ID del track recuperado.
            detection: Detección que lo recuperó.
            tracks: Diccionario de tracks activos.
            confirmed: Diccionario de confirmaciones.

        Returns:
            bool: True si la recuperación fue exitosa.
        """
        if recovered_track_id not in tracks:
            best_hyp = self.hypothesis_tree.get_best_hypothesis(recovered_track_id)
            if best_hyp is not None and best_hyp.probability > 0.3:
                centroid = detection.get('centroid')
                if centroid is not None:
                    best_hyp.update_position(
                        centroid,
                        detection.get('box')
                    )
                best_hyp.confidence = detection.get('confidence', 0.5)
                best_hyp.last_update = time.time()

                new_prob = min(0.6, best_hyp.probability * 1.05)
                best_hyp.probability = new_prob

                self.hypothesis_tree._normalize_probabilities(recovered_track_id)

                confirmed[recovered_track_id] = id(best_hyp)
                self._stats["tracks_recovered_by_mht"] += 1
                self._stats["successful_recoveries"] += 1
                self._stats["last_recovery_time"] = time.time()

                self.logger.debug(
                    "Track recuperado por MHT",
                    track_id=recovered_track_id,
                    probability=f"{best_hyp.probability:.3f}"
                )

                return True

        return False

    def _create_temporary_hypothesis(self, detection: Dict[str, Any]) -> None:
        """
        Crea una hipótesis temporal para una detección de alta confianza.

        Args:
            detection: Detección para crear hipótesis.
        """
        temp_id = -abs(hash(str(detection.get('centroid', (0, 0))))) % 10000
        if temp_id not in self.hypothesis_tree._hypotheses:
            new_hyp = self.create_hypothesis_from_detection(
                detection,
                temp_id,
                confidence=detection.get('confidence', 0.5) * 0.2
            )
            new_hyp.probability = 0.005
            self.hypothesis_tree.add_hypothesis(temp_id, new_hyp)

    def _create_temporary_hypotheses(
        self,
        detections: List[Dict[str, Any]],
        unmatched_detections: List[int]
    ) -> None:
        """
        Crea hipótesis temporales para detecciones no asociadas.

        Args:
            detections: Lista de detecciones.
            unmatched_detections: Índices de detecciones no asociadas.
        """
        recovered_count = 0

        for det_idx in unmatched_detections:
            if det_idx >= len(detections):
                continue

            detection = detections[det_idx]

            if detection.get('confidence', 0) > 0.5 and recovered_count < 2:
                self._create_temporary_hypothesis(detection)
                recovered_count += 1

    def _prune_tree(self) -> None:
        """Poda el árbol de hipótesis si es necesario."""
        if self._should_prune():
            pruned = self.hypothesis_tree.prune_all()
            if pruned > 0:
                self.logger.debug(
                    f"Árbol MHT podado: {pruned} hipótesis eliminadas"
                )

    def _should_prune(self) -> bool:
        """
        Determina si es necesario podar el árbol de hipótesis.

        Returns:
            bool: True si se debe podar.
        """
        stats = self.hypothesis_tree.get_stats()
        total_hyps = stats.get('total_hypotheses', 0)
        time_since_prune = time.time() - stats.get('last_prune_time', 0)

        return total_hyps > 30 or time_since_prune > 15.0

    def _update_stats(self, tracks: Dict[int, TrackState]) -> None:
        """
        Actualiza las estadísticas de la integración MHT.

        Args:
            tracks: Diccionario de tracks activos.
        """
        total_tracks = len(tracks)
        total_hyps = len(self.hypothesis_tree)

        self._stats.update({
            "total_hypotheses_integrated": total_hyps,
            "total_tracks_with_hyps": len(self.hypothesis_tree._active_hyps),
            "confirmed_tracks": len(self._confirmed_hypotheses),
            "active_tracks": total_tracks,
            "hypothesis_per_track": (
                total_hyps / total_tracks if total_tracks > 0 else 0.0
            ),
        })

    def _log_performance(self, start_time: float, matches: List[Tuple[int, int]]) -> None:
        """
        Registra métricas de rendimiento.

        Args:
            start_time: Timestamp de inicio.
            matches: Lista de matches procesados.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 10:
            self.logger.debug(
                "Actualización MHT completada",
                time_ms=f"{elapsed_ms:.1f}",
                matches=len(matches),
                total_hyps=len(self.hypothesis_tree)
            )

    def get_track_predictions(
        self,
        track_id: int,
        horizon: int = 10
    ) -> List[Tuple[int, int]]:
        """
        Obtiene predicciones de posición para un track usando MHT.

        Args:
            track_id: ID del track.
            horizon: Número de pasos a predecir.

        Returns:
            List[Tuple[int, int]]: Posiciones predichas.

        Note:
            Las predicciones se basan en la hipótesis más probable.
        """
        if not self.enable_mht:
            return []

        return self.hypothesis_tree.get_most_likely_positions(track_id, horizon)

    def get_hypothesis_confidence(self, track_id: int) -> float:
        """
        Obtiene la confianza del sistema MHT para un track.

        Args:
            track_id: ID del track.

        Returns:
            float: Confianza de la hipótesis más probable (0-1).
        """
        best_hyp = self.hypothesis_tree.get_best_hypothesis(track_id)
        if best_hyp is None:
            return 0.0

        return best_hyp.probability

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas completas de la integración MHT.

        Returns:
            Dict[str, Any]: Estadísticas del sistema MHT.
        """
        tree_stats = self.hypothesis_tree.get_stats()

        return {
            **self._stats,
            "tree_stats": tree_stats,
            "enabled": self.enable_mht,
            "recently_recovered": len(self._recently_recovered),
            "current_best_hyps": {
                track_id: hyp.probability
                for track_id, hyp in self._get_all_best_hypotheses().items()
                if hyp.probability > 0.1
            }
        }

    def _get_all_best_hypotheses(self) -> Dict[int, TrackHypothesis]:
        """
        Obtiene la mejor hipótesis para cada track activo.

        Returns:
            Dict[int, TrackHypothesis]: Mejores hipótesis por track.
        """
        result = {}
        for track_id in list(self.hypothesis_tree._active_hyps):
            best = self.hypothesis_tree.get_best_hypothesis(track_id)
            if best is not None:
                result[track_id] = best
        return result

    def clear(self) -> None:
        """Limpia todas las hipótesis y reinicia la integración."""
        self.hypothesis_tree.clear()
        self._track_hypothesis_map.clear()
        self._confirmed_hypotheses.clear()
        self._recently_recovered.clear()
        self._stats = {
            **self._stats,
            "total_hypotheses_integrated": 0,
            "tracks_recovered_by_mht": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "false_positives_filtered": 0,
        }
        self.logger.info("MHTIntegration limpiado")

    @property
    def enabled(self) -> bool:
        """Retorna si el sistema MHT está habilitado."""
        return self.enable_mht

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Activa o desactiva el sistema MHT."""
        self.enable_mht = value
        self._stats["mht_enabled"] = value
        if not value:
            self.clear()
        self.logger.info(f"MHT {'activado' if value else 'desactivado'}")
