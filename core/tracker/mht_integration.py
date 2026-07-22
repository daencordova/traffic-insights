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

        import time
        start_time = time.perf_counter()
        confirmed = {}

        if not detections:
            return confirmed

        current_time = time.time()
        expired = [
            tid for tid, ts in self._recently_recovered.items()
            if current_time - ts > self._recovery_cooldown
        ]
        for tid in expired:
            del self._recently_recovered[tid]

        track_ids = list(tracks.keys())

        for det_idx, track_idx in matches:
            if track_idx >= len(track_ids) or det_idx >= len(detections):
                continue

            track_id = track_ids[track_idx]
            detection = detections[det_idx]

            if detection is None:
                continue

            best_hyp = self.hypothesis_tree.get_best_hypothesis(track_id)

            try:
                if best_hyp is not None:
                    centroid = detection.get('centroid')
                    if centroid is not None:
                        best_hyp.update_position(
                            centroid,
                            detection.get('box')
                        )

                    features = detection.get('features')
                    if features is not None:
                        best_hyp.add_feature(features)

                    best_hyp.confidence = detection.get('confidence', 0.5)
                    best_hyp.last_update = time.time()

                    if len(detections) > 10:
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

                else:
                    track = tracks.get(track_id)
                    if track is not None:
                        new_hyp = self.create_hypothesis_from_track(
                            track,
                            detection,
                            confidence=detection.get('confidence', 0.5)
                        )
                        new_hyp.probability = 0.3
                        self.hypothesis_tree.add_hypothesis(track_id, new_hyp)
            except Exception as e:
                self.logger.warning(
                    f"Error procesando match para track {track_id}: {e}",
                    exc_info=True
                )
                continue

        for track_idx in unmatched_tracks:
            if track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            if track_id not in tracks:
                continue

            best_hyp = self.hypothesis_tree.get_best_hypothesis(track_id)

            if best_hyp is not None and len(best_hyp.positions) > 3:
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

        recovered_count = 0
        for det_idx in unmatched_detections:
            if det_idx >= len(detections):
                continue

            detection = detections[det_idx]
            if detection is None or detection.get('confidence', 0) < 0.3:
                continue

            bbox = detection.get('box')
            if bbox:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < 20 or height < 20 or width > 300 or height > 300:
                    continue

            if len(self.hypothesis_tree._hypotheses) > 0:
                recovered_track_id = self._attempt_recovery(detection)

                if recovered_track_id is not None:
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
                            recovered_count += 1

                            self.logger.debug(
                                "Track recuperado por MHT",
                                track_id=recovered_track_id,
                                probability=f"{best_hyp.probability:.3f}"
                            )

            if detection.get('confidence', 0) > 0.5 and recovered_count < 2:
                temp_id = -abs(hash(str(detection.get('centroid', (0, 0))))) % 10000
                if temp_id not in self.hypothesis_tree._hypotheses:
                    new_hyp = self.create_hypothesis_from_detection(
                        detection,
                        temp_id,
                        confidence=detection.get('confidence', 0.5) * 0.2
                    )
                    new_hyp.probability = 0.005
                    self.hypothesis_tree.add_hypothesis(temp_id, new_hyp)

        if self._should_prune():
            pruned = self.hypothesis_tree.prune_all()
            if pruned > 0:
                self.logger.debug(
                    f"Árbol MHT podado: {pruned} hipótesis eliminadas"
                )

        self._update_stats(tracks)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 10:
            self.logger.debug(
                "Actualización MHT completada",
                time_ms=f"{elapsed_ms:.1f}",
                matches=len(matches),
                total_hyps=len(self.hypothesis_tree)
            )

        return confirmed

    def _attempt_recovery(self, detection: Dict[str, Any]) -> Optional[int]:
        """
        Intenta recuperar un track perdido usando el árbol de hipótesis.

        Args:
            detection: Detección actual.

        Returns:
            Optional[int]: ID del track recuperado o None.

        Note:
            La recuperación se basa en:
            1. Similitud espacial (distancia)
            2. Similitud de features (si disponibles)
            3. Probabilidad de la hipótesis
        """
        if not self.enable_mht:
            return None

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

                    last_pos = hyp.positions[-1]
                    spatial_dist = np.linalg.norm(
                        np.array(centroid) - np.array(last_pos)
                    )

                    if spatial_dist > 80.0:
                        continue

                    feature_similarity = 0.2
                    if features is not None and hyp.features:
                        avg_feature = hyp.get_average_feature()
                        if avg_feature is not None:
                            norm_feat = np.linalg.norm(features)
                            norm_avg = np.linalg.norm(avg_feature)
                            if norm_feat > 0 and norm_avg > 0:
                                feature_similarity = np.dot(features, avg_feature) / (
                                    norm_feat * norm_avg + 1e-8
                                )
                                feature_similarity = max(0.0, min(0.5, feature_similarity))

                    spatial_score = 1.0 - min(1.0, spatial_dist / 80.0)
                    combined_score = (
                        0.5 * spatial_score +
                        0.3 * feature_similarity +
                        0.2 * hyp.probability
                    )

                    recovery_threshold = 0.35
                    if combined_score > best_score and combined_score > recovery_threshold:
                        if track_id < 10000:
                            best_score = combined_score
                            best_match = hyp
                            best_track_id = track_id

        if best_track_id is not None and best_match is not None:
            self._recently_recovered[best_track_id] = current_time

            if best_match.probability > 0.5:
                best_match.probability = 0.5

            return best_track_id

        return None

    def _should_prune(self) -> bool:
        """
        Determina si es necesario podar el árbol de hipótesis.

        Returns:
            bool: True si se debe podar.

        Note:
            La poda se realiza cuando hay muchas hipótesis
            o ha pasado mucho tiempo desde la última poda.
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
