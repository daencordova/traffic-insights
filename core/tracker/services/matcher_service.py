"""
Servicio de matching entre detecciones y tracks.

Maneja la asociación de detecciones con tracks existentes
usando diferentes estrategias de matching.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from core.tracker.matcher import TrackMatcher as TMatcher
from core.tracker.reidentifier import ReIDSystem
from utils.logger import LoggerMixin
from utils.geometry import calculate_iou


@dataclass
class MatchResult:
    """Resultado del matching."""
    matches: List[Tuple[int, int]]
    unmatched_detections: List[int]
    unmatched_tracks: List[int]
    match_scores: Dict[Tuple[int, int], float]
    reidentified: List[Tuple[int, int]]
    time_ms: float


class TrackMatcher(LoggerMixin):
    """
    Servicio de matching entre detecciones y tracks.

    Responsabilidades:
    - Matching basado en IoU
    - Matching basado en features (Re-ID)
    - Matching basado en movimiento
    - Matching jerárquico
    - Gestión de tracks no asociados

    Attributes:
        matcher: Matcher jerárquico
        reid_system: Sistema de re-identificación
        iou_threshold: Umbral de IoU para matching
        feature_threshold: Umbral de similitud de features
    """

    def __init__(
        self,
        matcher: Optional[TMatcher] = None,
        reid_system: Optional[ReIDSystem] = None,
        iou_threshold: float = 0.3,
        feature_threshold: float = 0.6,
        spatial_threshold: float = 50.0,
    ):
        self.matcher = matcher
        self.reid_system = reid_system
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.spatial_threshold = spatial_threshold

        self._stats = {
            "total_matches": 0,
            "iou_matches": 0,
            "feature_matches": 0,
            "reid_matches": 0,
            "unmatched_detections": 0,
            "unmatched_tracks": 0,
        }

        self.logger.info(
            "TrackMatcher inicializado",
            iou_threshold=iou_threshold,
            feature_threshold=feature_threshold,
            has_matcher=matcher is not None,
            has_reid=reid_system is not None
        )

    def match(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Any],
        frame: Optional[np.ndarray] = None
    ) -> MatchResult:
        """
        Realiza matching entre detecciones y tracks.

        Args:
            detections: Lista de detecciones
            tracks: Lista de tracks
            frame: Frame actual (opcional, para Re-ID)

        Returns:
            MatchResult: Resultado del matching
        """
        import time
        start_time = time.perf_counter()

        if not detections or not tracks:
            return MatchResult(
                matches=[],
                unmatched_detections=list(range(len(detections))),
                unmatched_tracks=list(range(len(tracks))),
                match_scores={},
                reidentified=[],
                time_ms=0.0
            )

        matches, unmatched_dets, unmatched_trks = self._hierarchical_match(
            detections, tracks
        )

        reidentified = []
        if self.reid_system and unmatched_dets and frame is not None:
            reid_results = self._attempt_reidentification(
                detections, unmatched_dets, tracks, frame
            )
            reidentified = reid_results.get("matches", [])
            unmatched_dets = reid_results.get("unmatched_dets", unmatched_dets)
            if reidentified:
                self._stats["reid_matches"] += len(reidentified)
                self.logger.info(
                    "Re-identificación exitosa",
                    count=len(reidentified)
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self._stats["total_matches"] += len(matches)
        self._stats["unmatched_detections"] += len(unmatched_dets)
        self._stats["unmatched_tracks"] += len(unmatched_trks)

        return MatchResult(
            matches=matches,
            unmatched_detections=unmatched_dets,
            unmatched_tracks=unmatched_trks,
            match_scores={},
            reidentified=reidentified,
            time_ms=elapsed_ms
        )

    def _hierarchical_match(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Any]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Realiza matching jerárquico.

        Args:
            detections: Lista de detecciones
            tracks: Lista de tracks

        Returns:
            Tuple: (matches, unmatched_dets, unmatched_tracks)
        """
        if self.matcher:
            result = self.matcher.match(detections, tracks)
            self._stats["iou_matches"] += len(result.matches)
            return (
                result.matches,
                result.unmatched_detections,
                result.unmatched_tracks
            )

        return self._iou_matching(detections, tracks)

    def _iou_matching(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Any]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Matching simple basado en IoU.

        Args:
            detections: Lista de detecciones
            tracks: Lista de tracks

        Returns:
            Tuple: (matches, unmatched_dets, unmatched_tracks)
        """
        n_dets = len(detections)
        n_trks = len(tracks)

        if n_dets == 0 or n_trks == 0:
            return [], list(range(n_dets)), list(range(n_trks))

        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as np

            iou_matrix = np.zeros((n_dets, n_trks))
            for i, det in enumerate(detections):
                det_box = det.get("box", (0, 0, 0, 0))
                for j, trk in enumerate(tracks):
                    iou_matrix[i, j] = calculate_iou(det_box, trk.bbox)

            iou_matrix[iou_matrix < self.iou_threshold] = 0

            row_indices, col_indices = linear_sum_assignment(-iou_matrix)

            matches = []
            unmatched_dets = list(range(n_dets))
            unmatched_trks = list(range(n_trks))

            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] > 0:
                    matches.append((row, col))
                    if row in unmatched_dets:
                        unmatched_dets.remove(row)
                    if col in unmatched_trks:
                        unmatched_trks.remove(col)

            self._stats["iou_matches"] += len(matches)
            return matches, unmatched_dets, unmatched_trks

        except Exception as e:
            self.logger.debug(f"Error en IoU matching: {e}")
            return [], list(range(n_dets)), list(range(n_trks))

    def _attempt_reidentification(
        self,
        detections: List[Dict[str, Any]],
        unmatched_dets: List[int],
        tracks: List[Any],
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Intenta re-identificar detecciones no asociadas.

        Args:
            detections: Lista de detecciones
            unmatched_dets: Índices de detecciones no asociadas
            tracks: Lista de tracks
            frame: Frame actual

        Returns:
            Dict: Resultados de re-identificación
        """
        if not self.reid_system or not unmatched_dets:
            return {"matches": [], "unmatched_dets": unmatched_dets}

        reidentified_matches = []
        remaining_dets = list(unmatched_dets)

        for det_idx in unmatched_dets:
            if det_idx >= len(detections):
                continue

            detection = detections[det_idx]

            track_id = self.reid_system.attempt_reidentification(
                detection=detection,
                frame=frame,
                current_tracks={t.track_id: t for t in tracks}
            )

            if track_id is not None:
                track_idx = next(
                    (i for i, t in enumerate(tracks) if t.track_id == track_id),
                    None
                )
                if track_idx is not None:
                    reidentified_matches.append((det_idx, track_idx))
                    if det_idx in remaining_dets:
                        remaining_dets.remove(det_idx)

        return {
            "matches": reidentified_matches,
            "unmatched_dets": remaining_dets
        }

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del matcher."""
        total = self._stats["iou_matches"] + self._stats["feature_matches"]
        return {
            **self._stats,
            "match_rate": total / max(1, self._stats["total_matches"]),
            "unmatched_rate": (
                self._stats["unmatched_detections"] /
                max(1, self._stats["total_matches"] + self._stats["unmatched_detections"])
            ),
        }
