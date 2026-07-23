"""
Sistema de matching jerárquico para re-identificación robusta.

Este módulo implementa un sistema de matching jerárquico que combina
múltiples estrategias para asociar detecciones con tracks existentes.

El matching jerárquico utiliza:
1. IoU (Intersection over Union) - Prioridad alta
2. Features visuales - Para re-identificación
3. Movimiento - Predicción de posición
4. Forma - Aspect ratio y área
5. Distancia espacial - Fallback
"""

from __future__ import annotations

import time
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.geometry import calculate_iou
from utils.logger import LoggerMixin
from utils.geometry import euclidean_distance
from core.constants import (
    TRACK_VALIDATION_IOU_THRESHOLD,
    TRACK_VALIDATION_FEATURE_THRESHOLD,
    TRACK_VALIDATION_MOTION_THRESHOLD,
    TRACK_VALIDATION_SHAPE_THRESHOLD,
    MAX_MATCH_DISTANCE,
)


class MatchLevel(Enum):
    """
    Niveles de matching en orden de prioridad.

    Attributes:
        IOU: Matching por IoU (más preciso)
        FEATURE: Matching por features visuales
        MOTION: Matching por predicción de movimiento
        SHAPE: Matching por forma (aspect ratio)
        SPATIAL: Matching por distancia espacial (fallback)
    """
    IOU = "iou"
    FEATURE = "feature"
    MOTION = "motion"
    SHAPE = "shape"
    SPATIAL = "spatial"


@dataclass
class MatchResult:
    """
    Resultado de una operación de matching.

    Attributes:
        matches: Lista de tuplas (detection_idx, track_idx) asociadas.
        unmatched_detections: Índices de detecciones no asociadas.
        unmatched_tracks: Índices de tracks no asociados.
        match_scores: Diccionario de puntuaciones por par.
        level_used: Nivel de matching utilizado.
        time_ms: Tiempo de ejecución en milisegundos.
    """
    matches: List[Tuple[int, int]]
    unmatched_detections: List[int]
    unmatched_tracks: List[int]
    match_scores: Dict[Tuple[int, int], float]
    level_used: MatchLevel
    time_ms: float


class TrackMatcher(LoggerMixin):
    """
    Matcher jerárquico que combina múltiples estrategias de matching.

    Este matcher intenta asociar detecciones con tracks usando
    diferentes criterios en orden de prioridad. Si un criterio no
    puede asociar todas las detecciones, se pasa al siguiente nivel.

    Características:
        - Matching en cascada por niveles de prioridad
        - Thresholds adaptativos según contexto
        - Soporte para features visuales
        - Manejo robusto de errores
        - Estadísticas por nivel de matching

    Attributes:
        iou_threshold: Umbral de IoU para matching.
        feature_threshold: Umbral de similitud de features.
        motion_threshold: Umbral de predicción de movimiento.
        shape_threshold: Umbral de similitud de forma.
        spatial_threshold: Umbral de distancia espacial.
        enable_adaptive_thresholds: Ajuste automático de umbrales.

    Example:
        >>> matcher = TrackMatcher(
        ...     iou_threshold=0.3,
        ...     feature_threshold=0.6,
        ...     enable_adaptive_thresholds=True
        ... )
        >>> result = matcher.match(detections, tracks)
        >>> for det_idx, trk_idx in result.matches:
        ...     print(f"Det {det_idx} -> Track {trk_idx}")
    """

    def __init__(
        self,
        iou_threshold: float = TRACK_VALIDATION_IOU_THRESHOLD,
        feature_threshold: float = TRACK_VALIDATION_FEATURE_THRESHOLD,
        motion_threshold: float = TRACK_VALIDATION_MOTION_THRESHOLD,
        shape_threshold: float = TRACK_VALIDATION_SHAPE_THRESHOLD,
        spatial_threshold: float = MAX_MATCH_DISTANCE,
        enable_adaptive_thresholds: bool = True,
    ):
        """
        Inicializa el matcher jerárquico.

        Args:
            iou_threshold: Umbral de IoU (0-1).
            feature_threshold: Umbral de similitud de features (0-1).
            motion_threshold: Umbral de predicción de movimiento (0-1).
            shape_threshold: Umbral de similitud de forma (0-1).
            spatial_threshold: Umbral de distancia espacial (píxeles).
            enable_adaptive_thresholds: Ajustar umbrales automáticamente.
        """
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.motion_threshold = motion_threshold
        self.shape_threshold = shape_threshold
        self.spatial_threshold = spatial_threshold
        self.enable_adaptive_thresholds = enable_adaptive_thresholds

        self._match_stats = {
            MatchLevel.IOU: {"success": 0, "total": 0},
            MatchLevel.FEATURE: {"success": 0, "total": 0},
            MatchLevel.MOTION: {"success": 0, "total": 0},
            MatchLevel.SHAPE: {"success": 0, "total": 0},
            MatchLevel.SPATIAL: {"success": 0, "total": 0},
        }

        self.logger.info(
            "TrackMatcher inicializado",
            iou_threshold=iou_threshold,
            feature_threshold=feature_threshold,
            adaptive_thresholds=enable_adaptive_thresholds
        )

    def match(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Any],
        frame_info: Optional[Dict[str, Any]] = None,
    ) -> MatchResult:
        """
        Realiza matching jerárquico entre detecciones y tracks.

        Args:
            detections: Lista de detecciones.
            tracks: Lista de tracks.
            frame_info: Información del frame (opcional).

        Returns:
            MatchResult: Resultado del matching.

        Note:
            El matching se realiza en cascada:
            1. IoU (más preciso)
            2. Features visuales
            3. Movimiento (predicción)
            4. Forma (aspect ratio)
            5. Distancia espacial (fallback)
        """
        start_time = time.perf_counter()

        if not detections or not tracks:
            return MatchResult(
                matches=[],
                unmatched_detections=list(range(len(detections))),
                unmatched_tracks=list(range(len(tracks))),
                match_scores={},
                level_used=MatchLevel.IOU,
                time_ms=0.0
            )

        unmatched_dets = set(range(len(detections)))
        unmatched_trks = set(range(len(tracks)))
        all_matches = []
        all_scores = {}

        match_levels = [
            (MatchLevel.IOU, self._match_iou_safe),
            (MatchLevel.FEATURE, self._match_features_safe),
            (MatchLevel.MOTION, self._match_motion_safe),
            (MatchLevel.SHAPE, self._match_shape_safe),
            (MatchLevel.SPATIAL, self._match_spatial_safe),
        ]

        last_used_level = MatchLevel.IOU

        for level, match_func in match_levels:
            if not unmatched_dets or not unmatched_trks:
                break

            dets_subset = [detections[i] for i in unmatched_dets]
            trks_subset = [tracks[i] for i in unmatched_trks]

            thresholds = self._get_adaptive_thresholds(level, dets_subset, trks_subset)

            try:
                matches, scores = match_func(
                    dets_subset,
                    trks_subset,
                    thresholds,
                    frame_info
                )
            except Exception as e:
                self.logger.warning(
                    f"Error en matching {level.value}: {e}",
                    dets=len(dets_subset),
                    trks=len(trks_subset)
                )
                continue

            if matches:
                mapped_matches = []
                for det_idx, trk_idx in matches:
                    orig_det_idx = list(unmatched_dets)[det_idx]
                    orig_trk_idx = list(unmatched_trks)[trk_idx]
                    mapped_matches.append((orig_det_idx, orig_trk_idx))

                    all_scores[(orig_det_idx, orig_trk_idx)] = scores.get(
                        (det_idx, trk_idx), 0.0
                    )

                all_matches.extend(mapped_matches)
                last_used_level = level

                matched_dets = {list(unmatched_dets)[i] for i, _ in matches}
                matched_trks = {list(unmatched_trks)[j] for _, j in matches}
                unmatched_dets -= matched_dets
                unmatched_trks -= matched_trks

                self._match_stats[level]["success"] += len(matches)

            self._match_stats[level]["total"] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MatchResult(
            matches=all_matches,
            unmatched_detections=list(unmatched_dets),
            unmatched_tracks=list(unmatched_trks),
            match_scores=all_scores,
            level_used=last_used_level,
            time_ms=elapsed_ms
        )

    def _compute_cost_matrix_safe(
        self,
        similarity_matrix: np.ndarray,
        threshold: float,
        matrix_name: str = "matrix"
    ) -> Tuple[np.ndarray, bool]:
        """
        Procesa una matriz de similitud de forma segura.

        Args:
            similarity_matrix: Matriz de similitud a procesar.
            threshold: Umbral para filtrar valores.
            matrix_name: Nombre de la matriz para logging.

        Returns:
            Tuple[np.ndarray, bool]: (matriz de costos, success)
        """
        if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
            self.logger.debug(
                f"Matriz {matrix_name} contiene valores inválidos",
                nan_count=np.sum(np.isnan(similarity_matrix)),
                inf_count=np.sum(np.isinf(similarity_matrix))
            )
            similarity_matrix = np.nan_to_num(
                similarity_matrix,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        similarity_matrix[similarity_matrix < threshold] = 0.0

        if not np.any(similarity_matrix > 0):
            return np.zeros_like(similarity_matrix), False

        cost_matrix = -similarity_matrix

        if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
            cost_matrix = np.nan_to_num(
                cost_matrix,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        return cost_matrix, True

    def _solve_assignment_safe(
        self,
        cost_matrix: np.ndarray,
        matrix_name: str = "cost"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Resuelve el problema de asignación de forma segura.

        Args:
            cost_matrix: Matriz de costos.
            matrix_name: Nombre de la matriz para logging.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (row_indices, col_indices)
        """
        try:
            if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
                return None, None

            if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
                self.logger.debug(
                    f"Matriz {matrix_name} tiene valores inválidos después de limpieza"
                )
                cost_matrix = np.nan_to_num(
                    cost_matrix,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

            if cost_matrix.shape[0] > 50 or cost_matrix.shape[1] > 50:
                return self._greedy_assignment(cost_matrix)

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            return row_indices, col_indices

        except Exception as e:
            self.logger.debug(
                f"Error en assignment para {matrix_name}: {e}",
                shape=cost_matrix.shape
            )
            return self._greedy_assignment(cost_matrix)

    def _greedy_assignment(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Asignación greedy para matrices problemáticas.

        Args:
            cost_matrix: Matriz de costos.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (row_indices, col_indices)
        """
        n_rows, n_cols = cost_matrix.shape

        if n_rows == 0 or n_cols == 0:
            return None, None

        row_indices = []
        col_indices = []
        used_cols = set()

        for i in range(n_rows):
            best_col = -1
            best_cost = float('inf')

            for j in range(n_cols):
                if j in used_cols:
                    continue
                if cost_matrix[i, j] < best_cost and not np.isinf(cost_matrix[i, j]):
                    best_cost = cost_matrix[i, j]
                    best_col = j

            if best_col >= 0 and best_cost < 1000:
                row_indices.append(i)
                col_indices.append(best_col)
                used_cols.add(best_col)

        return np.array(row_indices), np.array(col_indices)

    def _match_iou_safe(
        self,
        detections: List[Dict],
        tracks: List[Any],
        thresholds: Dict[str, float],
        frame_info: Optional[Dict] = None,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """
        Matching basado en IoU con manejo seguro de errores.

        Args:
            detections: Lista de detecciones.
            tracks: Lista de tracks.
            thresholds: Umbrales adaptativos.
            frame_info: Información del frame.

        Returns:
            Tuple: (matches, scores)
        """
        if not detections or not tracks:
            return [], {}

        n_dets = len(detections)
        n_trks = len(tracks)

        iou_matrix = np.zeros((n_dets, n_trks), dtype=np.float32)
        valid_pairs = 0

        for i, det in enumerate(detections):
            det_box = det.get("box")
            if not det_box or len(det_box) != 4:
                continue

            try:
                det_box = [float(v) for v in det_box]
                if any(np.isnan(v) or np.isinf(v) for v in det_box):
                    continue
            except (TypeError, ValueError):
                continue

            for j, track in enumerate(tracks):
                if not hasattr(track, 'bbox') or not track.bbox:
                    continue

                trk_box = track.bbox
                if len(trk_box) != 4:
                    continue

                try:
                    trk_box = [float(v) for v in trk_box]
                    if any(np.isnan(v) or np.isinf(v) for v in trk_box):
                        continue
                except (TypeError, ValueError):
                    continue

                try:
                    iou = calculate_iou(tuple(det_box), tuple(trk_box))
                    if isinstance(iou, (int, float)) and not np.isnan(iou) and not np.isinf(iou):
                        iou_matrix[i, j] = float(max(0.0, min(1.0, iou)))
                        valid_pairs += 1
                except Exception:
                    iou_matrix[i, j] = 0.0

        if valid_pairs == 0:
            return [], {}

        threshold = thresholds.get('iou', self.iou_threshold)
        cost_matrix, success = self._compute_cost_matrix_safe(iou_matrix, threshold, "iou")

        if not success:
            return [], {}

        row_indices, col_indices = self._solve_assignment_safe(cost_matrix, "iou")

        if row_indices is None or col_indices is None:
            return [], {}

        matches = []
        scores = {}

        for row, col in zip(row_indices, col_indices):
            if row < n_dets and col < n_trks:
                iou_value = iou_matrix[row, col]
                if iou_value > 0:
                    matches.append((row, col))
                    scores[(row, col)] = float(iou_value)

        return matches, scores

    def _match_features_safe(
        self,
        detections: List[Dict],
        tracks: List[Any],
        thresholds: Dict[str, float],
        frame_info: Optional[Dict] = None,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """Matching basado en features con manejo seguro de errores."""
        if not detections or not tracks:
            return [], {}

        n_dets = len(detections)
        n_trks = len(tracks)

        similarity_matrix = np.zeros((n_dets, n_trks), dtype=np.float32)
        valid_pairs = 0

        for i, det in enumerate(detections):
            det_features = det.get("features")
            if det_features is None:
                continue

            if not isinstance(det_features, np.ndarray):
                continue

            norm_det = np.linalg.norm(det_features)
            if norm_det == 0:
                continue

            for j, track in enumerate(tracks):
                if not hasattr(track, 'features') or track.features is None:
                    continue

                try:
                    norm_trk = np.linalg.norm(track.features)
                    if norm_trk > 0:
                        similarity = np.dot(det_features, track.features) / (norm_det * norm_trk)
                        similarity = max(0.0, min(1.0, similarity))
                        if not np.isnan(similarity) and not np.isinf(similarity):
                            similarity_matrix[i, j] = float(similarity)
                            valid_pairs += 1
                except Exception:
                    similarity_matrix[i, j] = 0.0

        if valid_pairs == 0:
            return [], {}

        threshold = thresholds.get('feature', self.feature_threshold)
        cost_matrix, success = self._compute_cost_matrix_safe(similarity_matrix, threshold, "feature")

        if not success:
            return [], {}

        row_indices, col_indices = self._solve_assignment_safe(cost_matrix, "feature")

        if row_indices is None or col_indices is None:
            return [], {}

        matches = []
        scores = {}

        for row, col in zip(row_indices, col_indices):
            if row < n_dets and col < n_trks:
                sim_value = similarity_matrix[row, col]
                if sim_value > 0:
                    matches.append((row, col))
                    scores[(row, col)] = float(sim_value)

        return matches, scores

    def _match_motion_safe(
        self,
        detections: List[Dict],
        tracks: List[Any],
        thresholds: Dict[str, float],
        frame_info: Optional[Dict] = None,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """Matching basado en movimiento con manejo seguro de errores."""
        if not detections or not tracks:
            return [], {}

        n_dets = len(detections)
        n_trks = len(tracks)

        distance_matrix = np.zeros((n_dets, n_trks), dtype=np.float32)
        valid_pairs = 0

        for i, det in enumerate(detections):
            det_centroid = det.get("centroid")
            if not det_centroid:
                continue

            for j, track in enumerate(tracks):
                if hasattr(track, 'predicted_centroid'):
                    track_pos = track.predicted_centroid
                elif hasattr(track, 'centroid'):
                    track_pos = track.centroid
                else:
                    continue

                try:
                    distance = euclidean_distance(det_centroid, track_pos)
                    score = 1.0 / (1.0 + distance / 10.0)
                    score = max(0.0, min(1.0, score))
                    if not np.isnan(score) and not np.isinf(score):
                        distance_matrix[i, j] = float(score)
                        valid_pairs += 1
                except Exception:
                    distance_matrix[i, j] = 0.0

        if valid_pairs == 0:
            return [], {}

        threshold = thresholds.get('motion', self.motion_threshold)
        cost_matrix, success = self._compute_cost_matrix_safe(distance_matrix, threshold, "motion")

        if not success:
            return [], {}

        row_indices, col_indices = self._solve_assignment_safe(cost_matrix, "motion")

        if row_indices is None or col_indices is None:
            return [], {}

        matches = []
        scores = {}

        for row, col in zip(row_indices, col_indices):
            if row < n_dets and col < n_trks:
                score_value = distance_matrix[row, col]
                if score_value > 0:
                    matches.append((row, col))
                    scores[(row, col)] = float(score_value)

        return matches, scores

    def _match_shape_safe(
        self,
        detections: List[Dict],
        tracks: List[Any],
        thresholds: Dict[str, float],
        frame_info: Optional[Dict] = None,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """Matching basado en forma con manejo seguro de errores."""
        if not detections or not tracks:
            return [], {}

        n_dets = len(detections)
        n_trks = len(tracks)

        shape_matrix = np.zeros((n_dets, n_trks), dtype=np.float32)
        valid_pairs = 0

        for i, det in enumerate(detections):
            det_box = det.get("box")
            if not det_box or len(det_box) != 4:
                continue

            try:
                det_w = float(det_box[2] - det_box[0])
                det_h = float(det_box[3] - det_box[1])
                if det_w <= 0 or det_h <= 0:
                    continue
                det_area = det_w * det_h
                det_ratio = det_w / det_h
            except (TypeError, ValueError):
                continue

            for j, track in enumerate(tracks):
                if not hasattr(track, 'bbox') or not track.bbox:
                    continue

                trk_box = track.bbox
                if len(trk_box) != 4:
                    continue

                try:
                    trk_w = float(trk_box[2] - trk_box[0])
                    trk_h = float(trk_box[3] - trk_box[1])
                    if trk_w <= 0 or trk_h <= 0:
                        continue
                    trk_area = trk_w * trk_h
                    trk_ratio = trk_w / trk_h
                except (TypeError, ValueError):
                    continue

                area_sim = min(det_area, trk_area) / (max(det_area, trk_area) + 1e-6)
                ratio_sim = 1.0 - abs(det_ratio - trk_ratio) / (max(det_ratio, trk_ratio) + 1e-6)

                score = 0.6 * area_sim + 0.4 * max(0.0, ratio_sim)
                score = max(0.0, min(1.0, score))

                if not np.isnan(score) and not np.isinf(score):
                    shape_matrix[i, j] = float(score)
                    valid_pairs += 1

        if valid_pairs == 0:
            return [], {}

        threshold = thresholds.get('shape', self.shape_threshold)
        cost_matrix, success = self._compute_cost_matrix_safe(shape_matrix, threshold, "shape")

        if not success:
            return [], {}

        row_indices, col_indices = self._solve_assignment_safe(cost_matrix, "shape")

        if row_indices is None or col_indices is None:
            return [], {}

        matches = []
        scores = {}

        for row, col in zip(row_indices, col_indices):
            if row < n_dets and col < n_trks:
                score_value = shape_matrix[row, col]
                if score_value > 0:
                    matches.append((row, col))
                    scores[(row, col)] = float(score_value)

        return matches, scores

    def _match_spatial_safe(
        self,
        detections: List[Dict],
        tracks: List[Any],
        thresholds: Dict[str, float],
        frame_info: Optional[Dict] = None,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """Matching basado en distancia espacial con manejo seguro de errores."""
        if not detections or not tracks:
            return [], {}

        n_dets = len(detections)
        n_trks = len(tracks)

        distance_matrix = np.zeros((n_dets, n_trks), dtype=np.float32)
        valid_pairs = 0

        for i, det in enumerate(detections):
            det_centroid = det.get("centroid")
            if not det_centroid:
                continue

            for j, track in enumerate(tracks):
                if not hasattr(track, 'centroid'):
                    continue

                try:
                    distance = euclidean_distance(det_centroid, track.centroid)
                    score = 1.0 / (1.0 + distance / 5.0)
                    score = max(0.0, min(1.0, score))
                    if not np.isnan(score) and not np.isinf(score):
                        distance_matrix[i, j] = float(score)
                        valid_pairs += 1
                except Exception:
                    distance_matrix[i, j] = 0.0

        if valid_pairs == 0:
            return [], {}

        threshold = thresholds.get('spatial', self.spatial_threshold) / 100.0
        cost_matrix, success = self._compute_cost_matrix_safe(distance_matrix, threshold, "spatial")

        if not success:
            return [], {}

        row_indices, col_indices = self._solve_assignment_safe(cost_matrix, "spatial")

        if row_indices is None or col_indices is None:
            return [], {}

        matches = []
        scores = {}

        for row, col in zip(row_indices, col_indices):
            if row < n_dets and col < n_trks:
                score_value = distance_matrix[row, col]
                if score_value > 0:
                    matches.append((row, col))
                    scores[(row, col)] = float(score_value)

        return matches, scores

    def _get_adaptive_thresholds(
        self,
        level: MatchLevel,
        detections: List[Dict],
        tracks: List[Any],
    ) -> Dict[str, float]:
        """
        Obtiene thresholds adaptativos según el contexto.

        Args:
            level: Nivel de matching.
            detections: Lista de detecciones.
            tracks: Lista de tracks.

        Returns:
            Dict[str, float]: Umbrales ajustados.

        Note:
            Los umbrales se ajustan basándose en:
            - Confianza promedio de las detecciones
            - Calidad de los features disponibles
            - Velocidad promedio de los tracks
        """
        if not self.enable_adaptive_thresholds:
            return {}

        thresholds = {}

        if level == MatchLevel.IOU:
            avg_confidence = np.mean([d.get('confidence', 0.5) for d in detections])
            thresholds['iou'] = self.iou_threshold * (0.8 + 0.4 * (1 - avg_confidence))
            thresholds['iou'] = max(0.1, min(0.7, thresholds['iou']))

        elif level == MatchLevel.FEATURE:
            feature_quality = self._estimate_feature_quality(detections, tracks)
            thresholds['feature'] = self.feature_threshold * (1.0 - 0.3 * feature_quality)
            thresholds['feature'] = max(0.3, min(0.9, thresholds['feature']))

        elif level == MatchLevel.MOTION:
            avg_speed = self._estimate_avg_speed(tracks)
            thresholds['motion'] = self.motion_threshold * (0.7 + 0.3 * min(1.0, avg_speed / 20.0))
            thresholds['motion'] = max(0.3, min(0.9, thresholds['motion']))

        return thresholds

    def _estimate_feature_quality(self, detections: List[Dict], tracks: List[Any]) -> float:
        """
        Estima la calidad promedio de los features disponibles.

        Args:
            detections: Lista de detecciones.
            tracks: Lista de tracks.

        Returns:
            float: Calidad de features (0-1).
        """
        qualities = []

        for det in detections:
            if det.get("features") is not None:
                qualities.append(1.0)
            else:
                qualities.append(0.0)

        for track in tracks:
            if hasattr(track, 'features') and track.features is not None:
                qualities.append(1.0)
            else:
                qualities.append(0.0)

        return float(np.mean(qualities)) if qualities else 0.0

    def _estimate_avg_speed(self, tracks: List[Any]) -> float:
        """
        Estima la velocidad promedio de los tracks.

        Args:
            tracks: Lista de tracks.

        Returns:
            float: Velocidad promedio en píxeles/frame.
        """
        speeds = []

        for track in tracks:
            if hasattr(track, 'velocity') and track.velocity:
                speed = np.linalg.norm(track.velocity)
                if not np.isnan(speed) and not np.isinf(speed):
                    speeds.append(speed)

        return float(np.mean(speeds)) if speeds else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del matcher.

        Returns:
            Dict[str, Any]: Estadísticas por nivel de matching.
        """
        return {
            level.value: {
                "success_rate": stats["success"] / max(1, stats["total"]),
                "total_attempts": stats["total"],
                "successful_matches": stats["success"],
            }
            for level, stats in self._match_stats.items()
        }
