"""
Tracker avanzado con re-identificación robusta.

Este módulo implementa el tracker principal del sistema que orquesta
todos los subsistemas de tracking:

- TrackManager: Gestión de ciclo de vida de tracks
- TrackMatcher: Matching entre detecciones y tracks
- TrackUpdater: Actualización de estado de tracks
- TrackStateMachine: Gestión de estados de tracks
- ReIdentificationService: Re-identificación de objetos perdidos
- SensorFusionService: Fusión de sensores
- PathPredictionService: Predicción de trayectoria
- OnlineLearningService: Aprendizaje en línea
- MHTService: Multi-Hypothesis Tracking
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from models.enums import TrackStatus
from utils.logger import LoggerMixin
from utils.helpers import get_memory_usage, force_garbage_collection
from core.validators import validate_detection
from core.tracker.services.matcher_service import TrackMatcher, MatchResult
from core.tracker.managers.track_manager import TrackManager
from core.tracker.state.state_machine import TrackStateMachine
from core.tracker.state.track_updater import TrackUpdater
from core.tracker.reidentifier import ReIDSystem
from core.tracker.sensor_fusion import SensorFusion
from core.tracker.path_predictor import PathPredictor
from core.tracker.online_learner import OnlineLearner
from core.tracker.mht_integration import MHTIntegration
from core.tracker.managers.feature_manager import FeatureManager
from core.interfaces import ITracker
from core.constants import (
    MAX_ACTIVE_TRACKS,
    MIN_HITS_TO_CONFIRM,
    MAX_FRAMES_MISSED,
    MEMORY_CHECK_INTERVAL,
    CLEANUP_INTERVAL
)


class MultiObjectTracker(ITracker, LoggerMixin):
    """
    Tracker avanzado con re-identificación robusta.

    Este tracker orquesta todos los subsistemas de tracking para
    proporcionar un seguimiento robusto de objetos en video.

    Características principales:
        - Matching jerárquico (IoU, features, movimiento, forma)
        - Re-identificación de objetos perdidos
        - Filtro de Kalman para predicción de posición
        - Fusión de sensores (visual, profundidad, térmico)
        - Predicción de trayectoria con modelos adaptativos
        - Aprendizaje en línea de features
        - Multi-Hypothesis Tracking (MHT)

    Attributes:
        config: Configuración del tracker.
        track_manager: Gestor de ciclo de vida de tracks.
        track_matcher: Servicio de matching.
        track_updater: Actualizador de estados.
        state_machine: Máquina de estados.
        feature_manager: Gestor de features.
        reid_system: Sistema de re-identificación.
        mht_integration: Sistema MHT.
        online_learner: Aprendizaje en línea.
        sensor_fusion: Fusión de sensores.
        path_predictor: Predicción de trayectoria.

    Example:
        >>> tracker = MultiObjectTracker()
        >>> frame = cv2.imread("frame.jpg")
        >>> detections = detector.detect(frame)
        >>> tracks = tracker.update(detections, frame)
        >>> for track_id, track_data in tracks.items():
        ...     print(f"Track {track_id}: {track_data['centroid']}")
    """

    MAX_DETECTIONS_PER_FRAME = 50
    CLEANUP_INTERVAL = CLEANUP_INTERVAL
    MEMORY_CHECK_INTERVAL = MEMORY_CHECK_INTERVAL

    def __init__(self) -> None:
        """Inicializa el tracker avanzado con configuración global."""
        from config.manager import config_manager

        self.config = config_manager.config.tracker
        self.global_config = config_manager.config

        self.logger.info("Inicializando MultiObjectTracker")

        self._init_managers()
        self._init_matchers()
        self._init_advanced_features()
        self._init_state_machine()

        self._frame_counter: int = 0
        self._tracking_time_ms: float = 0.0
        self._last_memory_check: float = time.time()
        self._last_cleanup_time: float = time.time()

        self._stats = self._init_stats()

        self.logger.info(
            "Tracker inicializado",
            features_enabled=self.feature_manager.is_available,
            reid_enabled=self.reid_system is not None,
            mht_enabled=self.mht_integration.enabled,
            active_tracks_limit=self.track_manager.max_active_tracks
        )

    def _init_managers(self) -> None:
        """Inicializa los gestores principales del tracker."""
        self.track_manager = TrackManager(
            max_active_tracks=self.config.max_active_tracks or MAX_ACTIVE_TRACKS
        )

        use_optimized_kalman = getattr(
            self.global_config.optimization,
            "use_optimized_kalman",
            True
        )
        self.track_updater = TrackUpdater(
            use_kalman=self.config.use_kalman,
            use_optimized_kalman=use_optimized_kalman,
            max_speed_change=50.0
        )

        self.feature_manager = self._init_feature_manager()

    def _init_matchers(self) -> None:
        """Inicializa los sistemas de matching."""
        self.reid_system = self._init_reid_system()

        self.track_matcher = TrackMatcher(
            matcher=None,
            reid_system=self.reid_system,
            iou_threshold=self.config.iou_threshold,
            feature_threshold=self.config.feature_threshold,
            spatial_threshold=self.config.max_distance
        )

    def _init_advanced_features(self) -> None:
        """Inicializa las características avanzadas del tracker."""
        self.mht_integration = self._init_mht()

        self.online_learner = self._init_online_learner()

        self.sensor_fusion = self._init_sensor_fusion()

        self.path_predictor = self._init_path_predictor()

    def _init_state_machine(self) -> None:
        """Inicializa la máquina de estados para los tracks."""
        self.state_machine = TrackStateMachine(
            min_hits_to_confirm=self.config.min_hits_to_confirm or MIN_HITS_TO_CONFIRM,
            max_frames_missed=self.config.max_frames_missed or MAX_FRAMES_MISSED
        )

    def _init_feature_manager(self) -> FeatureManager:
        """
        Inicializa el gestor de features.

        Returns:
            FeatureManager: Gestor de features configurado.

        Note:
            Los features se utilizan para re-identificación y matching.
            Se activan automáticamente si hay GPU disponible.
        """
        use_features = self._should_use_features()
        feature_extractor = None

        if use_features:
            try:
                from models.feature_extractor.factory import FeatureExtractorFactory

                feature_extractor = FeatureExtractorFactory.create_best_available()
                self.logger.info("Feature extractor activado")
            except Exception as e:
                self.logger.warning("Feature extractor desactivado", error=str(e))

        return FeatureManager(
            feature_extractor=feature_extractor,
            max_cache_size=self.config.reid_cache_size,
            max_age_seconds=self.config.reid_max_age_seconds,
            similarity_threshold=self.config.reid_similarity_threshold,
            spatial_threshold=self.config.reid_spatial_threshold
        )

    def _init_reid_system(self) -> Optional[ReIDSystem]:
        """
        Inicializa el sistema de re-identificación.

        Returns:
            Optional[ReIDSystem]: Sistema de re-identificación
                o None si está desactivado.

        Note:
            La re-identificación permite recuperar objetos perdidos
            basándose en features visuales.
        """
        if not self.config.enable_reidentification or not self.feature_manager.is_available:
            return None

        try:
            reid = ReIDSystem(
                feature_extractor=self.feature_manager.feature_extractor,
                max_cache_size=self.config.reid_cache_size,
                max_age_seconds=self.config.reid_max_age_seconds,
                similarity_threshold=self.config.reid_similarity_threshold,
                spatial_threshold=self.config.reid_spatial_threshold,
                min_features_for_reid=self.config.reid_min_features
            )
            self.logger.info("Sistema de re-identificación activado")
            return reid
        except Exception as e:
            self.logger.warning("Re-identificación desactivada", error=str(e))
            return None

    def _init_mht(self) -> MHTIntegration:
        """
        Inicializa el sistema de Multi-Hypothesis Tracking (MHT).

        Returns:
            MHTIntegration: Sistema MHT configurado.

        Note:
            MHT permite mantener múltiples hipótesis de trayectoria
            para manejar ambigüedades en la asociación de datos.
        """
        return MHTIntegration(
            max_depth=getattr(self.config, "mht_max_depth", 10),
            pruning_threshold=getattr(self.config, "mht_pruning_threshold", 0.01),
            max_hypotheses_per_track=getattr(self.config, "mht_max_hypotheses", 5),
            enable_mht=getattr(self.config, "enable_mht", False)
        )

    def _init_online_learner(self) -> Optional[OnlineLearner]:
        """
        Inicializa el sistema de aprendizaje en línea.

        Returns:
            Optional[OnlineLearner]: Sistema de aprendizaje
                o None si está desactivado.

        Note:
            El aprendizaje en línea permite adaptar los features
            a cambios de apariencia del objeto.
        """
        if not self.feature_manager.is_available or not self.config.enable_reidentification:
            return None

        try:
            learner = OnlineLearner(
                feature_dim=2048,
                learning_rate=getattr(self.config, "online_learning_rate", 0.05),
                min_samples=getattr(self.config, "online_learning_min_samples", 5),
                drift_threshold=getattr(self.config, "online_learning_drift_threshold", 0.35),
                max_history=getattr(self.config, "online_learning_max_history", 50),
                strategy=getattr(self.config, "online_learning_strategy", "adaptive")
            )
            self.logger.info("Sistema de aprendizaje en línea activado")
            return learner
        except Exception as e:
            self.logger.warning("Aprendizaje en línea desactivado", error=str(e))
            return None

    def _init_sensor_fusion(self) -> Optional[SensorFusion]:
        """
        Inicializa el sistema de fusión de sensores.

        Returns:
            Optional[SensorFusion]: Sistema de fusión
                o None si está desactivado.

        Note:
            La fusión de sensores combina información de múltiples
            fuentes (visual, profundidad, térmico) para mejorar
            la robustez del tracking.
        """
        if not getattr(self.config, "enable_sensor_fusion", False):
            return None

        try:
            from core.tracker.sensor_fusion import SensorType
            fusion = SensorFusion(
                sensor_weights={
                    SensorType.VISUAL: getattr(self.config, "fusion_visual_weight", 0.7),
                    SensorType.DEPTH: getattr(self.config, "fusion_depth_weight", 0.5),
                    SensorType.THERMAL: getattr(self.config, "fusion_thermal_weight", 0.4),
                    SensorType.MOTION: getattr(self.config, "fusion_motion_weight", 0.3),
                },
                fusion_method=getattr(self.config, "fusion_method", "weighted_average"),
                min_observations=getattr(self.config, "fusion_min_observations", 2),
                max_history=getattr(self.config, "fusion_max_history", 50),
                particle_count=getattr(self.config, "fusion_particle_count", 500)
            )
            self.logger.info("Sistema de fusión de sensores activado")
            return fusion
        except Exception as e:
            self.logger.warning("Fusión de sensores desactivada", error=str(e))
            return None

    def _init_path_predictor(self) -> Optional[PathPredictor]:
        """
        Inicializa el sistema de predicción de trayectoria.

        Returns:
            Optional[PathPredictor]: Sistema de predicción
                o None si está desactivado.

        Note:
            La predicción de trayectoria permite anticipar el
            movimiento futuro de los objetos para mejorar el tracking.
        """
        if not getattr(self.config, "enable_path_prediction", True):
            return None

        try:
            predictor = PathPredictor(
                history_length=getattr(self.config, "prediction_history_length", 30),
                prediction_horizon=getattr(self.config, "prediction_horizon", 2.0),
                prediction_steps=getattr(self.config, "prediction_steps", 20),
                min_samples=getattr(self.config, "prediction_min_samples", 5),
                motion_model=getattr(self.config, "prediction_motion_model", "adaptive"),
                uncertainty_threshold=getattr(self.config, "prediction_uncertainty_threshold", 0.7)
            )
            self.logger.info("Sistema de predicción de trayectoria activado")
            return predictor
        except Exception as e:
            self.logger.warning("Predicción de trayectoria desactivada", error=str(e))
            return None

    def _init_stats(self) -> Dict[str, Any]:
        """
        Inicializa las estadísticas del tracker.

        Returns:
            Dict[str, Any]: Diccionario de estadísticas iniciales.
        """
        return {
            "total_tracks": 0,
            "confirmed_tracks": 0,
            "lost_tracks": 0,
            "reidentified_tracks": 0,
            "tracking_time_ms": 0,
            "features_used": self.feature_manager.is_available,
        }

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Actualiza el tracker con nuevas detecciones.

        Args:
            detections: Lista de detecciones del frame actual.
            frame: Imagen actual para extraer features y contexto.

        Returns:
            Dict[int, Dict[str, Any]]: Información de tracking actualizada,
                donde la clave es el track_id y el valor contiene:
                - centroid: Centroide del objeto
                - bbox: Bounding box
                - status: Estado del track
                - age: Edad en frames
                - hits: Número de detecciones asociadas
                - confidence: Confianza del track
                - velocity: Velocidad actual
                - history: Historial de posiciones
                - predicted_centroid: Posición predicha

        Raises:
            TrackingError: Si ocurre un error durante el tracking.

        Example:
            >>> detections = detector.detect(frame)
            >>> tracks = tracker.update(detections, frame)
            >>> for track_id, data in tracks.items():
            ...     print(f"Track {track_id} en {data['centroid']}")
        """
        if frame is None or frame.size == 0:
            return {}

        start_time = time.perf_counter()
        self._frame_counter += 1
        self._check_memory()

        valid_detections = self._validate_detections(detections)

        if valid_detections and self.feature_manager.is_available:
            self._extract_features(valid_detections, frame)

        self._predict_positions()

        match_result = self._perform_matching(valid_detections, frame)

        self._update_tracks(valid_detections, match_result)

        self._update_advanced_systems(valid_detections, match_result)

        self._handle_unmatched(match_result)

        self._create_new_tracks(valid_detections, match_result)

        if self.reid_system and match_result.unmatched_detections:
            self._perform_reidentification(
                valid_detections,
                match_result.unmatched_detections,
                frame
            )

        self._perform_cleanup()

        self._update_stats()

        self._tracking_time_ms = (time.perf_counter() - start_time) * 1000
        self._stats["tracking_time_ms"] = self._tracking_time_ms

        return self.get_tracking_info()

    def _perform_matching(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray
    ) -> MatchResult:
        """
        Realiza matching entre detecciones y tracks existentes.

        Args:
            detections: Lista de detecciones.
            frame: Frame actual para contexto.

        Returns:
            MatchResult: Resultado del matching con matches y no-matches.

        Note:
            Utiliza el matcher jerárquico que combina múltiples
            criterios (IoU, features, movimiento, forma).
        """
        tracks = list(self.track_manager.get_all_tracks().values())

        if not detections or not tracks:
            return MatchResult(
                matches=[],
                unmatched_detections=list(range(len(detections))),
                unmatched_tracks=list(range(len(tracks))),
                match_scores={},
                reidentified=[],
                time_ms=0.0
            )

        return self.track_matcher.match(detections, tracks, frame)

    def _update_tracks(
        self,
        detections: List[Dict[str, Any]],
        match_result: MatchResult
    ) -> None:
        """
        Actualiza tracks con nuevas detecciones asociadas.

        Args:
            detections: Lista de detecciones.
            match_result: Resultado del matching con matches.
        """
        tracks = self.track_manager.get_all_tracks()
        track_ids = list(tracks.keys())

        for det_idx, track_idx in match_result.matches:
            if det_idx >= len(detections) or track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            detection = detections[det_idx]
            features = detection.get("features")

            self.track_manager.update_track(track_id, detection, features)
            track = self.track_manager.get_track(track_id)

            if track:
                self.track_updater.correct_position(track, detection)
                self.track_updater.update_motion_metrics(track)

                new_status = self.state_machine.transition(
                    track.status,
                    track.hits,
                    track.no_losses
                )
                track.status = new_status

                if features is not None:
                    self.feature_manager.cache_features(track_id, features, track.confidence)

                if track_id in match_result.reidentified:
                    self.logger.info(
                        "Track recuperado y actualizado",
                        track_id=track_id,
                        confidence=track.confidence
                    )

    def _handle_unmatched(self, match_result: MatchResult) -> None:
        """
        Maneja tracks no asociados (pérdidas).

        Args:
            match_result: Resultado del matching con tracks no asociados.
        """
        tracks = self.track_manager.get_all_tracks()
        track_ids = list(tracks.keys())

        for track_idx in match_result.unmatched_tracks:
            if track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            track = self.track_manager.get_track(track_id)

            if track:
                track.mark_lost()

                new_status = self.state_machine.transition(
                    track.status,
                    track.hits,
                    track.no_losses
                )
                track.status = new_status

                if track.status == TrackStatus.DEAD:
                    self._handle_dead_track(track_id)

    def _handle_dead_track(self, track_id: int) -> None:
        """
        Maneja un track que ha muerto (no recuperable).

        Args:
            track_id: ID del track muerto.

        Note:
            Guarda los features del track para posible re-identificación
            futura y limpia los subsistemas asociados.
        """
        track = self.track_manager.get_track(track_id)
        if track is None:
            return

        self.track_manager.mark_as_lost(track_id)

        if self.reid_system and track.features is not None:
            self.reid_system.add_lost_track(
                track_id,
                track.features,
                track.confidence
            )

        if self.online_learner:
            self.online_learner.clear_track(track_id)
        if self.sensor_fusion:
            self.sensor_fusion.clear_track(track_id)
        if self.path_predictor:
            self.path_predictor.clear_track(track_id)

    def _perform_reidentification(
        self,
        detections: List[Dict[str, Any]],
        unmatched_dets: List[int],
        frame: np.ndarray
    ) -> int:
        """
        Realiza re-identificación de objetos perdidos.

        Args:
            detections: Lista de detecciones.
            unmatched_dets: Índices de detecciones no asociadas.
            frame: Frame actual para extraer features.

        Returns:
            int: Número de tracks re-identificados exitosamente.

        Note:
            Utiliza features visuales para encontrar coincidencias
            con tracks perdidos anteriormente.
        """
        if not self.reid_system or not unmatched_dets:
            return 0

        reidentified = 0

        for det_idx in unmatched_dets:
            if det_idx >= len(detections):
                continue

            detection = detections[det_idx]

            track_id = self.reid_system.attempt_reidentification(
                detection=detection,
                frame=frame,
                current_tracks=self.track_manager.get_all_tracks()
            )

            if track_id is not None:
                track = self.track_manager.recover_track(track_id)
                if track:
                    track.update(detection, detection.get("features"))
                    track.status = TrackStatus.CONFIRMED
                    track.no_losses = 0

                    self.track_updater.init_kalman(track)

                    self._stats["reidentified_tracks"] += 1
                    reidentified += 1

                    self.logger.info(
                        "Track re-identificado",
                        track_id=track_id,
                        confidence=track.confidence
                    )

        return reidentified

    def _create_new_tracks(
        self,
        detections: List[Dict[str, Any]],
        match_result: MatchResult
    ) -> None:
        """
        Crea nuevos tracks a partir de detecciones no asociadas.

        Args:
            detections: Lista de detecciones.
            match_result: Resultado del matching con detecciones no asociadas.
        """
        tracks_created = 0

        for det_idx in match_result.unmatched_detections:
            if det_idx >= len(detections):
                continue

            detection = detections[det_idx]

            confidence = detection.get("confidence", 0.0)
            if confidence < self.global_config.model.confidence_threshold:
                continue

            if not detection.get("box") or not detection.get("centroid"):
                continue

            features = detection.get("features")
            track = self.track_manager.create_track(
                detection=detection,
                features=features
            )

            if track:
                self.track_updater.init_kalman(track)
                self._init_advanced_features_for_track(track, detection, confidence, features)
                tracks_created += 1

        if tracks_created > 0:
            self.logger.debug(
                "Nuevos tracks creados",
                count=tracks_created,
                active=self.track_manager.get_active_count()
            )

    def _init_advanced_features_for_track(
        self,
        track: Any,
        detection: Dict[str, Any],
        confidence: float,
        features: Optional[np.ndarray]
    ) -> None:
        """
        Inicializa características avanzadas para un nuevo track.

        Args:
            track: Track a inicializar.
            detection: Detección asociada.
            confidence: Confianza de la detección.
            features: Features extraídos (opcional).
        """
        track_id = track.track_id

        if self.online_learner and features is not None:
            try:
                self.online_learner.update(
                    track_id=track_id,
                    features=features,
                    confidence=confidence
                )
            except Exception as e:
                self.logger.debug(
                    "Error iniciando aprendizaje en línea",
                    track_id=track_id,
                    error=str(e)
                )

        if self.sensor_fusion:
            try:
                from core.tracker.sensor_fusion import SensorObservation, SensorType
                observation = SensorObservation(
                    sensor_type=SensorType.VISUAL,
                    bbox=track.bbox,
                    centroid=track.centroid,
                    confidence=confidence,
                    track_id=track_id,
                    metadata={
                        "class_id": track.class_id,
                        "label": track.label,
                        "frame": self._frame_counter,
                    }
                )
                self.sensor_fusion.add_observation(track_id, observation)
            except Exception as e:
                self.logger.debug(
                    "Error iniciando fusión de sensores",
                    track_id=track_id,
                    error=str(e)
                )

        if self.path_predictor:
            try:
                self.path_predictor.update(
                    track_id=track_id,
                    position=track.centroid,
                    confidence=confidence
                )
            except Exception as e:
                self.logger.debug(
                    "Error iniciando predicción de trayectoria",
                    track_id=track_id,
                    error=str(e)
                )

    def _predict_positions(self) -> None:
        """Predice posiciones de todos los tracks usando filtro de Kalman."""
        for track in self.track_manager.get_all_tracks().values():
            self.track_updater.predict_position(track)

    def _extract_features(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> None:
        """
        Extrae features para todas las detecciones.

        Args:
            detections: Lista de detecciones.
            frame: Frame actual para extraer features.
        """
        for det in detections:
            if "box" in det:
                features = self.feature_manager.extract_features(
                    frame, det["box"], det.get("confidence", 0.5)
                )
                if features is not None:
                    det["features"] = features

    def _update_advanced_systems(
        self,
        detections: List[Dict[str, Any]],
        match_result: MatchResult
    ) -> None:
        """
        Actualiza todos los sistemas avanzados.

        Args:
            detections: Lista de detecciones.
            match_result: Resultado del matching.
        """
        if self.online_learner:
            self._update_online_learning(detections, match_result)

        if self.sensor_fusion:
            self._update_sensor_fusion(detections, match_result)

        if self.path_predictor:
            self._update_path_prediction()

    def _update_online_learning(
        self,
        detections: List[Dict[str, Any]],
        match_result: MatchResult
    ) -> None:
        """
        Actualiza el aprendizaje en línea con nuevas observaciones.

        Args:
            detections: Lista de detecciones.
            match_result: Resultado del matching.
        """
        if self.online_learner is None:
            return

        tracks = self.track_manager.get_all_tracks()
        track_ids = list(tracks.keys())

        for det_idx, track_idx in match_result.matches:
            if det_idx >= len(detections) or track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            detection = detections[det_idx]
            features = detection.get("features")

            if features is not None:
                try:
                    self.online_learner.update(
                        track_id=track_id,
                        features=features,
                        confidence=detection.get("confidence", 0.5)
                    )
                except Exception as e:
                    self.logger.debug(
                        "Error en aprendizaje en línea",
                        track_id=track_id,
                        error=str(e)
                    )

    def _update_sensor_fusion(
        self,
        detections: List[Dict[str, Any]],
        match_result: MatchResult
    ) -> None:
        """
        Actualiza la fusión de sensores con nuevas observaciones.

        Args:
            detections: Lista de detecciones.
            match_result: Resultado del matching.
        """
        if self.sensor_fusion is None:
            return

        from core.tracker.sensor_fusion import SensorObservation, SensorType

        tracks = self.track_manager.get_all_tracks()
        track_ids = list(tracks.keys())

        for det_idx, track_idx in match_result.matches:
            if det_idx >= len(detections) or track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            detection = detections[det_idx]

            try:
                observation = SensorObservation(
                    sensor_type=SensorType.VISUAL,
                    bbox=detection.get("box", (0, 0, 0, 0)),
                    centroid=detection.get("centroid", (0, 0)),
                    confidence=detection.get("confidence", 0.5),
                    track_id=track_id,
                    metadata={
                        "class_id": detection.get("class_id", -1),
                        "label": detection.get("label", "unknown"),
                        "frame": self._frame_counter,
                    }
                )
                self.sensor_fusion.add_observation(track_id, observation)
            except Exception as e:
                self.logger.debug(
                    "Error en fusión de sensores",
                    track_id=track_id,
                    error=str(e)
                )

    def _update_path_prediction(self) -> None:
        """Actualiza la predicción de trayectoria para todos los tracks."""
        if self.path_predictor is None:
            return

        for track_id, track in self.track_manager.get_all_tracks().items():
            try:
                prediction = self.path_predictor.update(
                    track_id=track_id,
                    position=track.centroid,
                    velocity=track.velocity,
                    confidence=track.confidence
                )

                if prediction:
                    track.metadata["path_prediction"] = {
                        "positions": prediction.positions[:5],
                        "state": prediction.state.value,
                        "uncertainty": prediction.uncertainty,
                        "collision_risk": prediction.collision_risk,
                    }
            except Exception as e:
                self.logger.debug(
                    "Error en predicción de trayectoria",
                    track_id=track_id,
                    error=str(e)
                )

    def _validate_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Valida y filtra detecciones.

        Args:
            detections: Lista de detecciones a validar.

        Returns:
            List[Dict[str, Any]]: Lista de detecciones válidas.
        """
        if not detections:
            return []

        valid = []
        for det in detections:
            if validate_detection(det, require_all_fields=True).is_valid:
                valid.append(det)
                if len(valid) >= self.MAX_DETECTIONS_PER_FRAME:
                    break

        return valid

    def _perform_cleanup(self) -> None:
        """Realiza limpieza periódica de tracks muertos."""
        current_time = time.time()
        if current_time - self._last_cleanup_time >= self.CLEANUP_INTERVAL:
            self._last_cleanup_time = current_time
            removed = self.track_manager.cleanup_dead_tracks()

            if removed > 0:
                self.logger.debug(
                    "Limpieza de tracks completada",
                    removed=removed,
                    active=self.track_manager.get_active_count(),
                    lost=self.track_manager.get_lost_count()
                )

    def _check_memory(self) -> None:
        """Verifica el uso de memoria y limpia si es necesario."""
        current_time = time.time()
        if current_time - self._last_memory_check < self.MEMORY_CHECK_INTERVAL:
            return

        self._last_memory_check = current_time

        try:
            mem = get_memory_usage()
            mem_percent = mem.get("percent", 0)

            if mem_percent > 75:
                self.logger.warning(
                    "Memoria alta, limpiando",
                    memory_percent=f"{mem_percent:.1f}",
                    active_tracks=self.track_manager.get_active_count()
                )
                self.feature_manager.clear_cache()
                force_garbage_collection()
        except Exception as e:
            self.logger.debug("Error verificando memoria", error=str(e))

    def _update_stats(self) -> None:
        """Actualiza estadísticas del tracker."""
        self._stats["total_tracks"] = self.track_manager.get_active_count()
        self._stats["confirmed_tracks"] = sum(
            1 for t in self.track_manager.get_all_tracks().values()
            if t.status == TrackStatus.CONFIRMED
        )
        self._stats["lost_tracks"] = self.track_manager.get_lost_count()

    def _should_use_features(self) -> bool:
        """
        Determina si se deben usar features visuales.

        Returns:
            bool: True si se deben usar features.

        Note:
            Los features solo se activan si hay GPU disponible
            y no se está en modo CPU forzado.
        """
        from models.enums import DeviceType
        if self.global_config.model.device == DeviceType.CPU:
            return False

        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_tracking_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Retorna información de tracking actual.

        Returns:
            Dict[int, Dict[str, Any]]: Información de tracking.
        """
        result = {}

        for track_id, track in self.track_manager.get_all_tracks().items():
            track_data = {
                "centroid": track.centroid,
                "bbox": track.bbox,
                "status": track.status.value,
                "age": track.age,
                "hits": track.hits,
                "no_losses": track.no_losses,
                "confidence": track.confidence,
                "velocity": track.velocity,
                "label": track.label,
                "class_id": track.class_id,
                "history": list(track.history),
                "predicted_centroid": track.predicted_centroid,
            }

            self._enrich_track_data(track_id, track_data)
            result[track_id] = track_data

        return result

    def _enrich_track_data(self, track_id: int, track_data: Dict[str, Any]) -> None:
        """
        Enriquece los datos del track con información de subsistemas.

        Args:
            track_id: ID del track.
            track_data: Diccionario de datos del track a enriquecer.
        """
        if self.online_learner:
            learner_stats = self.online_learner.get_stats(track_id)
            if learner_stats:
                track_data["online_learning"] = {
                    "samples": learner_stats.get("n_samples", 0),
                    "updates": learner_stats.get("total_updates", 0),
                    "drift_detected": learner_stats.get("concept_drift_detected", False),
                }

        if self.sensor_fusion:
            fused_state = self.sensor_fusion.get_fused_state(track_id)
            if fused_state:
                track_data["sensor_fusion"] = {
                    "fused_confidence": fused_state.confidence,
                    "uncertainty": fused_state.uncertainty,
                    "sensor_count": len(fused_state.sensor_contributions),
                }

        if self.path_predictor:
            prediction = self.path_predictor.get_prediction(track_id)
            if prediction:
                track_data["path_prediction"] = {
                    "state": prediction.state.value,
                    "uncertainty": prediction.uncertainty,
                    "collision_risk": prediction.collision_risk,
                }

        if self.mht_integration and self.mht_integration.enabled:
            track_data["mht_confidence"] = self.mht_integration.get_hypothesis_confidence(track_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del tracker.

        Returns:
            Dict[str, Any]: Estadísticas detalladas del tracker.
        """
        return {
            **self._stats,
            "active_tracks": self.track_manager.get_active_count(),
            "lost_tracks_count": self.track_manager.get_lost_count(),
            "feature_manager": self.feature_manager.get_stats(),
            "state_machine": self.state_machine.get_stats(),
            "track_updater": self.track_updater.get_stats(),
            "track_matcher": self.track_matcher.get_stats(),
            "tracking_time_ms": self._tracking_time_ms,
            "frame_counter": self._frame_counter,
        }

    def get_track(self, track_id: int) -> Optional[Any]:
        """
        Obtiene un track por su ID.

        Args:
            track_id: ID del track a obtener.

        Returns:
            Optional[Any]: TrackState del track o None si no existe.
        """
        return self.track_manager.get_track(track_id)

    def reset(self) -> None:
        """Reinicia el tracker completamente."""
        self.logger.info("Reiniciando tracker")

        self.track_manager.clear_all()
        self.feature_manager.clear_cache()
        self._frame_counter = 0
        self._tracking_time_ms = 0.0

        if self.reid_system:
            self.reid_system.clear_cache()
        if self.mht_integration:
            self.mht_integration.clear()
        if self.online_learner:
            self.online_learner.reset()
        if self.sensor_fusion:
            self.sensor_fusion.clear_all()
        if self.path_predictor:
            self.path_predictor.reset()

        self._stats = self._init_stats()
        self.logger.info("Tracker reiniciado")
