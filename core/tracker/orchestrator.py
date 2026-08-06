"""Orquestador del sistema de tracking.

Este módulo coordina todos los servicios de tracking:
- Gestión de estado de tracks (TrackStateManager)
- Matching entre detecciones y tracks (TrackMatcher)
- Re-identificación (ReIDSystem)
- Fusión de sensores (SensorFusion)
- Predicción de trayectoria (PathPredictor)
- Aprendizaje en línea (OnlineLearner)
- MHT (MHTIntegration)

El orquestador sigue el patrón de diseño "Facade", proporcionando
una interfaz simplificada para el sistema de tracking.
"""

import time
from typing import Any

import numpy as np

from core.constants.pipeline import CLEANUP_INTERVAL, MEMORY_CHECK_INTERVAL
from core.constants.tracking import MAX_DETECTIONS_PER_FRAME
from core.constants.values import MEMORY_WARNING_PERCENT
from core.tracker.managers.feature_manager import FeatureManager
from core.tracker.mht_integration import MHTIntegration
from core.tracker.online_learner import OnlineLearner
from core.tracker.path_predictor import PathPredictor
from core.tracker.reidentifier import ReIDSystem
from core.tracker.sensor_fusion import SensorFusion, SensorObservation, SensorType
from core.tracker.services.matcher_service import MatchResult, TrackMatcher
from core.tracker.state.state_machine import TrackStateMachine
from core.tracker.state.track_updater import TrackUpdater
from core.tracker.state_manager import TrackStateManager
from core.validators import validate_detection
from models.enums import DeviceType, TrackStatus
from models.feature_extractor.factory import FeatureExtractorFactory
from models.track_state import TrackState
from utils.helpers import force_garbage_collection, get_memory_usage
from utils.logger import LoggerMixin


class TrackOrchestrator(LoggerMixin):
    """Orquestador del sistema de tracking.

    Esta clase coordina todos los componentes del sistema de tracking,
    proporcionando una interfaz unificada para el procesamiento de
    detecciones y actualización de tracks.

    Características:
        - Gestión centralizada de tracks
        - Matching jerárquico
        - Re-identificación robusta
        - Fusión de sensores
        - Predicción de trayectoria
        - Aprendizaje en línea
        - MHT (Multi-Hypothesis Tracking)
        - Gestión automática de memoria

    Attributes:
        config: Configuración del sistema.
        state_manager: Gestor de estado de tracks.
        track_updater: Actualizador de tracks.
        feature_manager: Gestor de features.
        track_matcher: Servicio de matching.
        reid_system: Sistema de re-identificación.
        mht_integration: Sistema MHT.
        online_learner: Aprendizaje en línea.
        sensor_fusion: Fusión de sensores.
        path_predictor: Predicción de trayectoria.
        state_machine: Máquina de estados.

    Example:
        >>> orchestrator = TrackOrchestrator(config)
        >>> frame = cv2.imread("frame.jpg")
        >>> detections = detector.detect(frame)
        >>> tracks = orchestrator.update(detections, frame)
        >>> for track_id, track_data in tracks.items():
        ...     print(f"Track {track_id}: {track_data['centroid']}")
    """

    def __init__(self, config):
        """Inicializa el orquestador de tracking.

        Args:
            config: Configuración del sistema.
        """
        self.config = config
        self.logger.info("Inicializando TrackOrchestrator")

        self._init_state_manager()
        self._init_feature_manager()
        self._init_matcher()
        self._init_advanced_services()
        self._init_state_machine()

        self._frame_counter = 0
        self._tracking_time_ms = 0.0
        self._last_memory_check = time.time()
        self._last_cleanup_time = time.time()

        self._stats = self._init_stats()

        self.logger.info(
            "TrackOrchestrator inicializado",
            features_enabled=self.feature_manager.is_available,
            reid_enabled=self.reid_system is not None,
            mht_enabled=self.mht_integration.enabled,
            active_tracks_limit=self.state_manager.max_active_tracks,
        )

    def _init_state_manager(self) -> None:
        """Inicializa el gestor de estado de tracks."""
        self.state_manager = TrackStateManager(
            max_active_tracks=getattr(self.config, "max_active_tracks", 50)
        )

        use_optimized_kalman = getattr(self.config, "use_optimized_kalman", True)
        self.track_updater = TrackUpdater(
            use_kalman=getattr(self.config, "use_kalman", True),
            use_optimized_kalman=use_optimized_kalman,
            max_speed_change=50.0,
        )

    def _init_feature_manager(self) -> None:
        """Inicializa el gestor de features."""
        use_features = self._should_use_features()
        feature_extractor = None

        if use_features:
            try:
                feature_extractor = FeatureExtractorFactory.create_best_available()
                self.logger.info("Feature extractor activado")
            except Exception as e:
                self.logger.warning("Feature extractor desactivado", error=str(e))

        self.feature_manager = FeatureManager(
            feature_extractor=feature_extractor,
            max_cache_size=getattr(self.config, "reid_cache_size", 1000),
            max_age_seconds=getattr(self.config, "reid_max_age_seconds", 30.0),
            similarity_threshold=getattr(self.config, "reid_similarity_threshold", 0.6),
            spatial_threshold=getattr(self.config, "reid_spatial_threshold", 100.0),
        )

    def _init_matcher(self) -> None:
        """Inicializa el sistema de matching."""
        self.reid_system = self._init_reid_system()

        max_search_radius = getattr(self.config, "max_search_radius", 150.0)

        self.track_matcher = TrackMatcher(
            matcher=None,
            reid_system=self.reid_system,
            iou_threshold=getattr(self.config, "iou_threshold", 0.3),
            feature_threshold=getattr(self.config, "feature_threshold", 0.6),
            spatial_threshold=getattr(self.config, "max_distance", 50.0),
            max_search_radius=max_search_radius,
        )

    def _init_reid_system(self) -> ReIDSystem | None:
        """Inicializa el sistema de re-identificación."""
        if not getattr(self.config, "enable_reidentification", True):
            return None

        if not self.feature_manager.is_available:
            return None

        try:
            reid = ReIDSystem(
                feature_extractor=self.feature_manager.feature_extractor,
                max_cache_size=getattr(self.config, "reid_cache_size", 1000),
                max_age_seconds=getattr(self.config, "reid_max_age_seconds", 30.0),
                similarity_threshold=getattr(self.config, "reid_similarity_threshold", 0.6),
                spatial_threshold=getattr(self.config, "reid_spatial_threshold", 100.0),
                min_features_for_reid=getattr(self.config, "reid_min_features", 3),
            )
            self.logger.info("Sistema de re-identificación activado")
            return reid
        except Exception as e:
            self.logger.warning("Re-identificación desactivada", error=str(e))
            return None

    def _init_advanced_services(self) -> None:
        """Inicializa los servicios avanzados."""
        self.mht_integration = self._init_mht()
        self.online_learner = self._init_online_learner()
        self.sensor_fusion = self._init_sensor_fusion()
        self.path_predictor = self._init_path_predictor()

    def _init_mht(self) -> MHTIntegration:
        """Inicializa el sistema MHT."""
        return MHTIntegration(
            max_depth=getattr(self.config, "mht_max_depth", 10),
            pruning_threshold=getattr(self.config, "mht_pruning_threshold", 0.01),
            max_hypotheses_per_track=getattr(self.config, "mht_max_hypotheses", 5),
            enable_mht=getattr(self.config, "enable_mht", False),
        )

    def _init_online_learner(self) -> OnlineLearner | None:
        """Inicializa el sistema de aprendizaje en línea."""
        if not self.feature_manager.is_available:
            return None

        if not getattr(self.config, "enable_reidentification", True):
            return None

        try:
            learner = OnlineLearner(
                feature_dim=2048,
                learning_rate=getattr(self.config, "online_learning_rate", 0.05),
                min_samples=getattr(self.config, "online_learning_min_samples", 5),
                drift_threshold=getattr(self.config, "online_learning_drift_threshold", 0.35),
                max_history=getattr(self.config, "online_learning_max_history", 50),
                strategy=getattr(self.config, "online_learning_strategy", "adaptive"),
            )
            self.logger.info("Sistema de aprendizaje en línea activado")
            return learner
        except Exception as e:
            self.logger.warning("Aprendizaje en línea desactivado", error=str(e))
            return None

    def _init_sensor_fusion(self) -> SensorFusion | None:
        """Inicializa el sistema de fusión de sensores."""
        if not getattr(self.config, "enable_sensor_fusion", False):
            return None

        try:
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
                particle_count=getattr(self.config, "fusion_particle_count", 500),
            )
            self.logger.info("Sistema de fusión de sensores activado")
            return fusion
        except Exception as e:
            self.logger.warning("Fusión de sensores desactivada", error=str(e))
            return None

    def _init_path_predictor(self) -> PathPredictor | None:
        """Inicializa el sistema de predicción de trayectoria."""
        if not getattr(self.config, "enable_path_prediction", True):
            return None

        try:
            predictor = PathPredictor(
                history_length=getattr(self.config, "prediction_history_length", 30),
                prediction_horizon=getattr(self.config, "prediction_horizon", 2.0),
                prediction_steps=getattr(self.config, "prediction_steps", 20),
                min_samples=getattr(self.config, "prediction_min_samples", 5),
                motion_model=getattr(self.config, "prediction_motion_model", "adaptive"),
                uncertainty_threshold=getattr(self.config, "prediction_uncertainty_threshold", 0.7),
            )
            self.logger.info("Sistema de predicción de trayectoria activado")
            return predictor
        except Exception as e:
            self.logger.warning("Predicción de trayectoria desactivada", error=str(e))
            return None

    def _init_state_machine(self) -> None:
        """Inicializa la máquina de estados para los tracks."""
        self.state_machine = TrackStateMachine(
            min_hits_to_confirm=getattr(self.config, "min_hits_to_confirm", 3),
            max_frames_missed=getattr(self.config, "max_frames_missed", 30),
        )

    def _init_stats(self) -> dict[str, Any]:
        """Inicializa las estadísticas del orquestador."""
        return {
            "total_tracks": 0,
            "confirmed_tracks": 0,
            "lost_tracks": 0,
            "reidentified_tracks": 0,
            "tracking_time_ms": 0,
            "features_used": self.feature_manager.is_available,
        }

    def _should_use_features(self) -> bool:
        """Determina si se deben usar features visuales.

        Returns:
            bool: True si se deben usar features.

        Note:
            Las features se usan solo en GPU para no degradar el rendimiento en CPU.
        """
        device = None

        if hasattr(self.config, "device"):
            device = self.config.device
        elif hasattr(self.config, "model") and hasattr(self.config.model, "device"):
            device = self.config.model.device

        if device == "cpu" or (hasattr(device, "value") and device.value == "cpu"):
            return False

        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def update(
        self, detections: list[dict[str, Any]], frame: np.ndarray
    ) -> dict[int, dict[str, Any]]:
        """Actualiza el tracker con nuevas detecciones.

        Args:
            detections: Lista de detecciones del frame actual.
            frame: Imagen actual para extraer features y contexto.

        Returns:
            Dict[int, Dict[str, Any]]: Información de tracking actualizada.

        Raises:
            TrackingError: Si ocurre un error durante el tracking.

        Note:
            El proceso de actualización incluye:
            1. Validación de detecciones
            2. Extracción de features
            3. Predicción de posición (Kalman)
            4. Matching jerárquico
            5. Actualización de tracks
            6. Re-identificación
            7. Creación de nuevos tracks
            8. Limpieza de tracks muertos
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

        self._update_advanced_services(valid_detections, match_result)

        self._handle_unmatched(match_result)

        self._create_new_tracks(valid_detections, match_result)

        if self.reid_system and match_result.unmatched_detections:
            self._perform_reidentification(
                valid_detections, match_result.unmatched_detections, frame
            )

        self._perform_cleanup()
        self._update_stats()

        self._tracking_time_ms = (time.perf_counter() - start_time) * 1000
        self._stats["tracking_time_ms"] = self._tracking_time_ms

        return self.get_tracking_info()

    def _validate_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Valida y filtra detecciones."""
        if not detections:
            return []

        valid = [
            det for det in detections if validate_detection(det, require_all_fields=True).is_valid
        ]

        return valid[:MAX_DETECTIONS_PER_FRAME]

    def _extract_features(self, detections: list[dict[str, Any]], frame: np.ndarray) -> None:
        """Extrae features para todas las detecciones."""
        for det in detections:
            if "box" in det:
                features = self.feature_manager.extract_features(
                    frame, det["box"], det.get("confidence", 0.5)
                )
                if features is not None:
                    det["features"] = features

    def _predict_positions(self) -> None:
        """Predice posiciones de todos los tracks usando filtro de Kalman."""
        for track in self.state_manager.get_all_tracks().values():
            self.track_updater.predict_position(track)

    def _perform_matching(self, detections: list[dict[str, Any]], frame: np.ndarray) -> Any:
        """Realiza matching entre detecciones y tracks existentes."""
        tracks = list(self.state_manager.get_all_tracks().values())

        if not detections or not tracks:
            return MatchResult(
                matches=[],
                unmatched_detections=list(range(len(detections))),
                unmatched_tracks=list(range(len(tracks))),
                match_scores={},
                reidentified=[],
                time_ms=0.0,
            )

        return self.track_matcher.match(detections, tracks, frame)

    def _update_tracks(self, detections: list[dict[str, Any]], match_result: Any) -> None:
        """Actualiza tracks con nuevas detecciones asociadas."""
        tracks = self.state_manager.get_all_tracks()
        track_ids = list(tracks.keys())

        for det_idx, track_idx in match_result.matches:
            if det_idx >= len(detections) or track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            detection = detections[det_idx]
            features = detection.get("features")

            self._update_single_track(track_id, detection, features, match_result)

    def _update_single_track(
        self,
        track_id: int,
        detection: dict[str, Any],
        features: np.ndarray | None,
        match_result: Any,
    ) -> None:
        """Actualiza un track individual."""
        self.state_manager.update_track(track_id, detection, features)
        track = self.state_manager.get_track(track_id)

        if track:
            self.track_updater.correct_position(track, detection)
            self.track_updater.update_motion_metrics(track)

            new_status = self.state_machine.transition(track.status, track.hits, track.no_losses)
            track.status = new_status

            if features is not None:
                self.feature_manager.cache_features(track_id, features, track.confidence)

            if track_id in match_result.reidentified:
                self.logger.info(
                    "Track recuperado y actualizado", track_id=track_id, confidence=track.confidence
                )

    def _update_advanced_services(
        self, detections: list[dict[str, Any]], match_result: Any
    ) -> None:
        """Actualiza todos los sistemas avanzados."""
        try:
            if self.online_learner:
                self._update_online_learning(detections, match_result)

            if self.sensor_fusion:
                self._update_sensor_fusion(detections, match_result)

            if self.path_predictor:
                self._update_path_prediction()
        except Exception as e:
            self.logger.warning(f"Error en subsistema avanzado: {e}", exc_info=True)

    def _update_online_learning(self, detections: list[dict[str, Any]], match_result: Any) -> None:
        """Actualiza el aprendizaje en línea con nuevas observaciones."""
        if self.online_learner is None:
            return

        tracks = self.state_manager.get_all_tracks()
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
                        confidence=detection.get("confidence", 0.5),
                    )
                except Exception as e:
                    self.logger.debug(
                        "Error en aprendizaje en línea", track_id=track_id, error=str(e)
                    )

    def _update_sensor_fusion(self, detections: list[dict[str, Any]], match_result: Any) -> None:
        """Actualiza la fusión de sensores con nuevas observaciones."""
        if self.sensor_fusion is None:
            return

        tracks = self.state_manager.get_all_tracks()
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
                    },
                )
                self.sensor_fusion.add_observation(track_id, observation)
            except Exception as e:
                self.logger.debug("Error en fusión de sensores", track_id=track_id, error=str(e))

    def _update_path_prediction(self) -> None:
        """Actualiza la predicción de trayectoria para todos los tracks."""
        if self.path_predictor is None:
            return

        for track_id, track in self.state_manager.get_all_tracks().items():
            try:
                prediction = self.path_predictor.update(
                    track_id=track_id,
                    position=track.centroid,
                    velocity=track.velocity,
                    confidence=track.confidence,
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
                    "Error en predicción de trayectoria", track_id=track_id, error=str(e)
                )

    def _handle_unmatched(self, match_result: Any) -> None:
        """Maneja tracks no asociados (pérdidas)."""
        tracks = self.state_manager.get_all_tracks()
        track_ids = list(tracks.keys())

        for track_idx in match_result.unmatched_tracks:
            if track_idx >= len(track_ids):
                continue

            track_id = track_ids[track_idx]
            track = self.state_manager.get_track(track_id)

            if track:
                track.mark_lost()

                new_status = self.state_machine.transition(
                    track.status, track.hits, track.no_losses
                )
                track.status = new_status

                if track.status == TrackStatus.DEAD:
                    self._handle_dead_track(track_id)

    def _handle_dead_track(self, track_id: int) -> None:
        """Maneja un track que ha muerto (no recuperable)."""
        track = self.state_manager.get_track(track_id)
        if track is None:
            return

        self.state_manager.mark_as_lost(track_id)

        if self.reid_system and track.features is not None:
            self.reid_system.add_lost_track(track_id, track.features, track.confidence)

        if self.online_learner:
            self.online_learner.clear_track(track_id)
        if self.sensor_fusion:
            self.sensor_fusion.clear_track(track_id)
        if self.path_predictor:
            self.path_predictor.clear_track(track_id)

    def _create_new_tracks(self, detections: list[dict[str, Any]], match_result: Any) -> None:
        """Crea nuevos tracks a partir de detecciones no asociadas."""
        tracks_created = 0

        for det_idx in match_result.unmatched_detections:
            if det_idx >= len(detections):
                continue

            detection = detections[det_idx]

            if not self._is_valid_new_track(detection):
                continue

            features = detection.get("features")
            track = self.state_manager.create_track(detection=detection, features=features)

            if track:
                self.track_updater.init_kalman(track)
                self._init_advanced_features_for_track(
                    track, detection, detection.get("confidence", 0.5), features
                )
                tracks_created += 1

        if tracks_created > 0:
            self.logger.debug(
                "Nuevos tracks creados",
                count=tracks_created,
                active=self.state_manager.get_active_count(),
            )

    def _is_valid_new_track(self, detection: dict[str, Any]) -> bool:
        """Verifica si una detección es válida para crear un nuevo track."""
        confidence = detection.get("confidence", 0.0)
        if confidence < 0.3:
            return False

        return bool(detection.get("box") and detection.get("centroid"))

    def _init_advanced_features_for_track(
        self,
        track: TrackState,
        detection: dict[str, Any],
        confidence: float,
        features: np.ndarray | None,
    ) -> None:
        """Inicializa características avanzadas para un nuevo track."""
        track_id = track.track_id

        if self.online_learner and features is not None:
            try:
                self.online_learner.update(
                    track_id=track_id, features=features, confidence=confidence
                )
            except Exception as e:
                self.logger.debug(
                    "Error iniciando aprendizaje en línea", track_id=track_id, error=str(e)
                )

        if self.sensor_fusion:
            try:
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
                    },
                )
                self.sensor_fusion.add_observation(track_id, observation)
            except Exception as e:
                self.logger.debug(
                    "Error iniciando fusión de sensores", track_id=track_id, error=str(e)
                )

        if self.path_predictor:
            try:
                self.path_predictor.update(
                    track_id=track_id, position=track.centroid, confidence=confidence
                )
            except Exception as e:
                self.logger.debug(
                    "Error iniciando predicción de trayectoria", track_id=track_id, error=str(e)
                )

    def _perform_reidentification(
        self,
        detections: list[dict[str, Any]],
        unmatched_dets: list[int],
        frame: np.ndarray,
    ) -> int:
        """Realiza re-identificación de objetos perdidos."""
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
                current_tracks=self.state_manager.get_all_tracks(),
            )

            if track_id is not None and self._recover_track(track_id, detection):
                reidentified += 1

        return reidentified

    def _recover_track(self, track_id: int, detection: dict[str, Any]) -> bool:
        """Recupera un track re-identificado."""
        track = self.state_manager.recover_track(track_id)
        if not track:
            return False

        track.update(detection, detection.get("features"))
        track.status = TrackStatus.CONFIRMED
        track.no_losses = 0

        self.track_updater.init_kalman(track)
        self._stats["reidentified_tracks"] += 1

        self.logger.info("Track re-identificado", track_id=track_id, confidence=track.confidence)

        return True

    def _check_memory(self) -> None:
        """Verifica el uso de memoria y limpia si es necesario."""
        current_time = time.time()
        if current_time - self._last_memory_check < MEMORY_CHECK_INTERVAL:
            return

        self._last_memory_check = current_time

        try:
            mem = get_memory_usage()
            mem_percent = mem.get("percent", 0)

            if mem_percent > MEMORY_WARNING_PERCENT:
                self.logger.warning(
                    "Memoria alta, limpiando",
                    memory_percent=f"{mem_percent:.1f}",
                    active_tracks=self.state_manager.get_active_count(),
                )
                self.feature_manager.clear_cache()
                force_garbage_collection()
        except Exception as e:
            self.logger.debug("Error verificando memoria", error=str(e))

    def _perform_cleanup(self) -> None:
        """Realiza limpieza periódica de tracks muertos."""
        current_time = time.time()
        if current_time - self._last_cleanup_time >= CLEANUP_INTERVAL:
            self._last_cleanup_time = current_time
            removed = self.state_manager.cleanup_dead_tracks()

            if removed > 0:
                self.logger.debug(
                    "Limpieza de tracks completada",
                    removed=removed,
                    active=self.state_manager.get_active_count(),
                    lost=self.state_manager.get_lost_count(),
                )

    def _update_stats(self) -> None:
        """Actualiza estadísticas del tracker."""
        self._stats["total_tracks"] = self.state_manager.get_active_count()
        self._stats["confirmed_tracks"] = sum(
            1
            for t in self.state_manager.get_all_tracks().values()
            if t.status == TrackStatus.CONFIRMED
        )
        self._stats["lost_tracks"] = self.state_manager.get_lost_count()

    def get_tracking_info(self) -> dict[int, dict[str, Any]]:
        """Retorna información de tracking actual."""
        result = {}

        for track_id, track in self.state_manager.get_all_tracks().items():
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

    def _enrich_track_data(self, track_id: int, track_data: dict[str, Any]) -> None:
        """Enriquece los datos del track con información de subsistemas."""
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

    def get_track(self, track_id: int) -> TrackState | None:
        """Obtiene un track por su ID."""
        return self.state_manager.get_track(track_id)

    def get_stats(self) -> dict[str, Any]:
        """Retorna estadísticas del tracker."""
        return {
            **self._stats,
            "active_tracks": self.state_manager.get_active_count(),
            "lost_tracks_count": self.state_manager.get_lost_count(),
            "feature_manager": self.feature_manager.get_stats(),
            "state_machine": self.state_machine.get_stats(),
            "track_updater": self.track_updater.get_stats(),
            "track_matcher": self.track_matcher.get_stats(),
            "tracking_time_ms": self._tracking_time_ms,
            "frame_counter": self._frame_counter,
        }

    def reset(self) -> None:
        """Reinicia el tracker completamente."""
        self.logger.info("Reiniciando tracker")

        self.state_manager.clear_all()
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
