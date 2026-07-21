"""
Sistema de fusión de sensores para tracking multi-modal robusto.

Este módulo implementa un sistema de fusión de múltiples fuentes de información
para mejorar la robustez del tracking, combinando observaciones visuales,
de profundidad, térmicas y de movimiento.

La fusión de sensores permite:
- Mayor robustez en condiciones adversas (poca luz, oclusión)
- Mejor precisión en la estimación de posición
- Reducción de falsos positivos
- Tracking continuo incluso cuando un sensor falla
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import field
from collections import deque

import numpy as np

from utils.logger import LoggerMixin
from core.constants import (
    SENSOR_FUSION_VISUAL_WEIGHT,
    SENSOR_FUSION_DEPTH_WEIGHT,
    SENSOR_FUSION_THERMAL_WEIGHT,
    SENSOR_FUSION_MOTION_WEIGHT,
    SENSOR_FUSION_MIN_OBSERVATIONS,
    SENSOR_FUSION_MAX_HISTORY,
    SENSOR_FUSION_PARTICLE_COUNT,
)


class SensorType(Enum):
    """
    Tipos de sensores soportados por el sistema de fusión.

    Attributes:
        VISUAL: Cámara RGB estándar
        DEPTH: Sensor de profundidad (ej: Kinect, Intel RealSense)
        THERMAL: Cámara térmica (infrarroja)
        MOTION: Sensor de movimiento (ej: radar Doppler)
        RADAR: Sensor de radar
        LIDAR: Sensor LIDAR
        GPS: Sistema de posicionamiento global
    """
    VISUAL = "visual"
    DEPTH = "depth"
    THERMAL = "thermal"
    MOTION = "motion"
    RADAR = "radar"
    LIDAR = "lidar"
    GPS = "gps"


class SensorFusionMethod(Enum):
    """
    Métodos de fusión de sensores disponibles.

    Attributes:
        WEIGHTED_AVERAGE: Promedio ponderado por confianza
        KALMAN: Filtro de Kalman extendido
        PARTICLE_FILTER: Filtro de partículas
        BAYESIAN: Fusión bayesiana
        DEMPSTER_SHAFER: Teoría de evidencia de Dempster-Shafer
    """
    WEIGHTED_AVERAGE = "weighted_average"
    KALMAN = "kalman"
    PARTICLE_FILTER = "particle_filter"
    BAYESIAN = "bayesian"
    DEMPSTER_SHAFER = "dempster_shafer"


class SensorObservation:
    """
    Representa una observación de un sensor.

    Attributes:
        sensor_type: Tipo de sensor que generó la observación.
        bbox: Bounding box de la observación (x1, y1, x2, y2).
        centroid: Centroide de la observación (x, y).
        confidence: Confianza de la observación (0-1).
        timestamp: Timestamp de la observación.
        metadata: Metadatos adicionales de la observación.
        track_id: ID del track asociado (si existe).
        sensor_id: Identificador único del sensor.
        calibration_matrix: Matriz de calibración para este sensor.
    """
    __slots__ = (
        'sensor_type', 'bbox', 'centroid', 'confidence', 'timestamp',
        'metadata', 'track_id', 'sensor_id', 'calibration_matrix'
    )

    def __init__(
        self,
        sensor_type: SensorType,
        bbox: Tuple[int, int, int, int],
        centroid: Tuple[int, int],
        confidence: float,
        timestamp: float = None,
        metadata: Dict[str, Any] = None,
        track_id: Optional[int] = None,
        sensor_id: str = "",
        calibration_matrix: Optional[np.ndarray] = None
    ):
        self.sensor_type = sensor_type
        self.bbox = bbox
        self.centroid = centroid
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        self.track_id = track_id
        self.sensor_id = sensor_id
        self.calibration_matrix = calibration_matrix


class FusedState:
    """
    Estado fusionado de un track.

    Attributes:
        track_id: ID del track.
        centroid: Centroide fusionado (x, y).
        bbox: Bounding box fusionado.
        confidence: Confianza fusionada (0-1).
        velocity: Velocidad fusionada (vx, vy).
        uncertainty: Incertidumbre del estado fusionado.
        timestamp: Timestamp de la última fusión.
        sensor_contributions: Contribución de cada sensor.
        history: Historial de estados fusionados.
        method: Método de fusión utilizado.
    """
    __slots__ = (
        'track_id', 'centroid', 'bbox', 'confidence', 'velocity',
        'uncertainty', 'timestamp', 'sensor_contributions', 'history', 'method'
    )

    def __init__(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        confidence: float,
        velocity: Tuple[float, float] = (0.0, 0.0),
        uncertainty: float = 0.0,
        timestamp: float = None,
        sensor_contributions: Dict[SensorType, float] = None,
        history: deque = None,
        method: str = "weighted_average"
    ):
        self.track_id = track_id
        self.centroid = centroid
        self.bbox = bbox
        self.confidence = confidence
        self.velocity = velocity
        self.uncertainty = uncertainty
        self.timestamp = timestamp or time.time()
        self.sensor_contributions = sensor_contributions or {}
        self.history = history or deque(maxlen=50)
        self.method = method


class ParticleFilter(LoggerMixin):
    """
    Filtro de partículas para fusión de sensores.

    Implementa un filtro de partículas para estimar el estado de un objeto
    combinando observaciones de múltiples sensores.

    Características:
        - Estimación no lineal de estado
        - Manejo de múltiples hipótesis
        - Robustez a ruido no gaussiano
        - Resampling adaptativo

    Attributes:
        num_particles: Número de partículas en el filtro.
        process_noise: Ruido del proceso (movimiento).
        measurement_noise: Ruido de la medición.
        resampling_threshold: Umbral para resampleo.
        particles: Estado de las partículas.
        weights: Pesos de las partículas.
        _initialized: Flag de inicialización.

    Example:
        >>> pf = ParticleFilter(num_particles=500)
        >>> pf.init((320, 240))
        >>> pf.predict(dt=0.1)
        >>> pf.update([observation1, observation2])
        >>> state = pf.get_estimate()
    """

    def __init__(
        self,
        num_particles: int = 500,
        process_noise: float = 0.1,
        measurement_noise: float = 0.3,
        resampling_threshold: float = 0.5
    ) -> None:
        """
        Inicializa el filtro de partículas.

        Args:
            num_particles: Número de partículas (mayor = más preciso pero más lento).
            process_noise: Ruido del proceso (movimiento).
            measurement_noise: Ruido de la medición.
            resampling_threshold: Umbral para resampleo (0-1).
        """
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.resampling_threshold = resampling_threshold

        self.particles: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self._initialized = False

        self.logger.info(
            "ParticleFilter inicializado",
            num_particles=num_particles,
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )

    def init(self, centroid: Tuple[float, float], spread: float = 50.0) -> None:
        """
        Inicializa las partículas alrededor de una posición.

        Args:
            centroid: Posición inicial (x, y).
            spread: Dispersión inicial de las partículas en píxeles.
        """
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float32)
        self.particles[:, 0] = centroid[0] + np.random.randn(self.num_particles) * spread
        self.particles[:, 1] = centroid[1] + np.random.randn(self.num_particles) * spread
        self.particles[:, 2] = np.random.randn(self.num_particles) * 2.0
        self.particles[:, 3] = np.random.randn(self.num_particles) * 2.0

        self.weights = np.ones(self.num_particles) / self.num_particles
        self._initialized = True

    def predict(self, dt: float = 0.1) -> None:
        """
        Predice el siguiente estado de las partículas.

        Args:
            dt: Delta de tiempo para la predicción en segundos.
        """
        if not self._initialized or self.particles is None:
            return

        self.particles[:, 0] += self.particles[:, 2] * dt + np.random.randn(self.num_particles) * self.process_noise
        self.particles[:, 1] += self.particles[:, 3] * dt + np.random.randn(self.num_particles) * self.process_noise
        self.particles[:, 2] += np.random.randn(self.num_particles) * self.process_noise * 0.1
        self.particles[:, 3] += np.random.randn(self.num_particles) * self.process_noise * 0.1

    def update(self, observations: List[SensorObservation]) -> None:
        """
        Actualiza los pesos de las partículas con las observaciones.

        Args:
            observations: Lista de observaciones de sensores.

        Note:
            Cada observación contribuye al peso de las partículas
            según su distancia y confianza.
        """
        if not self._initialized or self.particles is None or not observations:
            return

        for obs in observations:
            obs_centroid = np.array(obs.centroid)
            obs_confidence = obs.confidence

            distances = np.sqrt(
                (self.particles[:, 0] - obs_centroid[0]) ** 2 +
                (self.particles[:, 1] - obs_centroid[1]) ** 2
            )

            likelihood = np.exp(-distances ** 2 / (2 * self.measurement_noise ** 2))
            likelihood *= obs_confidence

            self.weights *= likelihood + 1e-8

        self.weights /= np.sum(self.weights) + 1e-8

        if self._needs_resampling():
            self._resample()

    def _needs_resampling(self) -> bool:
        """
        Verifica si es necesario resamplear las partículas.

        Returns:
            bool: True si se debe resamplear.

        Note:
            Usa el número efectivo de partículas (N_eff) para decidir.
        """
        if self.weights is None:
            return True

        n_eff = 1.0 / np.sum(self.weights ** 2 + 1e-8)
        return n_eff < self.num_particles * self.resampling_threshold

    def _resample(self) -> None:
        """
        Resamplea las partículas usando el método de muestreo sistemático.
        """
        if self.particles is None or self.weights is None:
            return

        n = self.num_particles
        indices = np.zeros(n, dtype=int)

        cumulative = np.cumsum(self.weights)
        step = 1.0 / n
        u = np.random.rand() * step

        j = 0
        for i in range(n):
            while u > cumulative[j]:
                j += 1
            indices[i] = j
            u += step

        self.particles = self.particles[indices]
        self.weights = np.ones(n) / n

    def get_estimate(self) -> Dict[str, Any]:
        """
        Obtiene la estimación del estado.

        Returns:
            Dict[str, Any]: Estimación del estado incluyendo:
                - centroid: Posición estimada (x, y)
                - velocity: Velocidad estimada (vx, vy)
                - uncertainty: Incertidumbre de la estimación
                - confidence: Confianza de la estimación
        """
        if not self._initialized or self.particles is None or self.weights is None:
            return {
                "centroid": (0.0, 0.0),
                "velocity": (0.0, 0.0),
                "uncertainty": 1.0,
                "confidence": 0.0,
            }

        weighted_centroid = np.average(self.particles[:, :2], weights=self.weights, axis=0)
        weighted_velocity = np.average(self.particles[:, 2:], weights=self.weights, axis=0)

        centered = self.particles[:, :2] - weighted_centroid
        covariance = np.average(
            centered[:, :, None] * centered[:, None, :],
            weights=self.weights,
            axis=0
        )
        uncertainty = np.trace(covariance)

        return {
            "centroid": (float(weighted_centroid[0]), float(weighted_centroid[1])),
            "velocity": (float(weighted_velocity[0]), float(weighted_velocity[1])),
            "uncertainty": float(uncertainty),
            "confidence": float(np.sum(self.weights * (1 / (1 + uncertainty)))),
        }

    @property
    def is_initialized(self) -> bool:
        """Verifica si el filtro está inicializado."""
        return self._initialized


class SensorFusionTracker(LoggerMixin):
    """
    Sistema de fusión de sensores para tracking multi-modal robusto.

    Este sistema combina observaciones de múltiples sensores para obtener
    un tracking más robusto y preciso, especialmente en condiciones adversas.

    Características:
        - Fusión de observaciones visuales, de profundidad y térmicas
        - Filtro de partículas para fusión robusta
        - Ponderación adaptativa por fiabilidad de cada sensor
        - Gestión de calibración entre sensores
        - Historial de estados fusionados
        - Estadísticas de rendimiento

    Attributes:
        sensor_weights: Pesos de confianza por tipo de sensor.
        fusion_method: Método de fusión utilizado.
        min_observations: Mínimo de observaciones para fusión.
        max_history: Tamaño máximo del historial.
        _states: Diccionario de estados fusionados por track_id.
        _particle_filters: Diccionario de filtros de partículas.
        _calibration_matrices: Matrices de calibración por sensor.

    Example:
        >>> fusion = SensorFusionTracker(
        ...     sensor_weights={SensorType.VISUAL: 0.7, SensorType.DEPTH: 0.5},
        ...     fusion_method="particle_filter"
        ... )
        >>> obs = SensorObservation(SensorType.VISUAL, bbox, centroid, 0.9)
        >>> fusion.add_observation(5, obs)
        >>> state = fusion.get_fused_state(5)
        >>> print(f"Posición fusionada: {state.centroid}")
    """

    def __init__(
        self,
        sensor_weights: Optional[Dict[SensorType, float]] = None,
        fusion_method: str = "weighted_average",
        min_observations: int = SENSOR_FUSION_MIN_OBSERVATIONS,
        max_history: int = SENSOR_FUSION_MAX_HISTORY,
        particle_count: int = SENSOR_FUSION_PARTICLE_COUNT,
    ) -> None:
        """
        Inicializa el sistema de fusión de sensores.

        Args:
            sensor_weights: Pesos de confianza por tipo de sensor.
            fusion_method: Método de fusión ('weighted_average', 'particle_filter').
            min_observations: Mínimo de observaciones para fusión.
            max_history: Tamaño máximo del historial.
            particle_count: Número de partículas (para particle_filter).
        """
        self.sensor_weights = sensor_weights or {
            SensorType.VISUAL: SENSOR_FUSION_VISUAL_WEIGHT,
            SensorType.DEPTH: SENSOR_FUSION_DEPTH_WEIGHT,
            SensorType.THERMAL: SENSOR_FUSION_THERMAL_WEIGHT,
            SensorType.MOTION: SENSOR_FUSION_MOTION_WEIGHT,
            SensorType.RADAR: 0.6,
            SensorType.LIDAR: 0.6,
            SensorType.GPS: 0.3,
        }

        self.fusion_method = SensorFusionMethod(fusion_method)
        self.min_observations = min_observations
        self.max_history = max_history
        self.particle_count = particle_count

        self._states: Dict[int, FusedState] = {}
        self._particle_filters: Dict[int, ParticleFilter] = {}
        self._observations: Dict[int, List[SensorObservation]] = {}
        self._calibration_matrices: Dict[str, np.ndarray] = {}
        self._recently_fused: Set[int] = set()

        self._stats = {
            "total_fusions": 0,
            "total_observations": 0,
            "total_tracks": 0,
            "active_tracks": 0,
            "avg_fusion_time_ms": 0.0,
            "fusion_times": deque(maxlen=100),
            "sensor_distribution": {sensor_type.value: 0 for sensor_type in SensorType},
            "method": self.fusion_method.value,
            "calibrated_sensors": set(),
        }

        self.logger.info(
            "SensorFusionTracker inicializado",
            fusion_method=self.fusion_method.value,
            min_observations=min_observations,
            particle_count=particle_count,
            sensor_weights=len(self.sensor_weights)
        )

    def add_observation(
        self,
        track_id: int,
        observation: SensorObservation
    ) -> None:
        """
        Añade una observación de un sensor para un track.

        Args:
            track_id: ID del track.
            observation: Observación del sensor.

        Note:
            Las observaciones se acumulan hasta tener suficientes
            para realizar la fusión.
        """
        if track_id not in self._observations:
            self._observations[track_id] = []
            self._states[track_id] = FusedState(
                track_id=track_id,
                centroid=(0.0, 0.0),
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                method=self.fusion_method.value
            )

        self._observations[track_id].append(observation)
        self._stats["total_observations"] += 1
        self._stats["sensor_distribution"][observation.sensor_type.value] += 1

        if len(self._observations[track_id]) > 50:
            self._observations[track_id] = self._observations[track_id][-50:]

        if self.fusion_method == SensorFusionMethod.PARTICLE_FILTER:
            if track_id not in self._particle_filters:
                self._particle_filters[track_id] = ParticleFilter(
                    num_particles=self.particle_count
                )
                self._particle_filters[track_id].init(observation.centroid)

        self._fuse_observations(track_id)

    def _fuse_observations(self, track_id: int) -> None:
        """
        Fusiona observaciones de múltiples sensores para un track.

        Args:
            track_id: ID del track.

        Note:
            Realiza la fusión según el método configurado y actualiza
            el estado fusionado del track.
        """
        import time
        start_time = time.perf_counter()

        observations = self._observations.get(track_id, [])
        if len(observations) < self.min_observations:
            return

        recent_obs = observations[-10:]

        if self.fusion_method == SensorFusionMethod.PARTICLE_FILTER:
            fused_state = self._fuse_with_particle_filter(track_id, recent_obs)
        else:
            fused_state = self._fuse_weighted_average(track_id, recent_obs)

        if fused_state is not None:
            self._states[track_id] = fused_state
            self._states[track_id].history.append(fused_state)

            self._stats["total_fusions"] += 1
            self._stats["total_tracks"] = max(
                self._stats["total_tracks"],
                len(self._states)
            )
            self._stats["active_tracks"] = len(self._states)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._stats["fusion_times"].append(elapsed_ms)
            self._stats["avg_fusion_time_ms"] = (
                sum(self._stats["fusion_times"]) / len(self._stats["fusion_times"])
            )

            self._recently_fused.add(track_id)

            self.logger.debug(
                "Fusión de sensores completada",
                track_id=track_id,
                confidence=fused_state.confidence,
                method=self.fusion_method.value,
                sensors=len(fused_state.sensor_contributions),
                time_ms=f"{elapsed_ms:.1f}"
            )

    def _fuse_weighted_average(
        self,
        track_id: int,
        observations: List[SensorObservation]
    ) -> Optional[FusedState]:
        """
        Fusión por promedio ponderado de observaciones.

        Args:
            track_id: ID del track.
            observations: Lista de observaciones.

        Returns:
            Optional[FusedState]: Estado fusionado o None.

        Note:
            Cada observación contribuye según su confianza y el peso
            de su sensor.
        """
        if not observations:
            return None

        total_weight = 0.0
        weighted_centroid = np.zeros(2)
        weighted_bbox = np.zeros(4)
        total_confidence = 0.0
        sensor_contributions = {}

        for obs in observations:
            weight = self.sensor_weights.get(obs.sensor_type, 0.1)
            weight *= obs.confidence

            if weight < 0.01:
                continue

            sensor_contributions[obs.sensor_type] = (
                sensor_contributions.get(obs.sensor_type, 0.0) + weight
            )

            weighted_centroid += weight * np.array(obs.centroid)
            weighted_bbox += weight * np.array(obs.bbox)
            total_confidence += weight * obs.confidence
            total_weight += weight

        if total_weight < 0.01:
            return None

        weighted_centroid /= total_weight
        weighted_bbox /= total_weight
        fused_confidence = total_confidence / len(observations)

        for sensor_type in sensor_contributions:
            sensor_contributions[sensor_type] /= total_weight

        return FusedState(
            track_id=track_id,
            centroid=tuple(weighted_centroid.astype(int)),
            bbox=tuple(weighted_bbox.astype(int)),
            confidence=fused_confidence,
            sensor_contributions=sensor_contributions,
            timestamp=time.time(),
            method="weighted_average"
        )

    def _fuse_with_particle_filter(
        self,
        track_id: int,
        observations: List[SensorObservation]
    ) -> Optional[FusedState]:
        """
        Fusión usando filtro de partículas.

        Args:
            track_id: ID del track.
            observations: Lista de observaciones.

        Returns:
            Optional[FusedState]: Estado fusionado o None.
        """
        if track_id not in self._particle_filters:
            return None

        pf = self._particle_filters[track_id]

        pf.predict(dt=0.1)
        pf.update(observations)

        estimate = pf.get_estimate()
        if estimate["confidence"] < 0.1:
            return None

        sensor_contributions = {}
        for obs in observations[:5]:
            sensor_contributions[obs.sensor_type] = (
                sensor_contributions.get(obs.sensor_type, 0.0) + obs.confidence
            )

        for sensor_type in sensor_contributions:
            sensor_contributions[sensor_type] /= len(observations)

        return FusedState(
            track_id=track_id,
            centroid=estimate["centroid"],
            bbox=(0, 0, 0, 0),
            confidence=estimate["confidence"],
            velocity=estimate["velocity"],
            uncertainty=estimate["uncertainty"],
            sensor_contributions=sensor_contributions,
            timestamp=time.time(),
            method="particle_filter"
        )

    def get_fused_state(self, track_id: int) -> Optional[FusedState]:
        """
        Obtiene el estado fusionado de un track.

        Args:
            track_id: ID del track.

        Returns:
            Optional[FusedState]: Estado fusionado o None.
        """
        return self._states.get(track_id)

    def get_sensor_contribution(self, track_id: int) -> Dict[SensorType, float]:
        """
        Obtiene la contribución de cada sensor al estado fusionado.

        Args:
            track_id: ID del track.

        Returns:
            Dict[SensorType, float]: Contribuciones de sensores.
        """
        state = self._states.get(track_id)
        if state is None:
            return {}
        return state.sensor_contributions

    def get_track_uncertainty(self, track_id: int) -> float:
        """
        Obtiene la incertidumbre del estado fusionado.

        Args:
            track_id: ID del track.

        Returns:
            float: Incertidumbre (0-1, donde 1 es máxima).
        """
        state = self._states.get(track_id)
        if state is None:
            return 1.0
        return state.uncertainty

    def calibrate_sensor(
        self,
        sensor_id: str,
        calibration_matrix: np.ndarray
    ) -> None:
        """
        Calibra un sensor con una matriz de transformación.

        Args:
            sensor_id: Identificador del sensor.
            calibration_matrix: Matriz de calibración (3x3 homográfica).

        Note:
            La matriz de calibración se aplica a las observaciones
            para alinearlas con el sistema de coordenadas global.
        """
        self._calibration_matrices[sensor_id] = calibration_matrix
        self._stats["calibrated_sensors"].add(sensor_id)
        self.logger.info(
            "Sensor calibrado",
            sensor_id=sensor_id,
            matrix_shape=calibration_matrix.shape
        )

    def transform_observation(
        self,
        observation: SensorObservation
    ) -> SensorObservation:
        """
        Transforma una observación usando la calibración del sensor.

        Args:
            observation: Observación a transformar.

        Returns:
            SensorObservation: Observación transformada.
        """
        if observation.sensor_id not in self._calibration_matrices:
            return observation

        matrix = self._calibration_matrices[observation.sensor_id]
        centroid = np.array(observation.centroid + (1.0,))
        transformed = matrix @ centroid
        transformed = transformed / transformed[2]

        return SensorObservation(
            sensor_type=observation.sensor_type,
            bbox=observation.bbox,
            centroid=(int(transformed[0]), int(transformed[1])),
            confidence=observation.confidence,
            timestamp=observation.timestamp,
            metadata=observation.metadata,
            track_id=observation.track_id,
            sensor_id=observation.sensor_id,
            calibration_matrix=matrix
        )

    def get_sensor_health(self) -> Dict[SensorType, float]:
        """
        Obtiene la salud de cada sensor basado en observaciones recientes.

        Returns:
            Dict[SensorType, float]: Puntuación de salud por sensor (0-1).
        """
        health = {}

        for sensor_type in SensorType:
            observations = []
            for obs_list in self._observations.values():
                for obs in obs_list:
                    if obs.sensor_type == sensor_type:
                        observations.append(obs)

            if not observations:
                health[sensor_type] = 0.0
                continue

            avg_confidence = np.mean([o.confidence for o in observations])
            count_factor = min(1.0, len(observations) / 100.0)

            health[sensor_type] = 0.5 * avg_confidence + 0.5 * count_factor

        return health

    def clear_track(self, track_id: int) -> None:
        """
        Elimina todas las observaciones de un track.

        Args:
            track_id: ID del track.
        """
        if track_id in self._observations:
            del self._observations[track_id]
        if track_id in self._states:
            del self._states[track_id]
        if track_id in self._particle_filters:
            del self._particle_filters[track_id]
        if track_id in self._recently_fused:
            self._recently_fused.remove(track_id)

        self._stats["active_tracks"] = len(self._states)

    def clear_all(self) -> None:
        """Limpia todas las observaciones y estados."""
        self._observations.clear()
        self._states.clear()
        self._particle_filters.clear()
        self._recently_fused.clear()

        self._stats["active_tracks"] = 0
        self._stats["total_tracks"] = 0

        self.logger.info("SensorFusionTracker limpiado")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema de fusión.

        Returns:
            Dict[str, Any]: Estadísticas del sistema.
        """
        return {
            **self._stats,
            "active_tracks": len(self._states),
            "total_tracks": len(self._states),
            "observations_per_track": {
                tid: len(obs_list)
                for tid, obs_list in list(self._observations.items())[:10]
            },
            "calibrated_sensors_count": len(self._stats["calibrated_sensors"]),
            "sensor_health": self.get_sensor_health(),
            "particle_filters_active": len(self._particle_filters),
            "recently_fused": len(self._recently_fused),
        }

    def get_state_history(self, track_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de estados fusionados de un track.

        Args:
            track_id: ID del track.
            limit: Número máximo de estados a retornar.

        Returns:
            List[Dict[str, Any]]: Historial de estados.
        """
        state = self._states.get(track_id)
        if state is None:
            return []

        history = list(state.history)
        return [
            {
                "centroid": s.centroid,
                "confidence": s.confidence,
                "timestamp": s.timestamp,
                "uncertainty": s.uncertainty,
            }
            for s in history[-limit:]
        ]

    def update_sensor_weight(
        self,
        sensor_type: SensorType,
        weight: float
    ) -> None:
        """
        Actualiza el peso de un sensor.

        Args:
            sensor_type: Tipo de sensor.
            weight: Nuevo peso (0-1).
        """
        self.sensor_weights[sensor_type] = max(0.0, min(1.0, weight))
        self.logger.info(
            "Peso de sensor actualizado",
            sensor_type=sensor_type.value,
            weight=weight
        )

    def get_recommended_weights(self) -> Dict[SensorType, float]:
        """
        Calcula pesos recomendados basados en el rendimiento de los sensores.

        Returns:
            Dict[SensorType, float]: Pesos recomendados.
        """
        health = self.get_sensor_health()
        total_health = sum(health.values()) + 0.001

        recommended = {}
        for sensor_type, score in health.items():
            recommended[sensor_type] = score / total_health

        return recommended
