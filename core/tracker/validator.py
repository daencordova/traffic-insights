"""Sistema de validación de tracks para asegurar calidad.

Proporciona reglas de validación para verificar la consistencia
de los tracks y filtrar aquellos que no cumplen con los criterios
de calidad establecidos.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from core.constants.tracking import (
    TRACK_VALIDATION_MAX_SPEED_CHANGE,
    TRACK_VALIDATION_MIN_CONFIDENCE,
)
from utils.geometry import euclidean_distance
from utils.logger import LoggerMixin


class ValidationSeverity(Enum):
    """Severidad de las violaciones de validación.

    Attributes:
        WARNING: Problema menor que no impide el tracking.
        ERROR: Problema que afecta la calidad del track.
        CRITICAL: Problema grave que invalida el track.
    """

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Resultado de una validación.

    Attributes:
        passed: Si la validación fue exitosa.
        violations: Lista de violaciones encontradas.
        score: Puntuación de calidad (0-1).
        details: Información adicional del resultado.

    Example:
        >>> result = validator.validate(track)
        >>> if not result.passed:
        ...     for violation in result.violations:
        ...         print(f"{violation['rule']}: {violation['severity']}")
    """

    passed: bool
    violations: list[dict[str, Any]] = field(default_factory=list)
    score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


class ValidationRule:
    """Regla de validación para tracks.

    Attributes:
        name: Nombre identificador de la regla.
        check_func: Función que realiza la validación.
        severity: Severidad de la violación.
        threshold: Umbral para considerar la regla como superada.

    Example:
        >>> rule = ValidationRule(
        ...     name="motion_consistency",
        ...     check_func=check_motion,
        ...     severity=ValidationSeverity.WARNING
        ... )
    """

    def __init__(
        self,
        name: str,
        check_func: Callable,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
        threshold: float = 0.5,
    ):
        self.name = name
        self.check_func = check_func
        self.severity = severity
        self.threshold = threshold

    def validate(self, track) -> tuple[bool, float, dict[str, Any]]:
        """Valida un track contra esta regla.

        Args:
            track: TrackState a validar.

        Returns:
            Tuple[bool, float, Dict[str, Any]]:
                - passed: Si la regla se cumple
                - score: Puntuación de cumplimiento (0-1)
                - details: Detalles de la validación
        """
        try:
            passed, score, details = self.check_func(track)
            return passed, score, details
        except Exception as e:
            return False, 0.0, {"error": str(e)}


class TrackValidator(LoggerMixin):
    """Validador de tracks con múltiples reglas y puntuación.

    Reglas de validación:
        1. Consistencia de movimiento (velocidad no cambia abruptamente)
        2. Suavidad de trayectoria (sin zigzags extremos)
        3. Consistencia de forma (aspect ratio estable)
        4. Verificación de posición (dentro de límites)
        5. Filtro de confianza

    Attributes:
        min_confidence: Confianza mínima requerida.
        max_speed_change: Cambio máximo de velocidad permitido.

    Example:
        >>> validator = TrackValidator(min_confidence=0.3)
        >>> result = validator.validate(track)
        >>> if result.passed:
        ...     print(f"Track válido con score: {result.score:.2f}")
    """

    def __init__(
        self,
        min_confidence: float = TRACK_VALIDATION_MIN_CONFIDENCE,
        max_speed_change: float = TRACK_VALIDATION_MAX_SPEED_CHANGE,
    ):
        self.min_confidence = min_confidence
        self.max_speed_change = max_speed_change

        self.rules: list[ValidationRule] = [
            ValidationRule(
                "motion_consistency",
                self._check_motion_consistency,
                ValidationSeverity.WARNING,
                threshold=0.6,
            ),
            ValidationRule(
                "trajectory_smoothness",
                self._check_trajectory_smoothness,
                ValidationSeverity.WARNING,
                threshold=0.5,
            ),
            ValidationRule(
                "shape_consistency",
                self._check_shape_consistency,
                ValidationSeverity.WARNING,
                threshold=0.4,
            ),
            ValidationRule(
                "position_validity",
                self._check_position_validity,
                ValidationSeverity.ERROR,
                threshold=0.8,
            ),
            ValidationRule(
                "confidence_filter",
                self._check_confidence,
                ValidationSeverity.WARNING,
                threshold=0.7,
            ),
        ]

        self.logger.info("TrackValidator inicializado", rules=len(self.rules))

    def validate(self, track) -> ValidationResult:
        """Valida un track contra todas las reglas.

        Args:
            track: TrackState a validar.

        Returns:
            ValidationResult: Resultado detallado de la validación.

        Note:
            Un track pasa la validación si:
            - Score promedio >= 0.4
            - No tiene más de 2 violaciones
            - No tiene violaciones CRITICAL
        """
        violations = []
        total_score = 0.0
        num_rules = len(self.rules)

        for rule in self.rules:
            passed, score, details = rule.validate(track)

            if not passed:
                violations.append(
                    {
                        "rule": rule.name,
                        "severity": rule.severity.value,
                        "score": score,
                        "details": details,
                    }
                )

            total_score += score

        avg_score = total_score / num_rules if num_rules > 0 else 0.0

        passed = avg_score >= 0.4 and len(violations) <= 2

        critical_violations = [
            v for v in violations if v["severity"] == ValidationSeverity.CRITICAL.value
        ]
        if critical_violations:
            passed = False

        return ValidationResult(
            passed=passed,
            violations=violations,
            score=avg_score,
            details={
                "total_rules": num_rules,
                "passed_rules": num_rules - len(violations),
                "violations_count": len(violations),
            },
        )

    def _check_motion_consistency(self, track) -> tuple[bool, float, dict]:
        """Verifica que el movimiento sea consistente.

        Calcula variación de velocidad y aceleración a través del tiempo.

        Args:
            track: Track a validar.

        Returns:
            Tuple[bool, float, Dict]: (passed, score, details)
        """
        if len(track.history) < 3:
            return True, 1.0, {"reason": "insufficient_history"}

        velocities = []
        for i in range(1, len(track.history)):
            dx = track.history[i][0] - track.history[i - 1][0]
            dy = track.history[i][1] - track.history[i - 1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))

        if len(velocities) < 2:
            return True, 1.0, {"reason": "insufficient_velocities"}

        mean_speed = np.mean(velocities)
        if mean_speed < 1.0:
            return True, 1.0, {"reason": "low_speed"}

        speed_std = np.std(velocities)
        speed_variation = speed_std / (mean_speed + 1e-6)

        score = max(0.0, 1.0 - speed_variation / 2.0)
        passed = score >= 0.4

        details = {
            "speed_variation": float(speed_variation),
            "mean_speed": float(mean_speed),
            "std_speed": float(speed_std),
        }

        return passed, score, details

    def _check_trajectory_smoothness(self, track) -> tuple[bool, float, dict]:
        """Verifica que la trayectoria sea suave (sin zigzags extremos).

        Usa regresión lineal para medir la desviación de una línea recta.

        Args:
            track: Track a validar.

        Returns:
            Tuple[bool, float, Dict]: (passed, score, details)
        """
        if len(track.history) < 5:
            return True, 1.0, {"reason": "insufficient_history"}

        points = list(track.history)[-10:]
        if len(points) < 5:
            return True, 1.0, {"reason": "insufficient_points"}

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        try:
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]

            predicted_ys = slope * xs + intercept
            error = np.mean((ys - predicted_ys) ** 2)

            max_error = 100.0

            score = max(0.0, 1.0 - error / max_error)
            passed = score >= 0.3

            details = {
                "regression_error": float(error),
                "slope": float(slope),
                "intercept": float(intercept),
                "points_used": len(points),
            }

            return passed, score, details

        except Exception as e:
            return True, 0.5, {"error": str(e)}

    def _check_shape_consistency(self, track) -> tuple[bool, float, dict]:
        """Verifica que la forma del objeto sea consistente.

        Analiza aspect ratio y área a través del tiempo.

        Args:
            track: Track a validar.

        Returns:
            Tuple[bool, float, Dict]: (passed, score, details)
        """
        if len(track.history) < 3:
            return True, 1.0, {"reason": "insufficient_history"}

        if not hasattr(track, "bbox_history") or len(track.bbox_history) < 3:
            return True, 1.0, {"reason": "no_bbox_history"}

        aspect_ratios = []
        areas = []

        for bbox in track.bbox_history[-10:]:
            if bbox:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w > 0 and h > 0:
                    aspect_ratios.append(w / h)
                    areas.append(w * h)

        if len(aspect_ratios) < 3:
            return True, 1.0, {"reason": "insufficient_data"}

        ratio_std = np.std(aspect_ratios)
        ratio_mean = np.mean(aspect_ratios)

        ratio_variation = ratio_std / (ratio_mean + 1e-6)
        score = max(0.0, 1.0 - ratio_variation)
        passed = score >= 0.3

        details = {
            "aspect_ratio_mean": float(ratio_mean),
            "aspect_ratio_std": float(ratio_std),
            "area_mean": float(np.mean(areas)) if areas else 0,
            "samples": len(aspect_ratios),
        }

        return passed, score, details

    def _check_position_validity(self, track) -> tuple[bool, float, dict]:
        """Verifica que la posición del track sea válida.

        Comprueba que esté dentro de límites razonables de la imagen.

        Args:
            track: Track a validar.

        Returns:
            Tuple[bool, float, Dict]: (passed, score, details)
        """
        if not hasattr(track, "centroid"):
            return False, 0.0, {"error": "no_centroid"}

        x, y = track.centroid

        max_width = 1920
        max_height = 1080

        passed = True
        issues = []

        if x < 0 or x > max_width:
            passed = False
            issues.append(f"x={x} fuera de límites [0, {max_width}]")

        if y < 0 or y > max_height:
            passed = False
            issues.append(f"y={y} fuera de límites [0, {max_height}]")

        x_center = max_width / 2
        y_center = max_height / 2

        distance_from_center = euclidean_distance((x, y), (x_center, y_center))
        max_distance = np.sqrt(max_width**2 + max_height**2) / 2

        score = max(0.0, 1.0 - distance_from_center / max_distance)

        details = {
            "position": (x, y),
            "distance_from_center": float(distance_from_center),
            "issues": issues if issues else None,
        }

        return passed, score, details

    def _check_confidence(self, track) -> tuple[bool, float, dict]:
        """Verifica que la confianza del track sea suficiente.

        Args:
            track: Track a validar.

        Returns:
            Tuple[bool, float, Dict]: (passed, score, details)
        """
        confidence = getattr(track, "confidence", 0.0)

        score = min(1.0, confidence / self.min_confidence)
        passed = confidence >= self.min_confidence

        details = {
            "confidence": float(confidence),
            "threshold": self.min_confidence,
        }

        return passed, score, details

    def validate_batch(self, tracks: list) -> dict[int, ValidationResult]:
        """Valida múltiples tracks.

        Args:
            tracks: Lista de tracks a validar.

        Returns:
            Dict[int, ValidationResult]: Resultados por track_id.

        Example:
            >>> results = validator.validate_batch(tracks)
            >>> for track_id, result in results.items():
            ...     if not result.passed:
            ...         print(f"Track {track_id} inválido")
        """
        results = {}
        for track in tracks:
            track_id = getattr(track, "track_id", id(track))
            results[track_id] = self.validate(track)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del validador.

        Returns:
            Dict[str, Any]: Estadísticas incluyendo:
                - rules_count: Número de reglas configuradas
                - min_confidence: Confianza mínima
                - max_speed_change: Cambio máximo de velocidad
        """
        return {
            "rules_count": len(self.rules),
            "min_confidence": self.min_confidence,
            "max_speed_change": self.max_speed_change,
        }
