"""
Validador de detecciones de objetos.

Proporciona funciones para validar detecciones individuales y listas,
asegurando que cumplan con los requisitos del sistema.
"""

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from core.validators.bbox_validator import validate_bbox, validate_centroid
from core.constants import MIN_DETECTION_AREA, MIN_DETECTION_CONFIDENCE, MAX_DETECTION_CONFIDENCE


@dataclass
class DetectionValidationResult:
    """Resultado de la validación de una detección."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float


def validate_detection(
    detection: Dict[str, Any],
    min_confidence: float = MIN_DETECTION_CONFIDENCE,
    max_confidence: float = MAX_DETECTION_CONFIDENCE,
    min_area: int = MIN_DETECTION_AREA,
    require_all_fields: bool = True
) -> DetectionValidationResult:
    """
    Valida una detección completa.

    Args:
        detection: Diccionario de detección.
        min_confidence: Confianza mínima permitida.
        max_confidence: Confianza máxima permitida.
        min_area: Área mínima permitida.
        require_all_fields: Si todos los campos requeridos deben existir.

    Returns:
        DetectionValidationResult: Resultado de la validación.
    """
    errors = []
    warnings = []
    score = 1.0

    if not isinstance(detection, dict):
        errors.append("La detección debe ser un diccionario")
        return DetectionValidationResult(False, errors, warnings, 0.0)

    required_fields = ["box", "centroid", "confidence"]
    if require_all_fields:
        missing = [f for f in required_fields if f not in detection]
        if missing:
            errors.append(f"Campos requeridos faltantes: {missing}")
            score *= 0.5

    box = detection.get("box")
    if box is not None:
        if not validate_bbox(box):
            errors.append(f"Bounding box inválido: {box}")
            score *= 0.3
    elif require_all_fields:
        errors.append("Campo 'box' faltante")
        score *= 0.3

    centroid = detection.get("centroid")
    if centroid is not None:
        if not validate_centroid(centroid):
            errors.append(f"Centroide inválido: {centroid}")
            score *= 0.3
    elif require_all_fields:
        errors.append("Campo 'centroid' faltante")
        score *= 0.3

    confidence = detection.get("confidence")
    if confidence is not None:
        try:
            conf = float(confidence)
            if conf < min_confidence or conf > max_confidence:
                warnings.append(
                    f"Confianza fuera de rango [{min_confidence}, {max_confidence}]: {conf}"
                )
                score *= 0.7
        except (TypeError, ValueError):
            errors.append(f"Confianza inválida: {confidence}")
            score *= 0.5
    elif require_all_fields:
        errors.append("Campo 'confidence' faltante")
        score *= 0.5

    if box and validate_bbox(box):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            warnings.append(f"Área muy pequeña: {area} < {min_area}")
            score *= 0.7

    class_id = detection.get("class_id")
    if class_id is not None:
        if not isinstance(class_id, int) or class_id < 0:
            errors.append(f"class_id inválido: {class_id}")
            score *= 0.5

    label = detection.get("label")
    if label is not None and not isinstance(label, str):
        errors.append(f"label inválido: {label}")
        score *= 0.5

    is_valid = len(errors) == 0
    return DetectionValidationResult(is_valid, errors, warnings, max(0.0, min(1.0, score)))


def validate_detection_list(
    detections: List[Dict[str, Any]],
    min_confidence: float = MIN_DETECTION_CONFIDENCE,
    max_confidence: float = MAX_DETECTION_CONFIDENCE,
    min_area: int = MIN_DETECTION_AREA
) -> Tuple[List[Dict[str, Any]], List[DetectionValidationResult]]:
    """
    Valida una lista de detecciones.

    Args:
        detections: Lista de detecciones a validar.
        min_confidence: Confianza mínima permitida.
        max_confidence: Confianza máxima permitida.
        min_area: Área mínima permitida.

    Returns:
        Tuple[List[Dict], List[DetectionValidationResult]]:
            Lista de detecciones válidas y resultados de validación.
    """
    valid_detections = []
    results = []

    for detection in detections:
        result = validate_detection(
            detection,
            min_confidence,
            max_confidence,
            min_area,
            require_all_fields=True
        )
        results.append(result)
        if result.is_valid:
            valid_detections.append(detection)

    return valid_detections, results


def validate_detection_required_fields(detection: Dict[str, Any]) -> bool:
    """
    Verifica rápidamente si una detección tiene todos los campos requeridos.

    Args:
        detection: Detección a verificar.

    Returns:
        bool: True si tiene todos los campos requeridos.
    """
    required = ["box", "centroid", "confidence"]
    return all(field in detection for field in required)


def filter_valid_detections(
    detections: List[Dict[str, Any]],
    min_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_area: int = MIN_DETECTION_AREA
) -> List[Dict[str, Any]]:
    """
    Filtra detecciones válidas según confianza y área.

    Args:
        detections: Lista de detecciones.
        min_confidence: Confianza mínima.
        min_area: Área mínima.

    Returns:
        List[Dict[str, Any]]: Lista de detecciones válidas.
    """
    valid = []

    for det in detections:
        confidence = det.get("confidence", 0.0)
        if confidence < min_confidence:
            continue

        box = det.get("box")
        if not box or not validate_bbox(box):
            continue

        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        valid.append(det)

    return valid


def get_detection_stats(
    detections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de una lista de detecciones.

    Args:
        detections: Lista de detecciones.

    Returns:
        Dict[str, Any]: Estadísticas de las detecciones.
    """
    if not detections:
        return {
            "count": 0,
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "class_distribution": {},
            "avg_area": 0.0,
        }

    confidences = []
    class_counts = {}
    areas = []

    for det in detections:
        conf = det.get("confidence", 0.0)
        confidences.append(conf)

        class_id = det.get("class_id", -1)
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

        box = det.get("box")
        if box and len(box) == 4:
            x1, y1, x2, y2 = box
            areas.append((x2 - x1) * (y2 - y1))

    return {
        "count": len(detections),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "class_distribution": class_counts,
        "avg_area": sum(areas) / len(areas) if areas else 0.0,
    }
