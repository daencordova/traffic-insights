"""
Validador de bounding boxes y centroides.

Proporciona funciones para validar bounding boxes y centroides,
asegurando que cumplan con los requisitos del sistema.
"""

from typing import Any, Tuple, List, Union, Optional

import numpy as np

from core.constants import MIN_BOX_SIZE, MAX_BOX_SIZE

BoundingBox = Tuple[int, int, int, int]
Centroid = Tuple[int, int]


def validate_bbox(
    bbox: Any,
    min_size: int = MIN_BOX_SIZE,
    max_size: int = MAX_BOX_SIZE,
    image_shape: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Valida un bounding box.

    Args:
        bbox: Bounding box a validar (x1, y1, x2, y2).
        min_size: Tamaño mínimo permitido.
        max_size: Tamaño máximo permitido.
        image_shape: Dimensiones de la imagen (height, width) para validar límites.

    Returns:
        bool: True si el bbox es válido.
    """
    if not isinstance(bbox, (tuple, list)):
        return False

    if len(bbox) != 4:
        return False

    try:
        x1, y1, x2, y2 = bbox

        if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
            return False

        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return False

        if x1 >= x2 or y1 >= y2:
            return False

        width = x2 - x1
        height = y2 - y1

        if width < min_size or height < min_size:
            return False

        if width > max_size or height > max_size:
            return False

        if image_shape is not None:
            h, w = image_shape[:2]
            if x1 >= w or y1 >= h or x2 > w or y2 > h:
                return False

        return True

    except (TypeError, ValueError):
        return False


def validate_centroid(centroid: Any, image_shape: Optional[Tuple[int, int]] = None) -> bool:
    """
    Valida un centroide.

    Args:
        centroid: Centroide a validar (x, y).
        image_shape: Dimensiones de la imagen (height, width) para validar límites.

    Returns:
        bool: True si el centroide es válido.
    """
    if not isinstance(centroid, (tuple, list)):
        return False

    if len(centroid) != 2:
        return False

    try:
        x, y = centroid

        if not all(isinstance(v, (int, float)) for v in [x, y]):
            return False

        if x < 0 or y < 0:
            return False

        if image_shape is not None:
            h, w = image_shape[:2]
            if x >= w or y >= h:
                return False

        return True

    except (TypeError, ValueError):
        return False


def normalize_bbox(bbox: BoundingBox, image_shape: Tuple[int, int]) -> BoundingBox:
    """
    Normaliza un bounding box para que esté dentro de los límites de la imagen.

    Args:
        bbox: Bounding box a normalizar.
        image_shape: Dimensiones de la imagen (height, width).

    Returns:
        BoundingBox: Bounding box normalizado.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return (x1, y1, x2, y2)


def validate_bbox_list(bboxes: List[Any]) -> List[bool]:
    """
    Valida una lista de bounding boxes.

    Args:
        bboxes: Lista de bounding boxes a validar.

    Returns:
        List[bool]: Lista de resultados de validación.
    """
    return [validate_bbox(bbox) for bbox in bboxes]


def is_bbox_valid(bbox: Any) -> bool:
    """
    Verificación rápida de validez de bbox (alias de validate_bbox).

    Args:
        bbox: Bounding box a verificar.

    Returns:
        bool: True si es válido.
    """
    return validate_bbox(bbox)


def bbox_to_numpy(bbox: BoundingBox) -> np.ndarray:
    """
    Convierte un bounding box a array numpy.

    Args:
        bbox: Bounding box a convertir.

    Returns:
        np.ndarray: Array de 4 elementos.
    """
    return np.array(bbox, dtype=np.float32)


def numpy_to_bbox(arr: np.ndarray) -> BoundingBox:
    """
    Convierte un array numpy a bounding box.

    Args:
        arr: Array de 4 elementos.

    Returns:
        BoundingBox: Bounding box como tupla.
    """
    return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))


def get_bbox_area(bbox: BoundingBox) -> int:
    """
    Calcula el área de un bounding box.

    Args:
        bbox: Bounding box.

    Returns:
        int: Área del bbox.
    """
    if not validate_bbox(bbox):
        return 0
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)
