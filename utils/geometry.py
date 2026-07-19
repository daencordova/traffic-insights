"""
Funciones utilitarias para operaciones geométricas
"""

from typing import Tuple, Union
import numpy as np

Point = Tuple[int, int]
BoundingBox = Tuple[int, int, int, int]
FloatPoint = Tuple[float, float]


def calculate_centroid(x1: int, y1: int, x2: int, y2: int) -> Point:
    """
    Calcula el centroide de un bounding box

    Args:
        x1, y1: Esquina superior izquierda
        x2, y2: Esquina inferior derecha

    Returns:
        Tupla (cx, cy) con las coordenadas del centroide
    """
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def check_crossing(
    prev_y: int,
    current_y: int,
    line_y: int,
    direction: str = "down"
) -> bool:
    """
    Verifica si un objeto ha cruzado una línea

    Args:
        prev_y: Posición Y anterior
        current_y: Posición Y actual
        line_y: Posición Y de la línea
        direction: Dirección de cruce ('down' o 'up')

    Returns:
        True si el objeto cruzó la línea
    """
    if direction.lower() == "down":
        return prev_y < line_y and current_y >= line_y
    elif direction.lower() == "up":
        return prev_y > line_y and current_y <= line_y
    return False


def euclidean_distance(p1: Union[Point, FloatPoint], p2: Union[Point, FloatPoint]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos

    Args:
        p1, p2: Tuplas (x, y)

    Returns:
        Distancia euclidiana
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_in_bbox(point: Point, bbox: BoundingBox) -> bool:
    """
    Verifica si un punto está dentro de un bounding box

    Args:
        point: Tupla (x, y)
        bbox: Tupla (x1, y1, x2, y2)

    Returns:
        True si el punto está dentro del bbox
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_iou(
    bbox1: BoundingBox,
    bbox2: BoundingBox
) -> float:
    """
    Calcula el Intersection over Union (IoU) entre dos bounding boxes

    Args:
        bbox1, bbox2: Tuplas (x1, y1, x2, y2)

    Returns:
        IoU entre 0 y 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_color(index: int) -> tuple:
    """
    Obtiene un color para identificar elementos

    Args:
        index: Índice para seleccionar color

    Returns:
        Tupla (B, G, R)
    """
    colors = [
        (0, 255, 0),
        (255, 165, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 128, 255),
        (128, 0, 255),
    ]
    return colors[index % len(colors)]
