"""
Operaciones geométricas optimizadas con Numba para CPU.
"""

import numpy as np

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range


@jit(nopython=True, cache=True, parallel=True)
def calculate_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calcula IoU entre dos conjuntos de bounding boxes (vectorizado).

    Args:
        boxes1: Array de boxes [N, 4] en formato [x1, y1, x2, y2]
        boxes2: Array de boxes [M, 4] en formato [x1, y1, x2, y2]

    Returns:
        np.ndarray: Matriz de IoU [N, M]
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    iou_matrix = np.zeros((n, m), dtype=np.float32)

    for i in prange(n):
        x1_i = boxes1[i, 0]
        y1_i = boxes1[i, 1]
        x2_i = boxes1[i, 2]
        y2_i = boxes1[i, 3]
        area_i = (x2_i - x1_i) * (y2_i - y1_i)

        for j in range(m):
            x1_j = boxes2[j, 0]
            y1_j = boxes2[j, 1]
            x2_j = boxes2[j, 2]
            y2_j = boxes2[j, 3]

            xi1 = x1_i if x1_i > x1_j else x1_j
            yi1 = y1_i if y1_i > y1_j else y1_j
            xi2 = x2_i if x2_i < x2_j else x2_j
            yi2 = y2_i if y2_i < y2_j else y2_j

            if xi2 <= xi1 or yi2 <= yi1:
                continue

            inter = (xi2 - xi1) * (yi2 - yi1)
            area_j = (x2_j - x1_j) * (y2_j - y1_j)
            union = area_i + area_j - inter

            if union > 0:
                iou_matrix[i, j] = inter / union

    return iou_matrix


@jit(nopython=True, cache=True)
def euclidean_distance_batch(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Calcula distancias euclidianas entre dos conjuntos de puntos.

    Args:
        points1: Array de puntos [N, 2]
        points2: Array de puntos [M, 2]

    Returns:
        np.ndarray: Matriz de distancias [N, M]
    """
    n = points1.shape[0]
    m = points2.shape[0]
    distances = np.zeros((n, m), dtype=np.float32)

    for i in prange(n):
        for j in range(m):
            dx = points1[i, 0] - points2[j, 0]
            dy = points1[i, 1] - points2[j, 1]
            distances[i, j] = np.sqrt(dx * dx + dy * dy)

    return distances


@jit(nopython=True, cache=True)
def centroid_to_box(centroids: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """
    Convierte centroides y tamaños a bounding boxes.

    Args:
        centroids: Array de centroides [N, 2]
        sizes: Array de tamaños [N, 2] (width, height)

    Returns:
        np.ndarray: Bounding boxes [N, 4] en formato [x1, y1, x2, y2]
    """
    n = centroids.shape[0]
    boxes = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        w = sizes[i, 0] / 2
        h = sizes[i, 1] / 2
        boxes[i, 0] = centroids[i, 0] - w
        boxes[i, 1] = centroids[i, 1] - h
        boxes[i, 2] = centroids[i, 0] + w
        boxes[i, 3] = centroids[i, 1] + h

    return boxes


@jit(nopython=True, cache=True)
def check_crossing_batch(prev_y: np.ndarray, curr_y: np.ndarray, line_y: float) -> np.ndarray:
    """
    Verifica cruce de línea para múltiples objetos.

    Args:
        prev_y: Array de posiciones Y anteriores
        curr_y: Array de posiciones Y actuales
        line_y: Posición Y de la línea

    Returns:
        np.ndarray: Array booleano de cruces
    """
    return (prev_y < line_y) & (curr_y >= line_y)
