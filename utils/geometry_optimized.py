"""Optimized geometry operations with Numba for CPU.

This module provides high-performance geometry calculations using
Numba JIT compilation for CPU optimization.

Features:
    - Batch IoU (Intersection over Union) calculation
    - Batch Euclidean distance computation
    - Centroid to bounding box conversion
    - Line crossing detection
    - Vectorized operations with parallel execution

Example:
    >>> from utils.geometry_numba import (
    ...     calculate_iou_batch,
    ...     euclidean_distance_batch,
    ...     centroid_to_box,
    ...     check_crossing_batch,
    ... )
    >>> import numpy as np
    >>>
    >>> # Calculate IoU between two sets of boxes
    >>> boxes1 = np.array([[10, 10, 50, 50], [20, 20, 60, 60]])
    >>> boxes2 = np.array([[30, 30, 70, 70], [40, 40, 80, 80]])
    >>> iou_matrix = calculate_iou_batch(boxes1, boxes2)
    >>> print(iou_matrix.shape)  # (2, 2)
    >>>
    >>> # Calculate distances between points
    >>> points1 = np.array([[10, 20], [30, 40]])
    >>> points2 = np.array([[50, 60], [70, 80]])
    >>> distances = euclidean_distance_batch(points1, points2)
    >>>
    >>> # Check line crossings
    >>> prev_y = np.array([100, 200, 150])
    >>> curr_y = np.array([150, 180, 200])
    >>> crossings = check_crossing_batch(prev_y, curr_y, line_y=160)
"""

import numpy as np

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Dummy decorator for when Numba is not available."""

        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator

    prange = range


@jit(nopython=True, cache=True, parallel=True)
def calculate_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calculates IoU between two sets of bounding boxes (vectorized).

    This function computes the Intersection over Union (IoU) for all
    pairs of bounding boxes between two sets using Numba JIT and
    parallel execution for maximum performance.

    Args:
        boxes1: Array of boxes [N, 4] in [x1, y1, x2, y2] format.
        boxes2: Array of boxes [M, 4] in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: IoU matrix [N, M] with values in [0, 1].

    Example:
        >>> boxes1 = np.array([[10, 10, 50, 50], [30, 30, 70, 70]])
        >>> boxes2 = np.array([[20, 20, 60, 60], [40, 40, 80, 80]])
        >>> iou = calculate_iou_batch(boxes1, boxes2)
        >>> print(iou)  # IoU matrix
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
    """Calculates Euclidean distances between two sets of points.

    This function computes distances for all pairs of points between
    two sets using Numba JIT optimization.

    Args:
        points1: Array of points [N, 2] with (x, y) coordinates.
        points2: Array of points [M, 2] with (x, y) coordinates.

    Returns:
        np.ndarray: Distance matrix [N, M].

    Example:
        >>> pts1 = np.array([[0, 0], [10, 10]])
        >>> pts2 = np.array([[5, 5], [15, 15]])
        >>> distances = euclidean_distance_batch(pts1, pts2)
        >>> print(distances)  # Distance matrix
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
    """Converts centroids and sizes to bounding boxes.

    This function converts center points and size dimensions to
    bounding boxes in [x1, y1, x2, y2] format.

    Args:
        centroids: Array of centroids [N, 2] with (cx, cy).
        sizes: Array of sizes [N, 2] with (width, height).

    Returns:
        np.ndarray: Bounding boxes [N, 4] in [x1, y1, x2, y2] format.

    Example:
        >>> centroids = np.array([[50, 50], [100, 100]])
        >>> sizes = np.array([[40, 30], [60, 50]])
        >>> boxes = centroid_to_box(centroids, sizes)
        >>> print(boxes)  # [[30, 35, 70, 65], [70, 75, 130, 125]]
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
    """Checks line crossing for multiple objects.

    This function determines which objects crossed a horizontal line
    by comparing previous and current Y positions.

    Args:
        prev_y: Array of previous Y positions.
        curr_y: Array of current Y positions.
        line_y: Y position of the line.

    Returns:
        np.ndarray: Boolean array of crossings (True = crossed).

    Example:
        >>> prev = np.array([100, 200, 150])
        >>> curr = np.array([120, 180, 160])
        >>> crossings = check_crossing_batch(prev, curr, line_y=150)
        >>> print(crossings)  # [False, False, True]
    """
    return (prev_y < line_y) & (curr_y >= line_y)


@jit(nopython=True, cache=True)
def calculate_iou_vectorized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Vectorized IoU calculation with Numba for maximum speed.

    This is an optimized version of IoU calculation that precomputes
    areas for faster execution.

    Args:
        boxes1: Array of boxes [N, 4] in [x1, y1, x2, y2] format.
        boxes2: Array of boxes [M, 4] in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: IoU matrix [N, M] with values in [0, 1].

    Example:
        >>> boxes1 = np.array([[10, 10, 50, 50], [30, 30, 70, 70]])
        >>> boxes2 = np.array([[20, 20, 60, 60], [40, 40, 80, 80]])
        >>> iou = calculate_iou_vectorized(boxes1, boxes2)
        >>> print(iou)  # IoU matrix
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    iou_matrix = np.zeros((n, m), dtype=np.float32)

    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    for i in range(n):
        x1_i = boxes1[i, 0]
        y1_i = boxes1[i, 1]
        x2_i = boxes1[i, 2]
        y2_i = boxes1[i, 3]
        area_i = areas1[i]

        for j in range(m):
            xi1 = x1_i if x1_i > boxes2[j, 0] else boxes2[j, 0]
            yi1 = y1_i if y1_i > boxes2[j, 1] else boxes2[j, 1]
            xi2 = x2_i if x2_i < boxes2[j, 2] else boxes2[j, 2]
            yi2 = y2_i if y2_i < boxes2[j, 3] else boxes2[j, 3]

            if xi2 > xi1 and yi2 > yi1:
                inter = (xi2 - xi1) * (yi2 - yi1)
                union = area_i + areas2[j] - inter
                if union > 0:
                    iou_matrix[i, j] = inter / union

    return iou_matrix
