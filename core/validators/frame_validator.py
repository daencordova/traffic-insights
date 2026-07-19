"""
Validador de frames para imágenes y arrays numpy.

Proporciona funciones para validar la integridad y formato de los frames
antes de su procesamiento en el pipeline.
"""

from typing import Optional, Tuple

import numpy as np

from core.constants import (
    MIN_FRAME_WIDTH,
    MIN_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_CHANNELS,
    DEFAULT_RENDER_WIDTH,
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_CHANNELS,
)


def validate_frame(
    frame: np.ndarray,
    min_width: int = MIN_FRAME_WIDTH,
    min_height: int = MIN_FRAME_HEIGHT
) -> bool:
    """
    Valida que el frame sea un array numpy válido y tenga tamaño mínimo.

    Args:
        frame: Imagen a validar (numpy array).
        min_width: Ancho mínimo permitido.
        min_height: Alto mínimo permitido.

    Returns:
        bool: True si el frame es válido, False en caso contrario.
    """
    if frame is None:
        return False

    if not isinstance(frame, np.ndarray):
        return False

    if frame.size == 0:
        return False

    if len(frame.shape) not in (2, 3):
        return False

    h, w = frame.shape[:2]
    if h < min_height or w < min_width:
        return False

    if not np.isfinite(frame).all():
        return False

    return True


def validate_frame_shape(
    frame: np.ndarray,
    expected_dims: int = 3,
    expected_channels: Optional[int] = None
) -> bool:
    """
    Valida las dimensiones y canales del frame.

    Args:
        frame: Imagen a validar.
        expected_dims: Número esperado de dimensiones (2 o 3).
        expected_channels: Número esperado de canales (opcional).

    Returns:
        bool: True si las dimensiones son válidas.
    """
    if not validate_frame(frame):
        return False

    if len(frame.shape) != expected_dims:
        return False

    if expected_channels is not None and expected_dims == 3:
        if frame.shape[2] != expected_channels:
            return False

    return True


def ensure_valid_frame(
    frame: Optional[np.ndarray],
    default_shape: Tuple[int, int, int] = (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_CHANNELS),
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Asegura que el frame sea válido, creando uno por defecto si es necesario.

    Args:
        frame: Frame a validar (puede ser None).
        default_shape: Shape por defecto (height, width, channels).
        dtype: Tipo de dato por defecto.

    Returns:
        np.ndarray: Frame válido (el original o uno por defecto).
    """
    if validate_frame(frame):
        return frame

    return np.zeros(default_shape, dtype=dtype)


def create_default_frame(
    width: int = DEFAULT_RENDER_WIDTH,
    height: int = DEFAULT_RENDER_HEIGHT,
    channels: int = DEFAULT_RENDER_CHANNELS,
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Crea un frame por defecto (negro) con las dimensiones especificadas.

    Args:
        width: Ancho del frame.
        height: Alto del frame.
        channels: Número de canales.
        dtype: Tipo de datos.

    Returns:
        np.ndarray: Frame por defecto.
    """
    return np.zeros((height, width, channels), dtype=dtype)


def get_frame_dimensions(frame: np.ndarray) -> Tuple[int, int]:
    """
    Obtiene las dimensiones (height, width) de un frame válido.

    Args:
        frame: Frame del cual obtener dimensiones.

    Returns:
        Tuple[int, int]: (height, width) o (0, 0) si es inválido.
    """
    if not validate_frame(frame):
        return (0, 0)

    return frame.shape[:2]


def is_grayscale(frame: np.ndarray) -> bool:
    """
    Verifica si el frame es en escala de grises.

    Args:
        frame: Frame a verificar.

    Returns:
        bool: True si es en escala de grises (2D).
    """
    return validate_frame(frame) and len(frame.shape) == 2


def is_color(frame: np.ndarray) -> bool:
    """
    Verifica si el frame es a color.

    Args:
        frame: Frame a verificar.

    Returns:
        bool: True si es a color (3D con 3 canales).
    """
    return validate_frame(frame) and len(frame.shape) == 3 and frame.shape[2] == 3
