"""Frame validator for images and numpy arrays.

Provides functions for validating frame integrity and format
before processing in the pipeline.

This module handles:
    - Frame validation (existence, type, size, finite values)
    - Shape validation (dimensions, channels)
    - Default frame creation and validation
    - Frame dimension extraction
    - Color/grayscale detection

Example:
    >>> from core.validators import (
    ...     validate_frame,
    ...     validate_frame_shape,
    ...     ensure_valid_frame,
    ...     get_frame_dimensions,
    ...     is_color,
    ...     is_grayscale,
    ... )
    >>>
    >>> # Validate a frame
    >>> if validate_frame(frame, min_width=100, min_height=100):
    ...     print("Frame is valid")
    >>>
    >>> # Check frame shape
    >>> if validate_frame_shape(frame, expected_dims=3, expected_channels=3):
    ...     print("Frame is color image")
    >>>
    >>> # Ensure valid frame (creates default if invalid)
    >>> frame = ensure_valid_frame(frame)
    >>>
    >>> # Get dimensions
    >>> h, w = get_frame_dimensions(frame)
    >>> print(f"Frame size: {w}x{h}")
"""

from __future__ import annotations

import numpy as np

from core.constants import (
    DEFAULT_FRAME_CHANNELS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_RENDER_CHANNELS,
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    MIN_FRAME_HEIGHT,
    MIN_FRAME_WIDTH,
)


def validate_frame(
    frame: np.ndarray, min_width: int = MIN_FRAME_WIDTH, min_height: int = MIN_FRAME_HEIGHT
) -> bool:
    """Validates that the frame is a valid numpy array with minimum size.

    This function checks:
        - Frame is not None
        - Frame is a numpy array
        - Frame has non-zero size
        - Frame has valid dimensions (2D or 3D)
        - Frame meets minimum width and height requirements
        - Frame contains finite values (no NaN or Inf)

    Args:
        frame: Image to validate (numpy array).
        min_width: Minimum allowed width.
        min_height: Minimum allowed height.

    Returns:
        bool: True if the frame is valid, False otherwise.

    Example:
        >>> # Valid frame
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> validate_frame(frame)
        True
        >>>
        >>> # Invalid frame (too small)
        >>> small_frame = np.zeros((10, 10), dtype=np.uint8)
        >>> validate_frame(small_frame, min_width=100, min_height=100)
        False
        >>>
        >>> # Invalid frame (contains NaN)
        >>> nan_frame = np.array([[np.nan]])
        >>> validate_frame(nan_frame)
        False
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

    return bool(np.isfinite(frame).all())


def validate_frame_shape(
    frame: np.ndarray, expected_dims: int = 3, expected_channels: int | None = None
) -> bool:
    """Validates frame dimensions and channels.

    This function checks:
        - Frame passes basic validation
        - Frame has expected number of dimensions (2 or 3)
        - Frame has expected number of channels (if specified)

    Args:
        frame: Image to validate.
        expected_dims: Expected number of dimensions (2 or 3).
        expected_channels: Expected number of channels (optional).

    Returns:
        bool: True if dimensions are valid.

    Example:
        >>> # Validate color image
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> validate_frame_shape(frame, expected_dims=3, expected_channels=3)
        True
        >>>
        >>> # Validate grayscale image
        >>> gray = np.zeros((480, 640), dtype=np.uint8)
        >>> validate_frame_shape(gray, expected_dims=2)
        True
        >>>
        >>> # Check channel count
        >>> frame = np.zeros((480, 640, 4), dtype=np.uint8)  # RGBA
        >>> validate_frame_shape(frame, expected_channels=3)
        False
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
    frame: np.ndarray | None,
    default_shape: tuple[int, int, int] = (
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        DEFAULT_FRAME_CHANNELS,
    ),
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Ensures the frame is valid, creating a default one if necessary.

    This function checks if the provided frame is valid and returns it.
    If the frame is invalid or None, it creates and returns a default
    black frame with the specified shape.

    Args:
        frame: Frame to validate (may be None).
        default_shape: Default shape (height, width, channels).
        dtype: Default data type.

    Returns:
        np.ndarray: Valid frame (original or default).

    Example:
        >>> # Valid frame is returned unchanged
        >>> valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> result = ensure_valid_frame(valid_frame)
        >>> result is valid_frame
        True
        >>>
        >>> # Invalid frame creates default
        >>> result = ensure_valid_frame(None)
        >>> result.shape
        (480, 640, 3)
    """
    if validate_frame(frame):
        return frame

    return np.zeros(default_shape, dtype=dtype)


def create_default_frame(
    width: int = DEFAULT_RENDER_WIDTH,
    height: int = DEFAULT_RENDER_HEIGHT,
    channels: int = DEFAULT_RENDER_CHANNELS,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Creates a default (black) frame with specified dimensions.

    Args:
        width: Frame width.
        height: Frame height.
        channels: Number of channels.
        dtype: Data type.

    Returns:
        np.ndarray: Default black frame.

    Example:
        >>> # Create default frame
        >>> frame = create_default_frame(width=640, height=480)
        >>> print(f"Shape: {frame.shape}")  # (480, 640, 3)
        >>> print(f"Max value: {frame.max()}")  # 0
    """
    return np.zeros((height, width, channels), dtype=dtype)


def get_frame_dimensions(frame: np.ndarray) -> tuple[int, int]:
    """Gets the dimensions (height, width) of a valid frame.

    Args:
        frame: Frame to get dimensions from.

    Returns:
        Tuple[int, int]: (height, width) or (0, 0) if invalid.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> h, w = get_frame_dimensions(frame)
        >>> print(f"Height: {h}, Width: {w}")  # Height: 480, Width: 640
        >>>
        >>> # Invalid frame returns zeros
        >>> h, w = get_frame_dimensions(None)
        >>> print(f"Height: {h}, Width: {w}")  # Height: 0, Width: 0
    """
    if not validate_frame(frame):
        return (0, 0)

    return frame.shape[:2]


def is_grayscale(frame: np.ndarray) -> bool:
    """Checks if the frame is grayscale.

    A grayscale image is a 2D array with no channel dimension.

    Args:
        frame: Frame to check.

    Returns:
        bool: True if grayscale (2D).

    Example:
        >>> gray = np.zeros((480, 640), dtype=np.uint8)
        >>> is_grayscale(gray)
        True
        >>>
        >>> color = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> is_grayscale(color)
        False
    """
    return validate_frame(frame) and len(frame.shape) == 2


def is_color(frame: np.ndarray) -> bool:
    """Checks if the frame is color.

    A color image is a 3D array with 3 channels (BGR/RGB).

    Args:
        frame: Frame to check.

    Returns:
        bool: True if color (3D with 3 channels).

    Example:
        >>> color = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> is_color(color)
        True
        >>>
        >>> rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        >>> is_color(rgba)
        False  # 4 channels, not 3
        >>>
        >>> gray = np.zeros((480, 640), dtype=np.uint8)
        >>> is_color(gray)
        False
    """
    return validate_frame(frame) and len(frame.shape) == 3 and frame.shape[2] == 3
