"""Path utilities for file system operations.

This module provides utility functions for working with file paths,
including conversion, normalization, and directory creation.

Features:
    - Path type conversion (str -> Path)
    - Automatic directory creation
    - Absolute path resolution
    - Safe path joining

Example:
    >>> from utils.path_utils import ensure_path, get_absolute_path, join_paths
    >>>
    >>> # Ensure directory exists
    >>> path = ensure_path("data/output/")
    >>> print(path)  # Path('data/output')
    >>>
    >>> # Get absolute path
    >>> abs_path = get_absolute_path("config.yaml")
    >>> print(abs_path)  # Path('/home/user/project/config.yaml')
    >>>
    >>> # Join paths safely
    >>> full_path = join_paths("data", "screenshots", "image.jpg")
    >>> print(full_path)  # Path('data/screenshots/image.jpg')
"""

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
"""Type alias for path-like objects (string or Path)."""


def ensure_path(path: PathLike) -> Path:
    """Converts any input to Path and ensures the directory exists.

    This function converts a string or Path to a Path object and
    creates all parent directories if they don't exist.

    Args:
        path: Path-like object (string or Path).

    Returns:
        Path: Path object with parent directories created.

    Example:
        >>> path = ensure_path("data/output/results.json")
        >>> # Creates data/output/ directory if it doesn't exist
        >>> print(path)  # Path('data/output/results.json')
        >>>
        >>> # Also works with Path objects
        >>> path = ensure_path(Path("logs/app.log"))
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_absolute_path(path: PathLike) -> Path:
    """Gets the absolute path.

    This function resolves a path to its absolute form,
    expanding user home directory (~) and resolving symlinks.

    Args:
        path: Path-like object (string or Path).

    Returns:
        Path: Absolute path.

    Example:
        >>> abs_path = get_absolute_path("config/settings.yaml")
        >>> print(abs_path)  # Path('/home/user/project/config/settings.yaml')
        >>>
        >>> # With relative path
        >>> abs_path = get_absolute_path("~/documents/file.txt")
        >>> print(abs_path)  # Path('/home/user/documents/file.txt')
    """
    return Path(path).resolve()


def join_paths(*paths: PathLike) -> Path:
    """Joins multiple paths safely.

    This function safely concatenates multiple path components
    into a single Path object, handling platform-specific separators.

    Args:
        *paths: Variable number of path components.

    Returns:
        Path: Joined path.

    Example:
        >>> full_path = join_paths("data", "exports", "2024", "report.csv")
        >>> print(full_path)  # Path('data/exports/2024/report.csv')
        >>>
        >>> # With mixed types
        >>> full_path = join_paths(Path("output"), "processed", "video.mp4")
        >>> print(full_path)  # Path('output/processed/video.mp4')
        >>>
        >>> # With a single path
        >>> path = join_paths("single_file.txt")
        >>> print(path)  # Path('single_file.txt')
    """
    return Path(*[str(p) for p in paths])
