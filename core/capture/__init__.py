"""
Video capture module.

This module provides components for capturing and managing video streams
from various sources including cameras, video files, and network streams.

Submodules:
    - core.capture.manager: Manages video capture with configuration support
    - core.capture.reconnector: Handles automatic reconnection on failures

The module exports the main components for video capture management:
    - CaptureManager: Manages video capture lifecycle and configuration
    - Reconnector: Provides automatic reconnection logic for reliable streaming

Example:
    >>> from core.capture import CaptureManager, Reconnector
    >>>
    >>> # Create capture manager with configuration
    >>> manager = CaptureManager(source="0", width=640, height=480)
    >>>
    >>> # Start capturing
    >>> manager.start()
    >>>
    >>> # Get frames
    >>> ret, frame = manager.read()
    >>>
    >>> # Stop and cleanup
    >>> manager.stop()
"""

from core.capture.manager import CaptureManager
from core.capture.reconnector import Reconnector

__all__ = [
    "CaptureManager",
    "Reconnector",
]
