"""
Models module.

This module provides core models for the tracking system including
state management, Kalman filtering, and feature extraction.

Submodules:
    - models.track_state: Track state and status management
    - models.kalman: Enhanced Kalman filter for motion prediction
    - models.feature_extractor: Feature extraction for re-identification

Example:
    >>> from models import TrackState, TrackStatus, EnhancedKalmanFilter, FeatureExtractor
    >>>
    >>> # Create a track state
    >>> track = TrackState(track_id=1, centroid=(100, 200), bbox=(50, 50, 150, 150))
    >>>
    >>> # Create a Kalman filter for tracking
    >>> kalman = EnhancedKalmanFilter()
    >>>
    >>> # Create a feature extractor for re-identification
    >>> extractor = FeatureExtractor(backend="resnet")
    >>>
    >>> # Check track status
    >>> if track.is_active():
    ...     print(f"Track {track.track_id} is active")
"""

from models.feature_extractor import FeatureExtractor
from models.kalman import EnhancedKalmanFilter
from models.track_state import TrackState, TrackStatus

__all__ = [
    "TrackState",
    "TrackStatus",
    "EnhancedKalmanFilter",
    "FeatureExtractor",
]
