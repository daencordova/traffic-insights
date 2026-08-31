"""Definition of abstract interfaces for the system using Protocol.

This module defines the contracts that the main system components
must fulfill: detector, tracker, counter, and pipeline.

The use of Protocol (from typing) allows for structural subtyping,
enabling duck typing while maintaining type safety. This approach
is more flexible than traditional abstract base classes and works
well with the dynamic nature of the system.

Interfaces defined:
    - IDetector: Object detection interface
    - ITracker: Object tracking interface
    - ICounter: Object counting interface
    - IPipeline: Main processing pipeline interface

Example:
    >>> from core.interfaces import IDetector, ITracker, ICounter
    >>>
    >>> class MyDetector:
    ...     def detect(self, frame: np.ndarray) -> DetectionList:
    ...         # Implementation
    ...         return detections
    ...
    ...     def get_classes(self) -> list[int]:
    ...         return [2, 3, 5, 7]
    ...
    ...     def get_performance_stats(self) -> dict[str, Any]:
    ...         return {"fps": 30}
    ...
    ...     def clear_cache(self) -> None:
    ...         pass
    >>>
    >>> # MyDetector is compatible with IDetector protocol
    >>> detector: IDetector = MyDetector()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from core.types import DetectionList, StatsDict, TrackInfoDict


@runtime_checkable
class IDetector(Protocol):
    """Interface for object detectors.

    This protocol defines the contract that all object detectors
    in the system must implement.

    Methods:
        detect: Detect objects in a frame.
        get_classes: Get the classes the detector is configured for.
        get_performance_stats: Get performance statistics.
        clear_cache: Clear the detection cache.

    Example:
        >>> class YOLODetector:
        ...     def detect(self, frame: np.ndarray) -> DetectionList:
        ...         # Implementation
        ...         return detections
        ...
        ...     def get_classes(self) -> list[int]:
        ...         return self._classes
        ...
        ...     def get_performance_stats(self) -> dict[str, Any]:
        ...         return {"inference_time_ms": 15.2}
        ...
        ...     def clear_cache(self) -> None:
        ...         self._cache.clear()
        >>>
        >>> detector: IDetector = YOLODetector()
    """

    def detect(self, frame: np.ndarray) -> DetectionList:
        """Detects objects in a frame.

        Args:
            frame: Image as numpy array (H, W, C) in BGR format.

        Returns:
            DetectionList: List of detections, where each detection contains:
                - 'box': Bounding box [x1, y1, x2, y2]
                - 'centroid': Center point [cx, cy]
                - 'confidence': Confidence score (0.0-1.0)
                - 'class_id': Class identifier
                - Additional optional fields depending on the detector

        Raises:
            DetectionError: If an error occurs during detection.

        Example:
            >>> detections = detector.detect(frame)
            >>> for det in detections:
            ...     print(f"Found {det['class_id']} at {det['centroid']}")
        """
        ...

    def get_classes(self) -> list[int]:
        """Returns the classes the detector is configured to detect.

        Returns:
            list[int]: List of class IDs that the detector will detect.

        Example:
            >>> vehicle_classes = detector.get_classes()
            >>> # [2, 3, 5, 7] for cars, motorcycles, buses, trucks
        """
        ...

    def get_performance_stats(self) -> dict[str, Any]:
        """Returns performance statistics for the detector.

        Returns:
            dict[str, Any]: Statistics including:
                - inference_time_ms: Average inference time
                - fps: Frames per second
                - total_detections: Total detections processed
                - Additional detector-specific metrics

        Example:
            >>> stats = detector.get_performance_stats()
            >>> print(f"Inference: {stats['inference_time_ms']:.1f}ms")
        """
        ...

    def clear_cache(self) -> None:
        """Clears the detection cache.

        This method should clear any cached detections or intermediate
        results to free memory or reset state.

        Example:
            >>> detector.clear_cache()  # Free memory
        """
        ...


@runtime_checkable
class ITracker(Protocol):
    """Interface for object trackers.

    This protocol defines the contract that all object tracking
    systems must implement.

    Methods:
        update: Update tracker with new detections.
        get_tracking_info: Get current tracking information.
        get_stats: Get tracker statistics.
        reset: Reset the tracker completely.

    Example:
        >>> class MultiObjectTracker:
        ...     def update(self, detections: DetectionList, frame: np.ndarray) -> TrackInfoDict:
        ...         # Track association and update
        ...         return active_tracks
        ...
        ...     def get_tracking_info(self) -> TrackInfoDict:
        ...         return self._tracks
        ...
        ...     def get_stats(self) -> dict[str, Any]:
        ...         return {"total_tracks": len(self._tracks)}
        ...
        ...     def reset(self) -> None:
        ...         self._tracks.clear()
        >>>
        >>> tracker: ITracker = MultiObjectTracker()
    """

    def update(self, detections: DetectionList, frame: np.ndarray) -> TrackInfoDict:
        """Updates the tracker with new detections.

        This method processes new detections and updates the tracking state,
        associating detections with existing tracks or creating new tracks.

        Args:
            detections: List of detections from the current frame.
            frame: Current image frame for visual feature extraction.

        Returns:
            TrackInfoDict: Dictionary of active tracks, where the key is
                the track_id and the value contains:
                - 'centroid': Current center point
                - 'bbox': Current bounding box
                - 'state': Track state (active, lost, etc.)
                - 'age': Number of frames tracked
                - 'hits': Number of successful associations
                - Additional tracker-specific fields

        Raises:
            TrackingError: If an error occurs during tracking.

        Example:
            >>> tracks = tracker.update(detections, frame)
            >>> for track_id, info in tracks.items():
            ...     print(f"Track {track_id}: {info['centroid']}")
        """
        ...

    def get_tracking_info(self) -> TrackInfoDict:
        """Returns current tracking information.

        Returns:
            TrackInfoDict: Current state of all active tracks.

        Example:
            >>> tracks = tracker.get_tracking_info()
            >>> active_count = len(tracks)
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Returns tracker statistics.

        Returns:
            dict[str, Any]: Tracker statistics including:
                - active_tracks: Number of currently active tracks
                - total_tracks: Total tracks created
                - lost_tracks: Number of lost tracks
                - association_rate: Rate of successful associations

        Example:
            >>> stats = tracker.get_stats()
            >>> print(f"Active tracks: {stats['active_tracks']}")
        """
        ...

    def reset(self) -> None:
        """Resets the tracker completely.

        This method should:
            - Clear all active tracks
            - Clear any caches or state
            - Reset counters and statistics
            - Prepare the tracker for a fresh start

        Example:
            >>> tracker.reset()  # Start fresh
        """
        ...


@runtime_checkable
class ICounter(Protocol):
    """Interface for object counters.

    This protocol defines the contract that all object counting
    systems must implement.

    Methods:
        process: Process tracks and update counts.
        get_stats: Get current counting statistics.
        reset: Reset counters.

    Example:
        >>> class VehicleCounter:
        ...     def process(self, tracks: TrackInfoDict, frame: np.ndarray) -> StatsDict:
        ...         # Count vehicles crossing lines
        ...         return updated_stats
        ...
        ...     def get_stats(self) -> StatsDict:
        ...         return self._stats
        ...
        ...     def reset(self) -> None:
        ...         self._stats.clear()
        >>>
        >>> counter: ICounter = VehicleCounter()
    """

    def process(self, tracks: TrackInfoDict, frame: np.ndarray) -> StatsDict:
        """Processes tracks and updates counts.

        This method analyzes track movements and updates counting
        statistics based on configured counting lines or zones.

        Args:
            tracks: Dictionary of active tracks.
            frame: Current image frame for spatial references.

        Returns:
            StatsDict: Updated counting statistics including:
                - total_count: Total objects counted
                - counts_by_class: Counts per class
                - counts_by_line: Counts per counting line
                - direction_counts: Counts by direction
                - Additional metrics

        Example:
            >>> stats = counter.process(tracks, frame)
            >>> print(f"Total vehicles: {stats['total_count']}")
        """
        ...

    def get_stats(self) -> StatsDict:
        """Returns current counting statistics.

        Returns:
            StatsDict: Detailed counting statistics.

        Example:
            >>> stats = counter.get_stats()
            >>> for direction, count in stats["direction_counts"].items():
            ...     print(f"{direction}: {count} vehicles")
        """
        ...

    def reset(self) -> None:
        """Resets the counters.

        This method should:
            - Clear all counts and statistics
            - Reset internal state
            - Prepare the counter for a fresh start

        Example:
            >>> counter.reset()  # Reset all counts
        """
        ...


@runtime_checkable
class IPipeline(Protocol):
    """Interface for the main processing pipeline.

    This protocol defines the contract that all pipeline
    implementations must follow for processing video streams.

    Methods:
        run: Run the pipeline with a source.
        process_frame: Process a single frame.
        pause: Pause pipeline execution.
        resume: Resume pipeline execution.
        stop: Stop pipeline execution.

    Example:
        >>> class SyncPipeline:
        ...     def run(self, source: str | None = None) -> None:
        ...         # Process frames in a loop
        ...         pass
        ...
        ...     def process_frame(self, frame: np.ndarray) -> np.ndarray:
        ...         # Process single frame
        ...         return annotated_frame
        ...
        ...     def pause(self) -> None:
        ...         self._paused = True
        ...
        ...     def resume(self) -> None:
        ...         self._paused = False
        ...
        ...     def stop(self) -> None:
        ...         self._running = False
        >>>
        >>> pipeline: IPipeline = SyncPipeline()
    """

    def run(self, source: str | None = None) -> None:
        """Runs the main pipeline.

        This method starts the processing pipeline with the specified
        video source.

        Args:
            source: Video source (camera index, file path, or URL).
                If None, uses the default source from configuration.

        Raises:
            PipelineError: If the pipeline cannot start.

        Example:
            >>> pipeline.run("0")  # Run from camera
            >>> pipeline.run("video.mp4")  # Run from file
        """
        ...

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single frame through the pipeline.

        This method applies the full pipeline (detection, tracking,
        counting, visualization) to a single frame.

        Args:
            frame: Input image frame (H, W, C) in BGR format.

        Returns:
            np.ndarray: Annotated frame with visualization overlays.

        Raises:
            FrameProcessingError: If the frame cannot be processed.

        Example:
            >>> annotated = pipeline.process_frame(frame)
            >>> cv2.imshow("Result", annotated)
        """
        ...

    def pause(self) -> None:
        """Pauses pipeline execution.

        This method temporarily pauses frame processing while keeping
        resources allocated.

        Example:
            >>> pipeline.pause()  # Pause processing
        """
        ...

    def resume(self) -> None:
        """Resumes pipeline execution.

        This method resumes frame processing after a pause.

        Example:
            >>> pipeline.resume()  # Resume processing
        """
        ...

    def stop(self) -> None:
        """Stops pipeline execution.

        This method stops the pipeline and releases all resources.

        Example:
            >>> pipeline.stop()  # Stop and cleanup
        """
        ...
