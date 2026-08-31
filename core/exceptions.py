"""Custom exceptions for the traffic tracking system.

This hierarchy allows granular error handling and facilitates
automatic recovery in different scenarios.

The structure follows an inheritance pattern where all domain
exceptions inherit from VehicleCountingError, allowing consistent
catching of any system error.

Exception Hierarchy:
    VehicleCountingError (base)
    ├── ConfigurationError
    │   └── ValidationError
    ├── ModelLoadError
    ├── DetectionError
    │   └── InferenceError
    ├── TrackingError
    │   ├── MatchingError
    │   └── ReIdentificationError
    ├── PipelineError
    │   ├── FrameProcessingError
    │   └── CaptureError
    ├── ResourceError
    │   ├── CacheError
    │   └── MemoryError
    ├── IOError
    │   ├── FileNotFoundError
    │   └── CameraError
    ├── CountingError
    ├── ConnectionError
    └── TimeoutError
"""

from __future__ import annotations

from typing import Any


class VehicleCountingError(Exception):
    """Base exception for the entire traffic tracking system.

    All custom exceptions inherit from this class, allowing
    catching any domain error consistently.

    Attributes:
        message: Descriptive error message.
        details: Dictionary with additional contextual information.

    Example:
        >>> try:
        ...     process_frame(frame)
        ... except VehicleCountingError as e:
        ...     print(f"System error: {e}")
        ...     if e.details:
        ...         print(f"Details: {e.details}")
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initializes the base exception.

        Args:
            message: Descriptive error message.
            details: Dictionary with additional debugging information.
                May include: 'component', 'frame_number', 'source', etc.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        """Returns a human-readable representation of the error."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(VehicleCountingError):
    """Error related to system configuration.

    Raised when there are issues with:
        - Malformed configuration files
        - Invalid parameters
        - Failed configuration validation

    Example:
        >>> try:
        ...     config = load_config("invalid.yaml")
        ... except ConfigurationError as e:
        ...     print(f"Configuration error: {e}")
    """


class ValidationError(VehicleCountingError):
    """Error in data or parameter validation.

    Raised when:
        - Input data does not meet specifications
        - Parameters are out of range
        - Data structures are invalid

    Example:
        >>> try:
        ...     validate_coordinates(x, y)
        ... except ValidationError as e:
        ...     print(f"Validation failed: {e}")
    """


class ModelLoadError(VehicleCountingError):
    """Error loading a machine learning model.

    Raised when:
        - The model file does not exist
        - The model is incompatible
        - There are memory issues during loading

    Example:
        >>> try:
        ...     model = YOLO("nonexistent.pt")
        ... except ModelLoadError as e:
        ...     print(f"Model loading failed: {e}")
    """


class DetectionError(VehicleCountingError):
    """Error in the object detection system.

    Raised when:
        - Model inference fails
        - Results are invalid
        - There are GPU/memory issues

    Example:
        >>> try:
        ...     detections = detector.detect(frame)
        ... except DetectionError as e:
        ...     print(f"Detection failed: {e}")
    """


class InferenceError(DetectionError):
    """Error during model inference.

    Specific to failures in the inference engine
    (ONNX, PyTorch, etc.).

    Example:
        >>> try:
        ...     results = model(frame)
        ... except InferenceError as e:
        ...     print(f"Inference engine error: {e}")
    """


class TrackingError(VehicleCountingError):
    """Error in the tracking system.

    Raised when:
        - Matching between detections and tracks fails
        - Kalman filter has numerical issues
        - There are inconsistencies in track states

    Example:
        >>> try:
        ...     tracks = tracker.update(detections)
        ... except TrackingError as e:
        ...     print(f"Tracking failed: {e}")
    """


class MatchingError(TrackingError):
    """Error in the matching process between detections and tracks.

    Specific to failures in:
        - Cost matrix calculation
        - Optimal assignment (Hungarian algorithm)
        - Hierarchical matching

    Example:
        >>> try:
        ...     matches = matcher.match(detections, tracks)
        ... except MatchingError as e:
        ...     print(f"Matching failed: {e}")
    """


class ReIdentificationError(TrackingError):
    """Error in the re-identification system.

    Raised when:
        - Feature extraction fails
        - Feature cache has issues
        - Feature comparison is invalid

    Example:
        >>> try:
        ...     features = reid.extract(frame)
        ... except ReIdentificationError as e:
        ...     print(f"Re-identification failed: {e}")
    """


class PipelineError(VehicleCountingError):
    """Error in the processing pipeline.

    Generic error for issues in the main processing flow.

    Example:
        >>> try:
        ...     pipeline.process(frame)
        ... except PipelineError as e:
        ...     print(f"Pipeline error: {e}")
    """


class FrameProcessingError(PipelineError):
    """Error processing a video frame.

    Raised when:
        - A frame cannot be processed correctly
        - The frame format is invalid
        - There are errors in image transformations

    Example:
        >>> try:
        ...     processed = preprocess_frame(frame)
        ... except FrameProcessingError as e:
        ...     print(f"Frame processing failed: {e}")
    """


class CaptureError(PipelineError):
    """Error in video capture.

    Raised when:
        - The camera does not respond
        - The video source is inaccessible
        - There are frame reading issues

    Example:
        >>> try:
        ...     ret, frame = cap.read()
        ...     if not ret:
        ...         raise CaptureError("Failed to read frame")
        ... except CaptureError as e:
        ...     print(f"Capture failed: {e}")
    """


class ResourceError(VehicleCountingError):
    """Error managing resources (memory, files, etc.).

    Generic error for system resource issues.

    Example:
        >>> try:
        ...     allocate_large_buffer()
        ... except ResourceError as e:
        ...     print(f"Resource error: {e}")
    """


class CacheError(ResourceError):
    """Error in the caching system.

    Raised when:
        - The cache is corrupted
        - There are memory issues in the cache
        - The eviction policy fails

    Example:
        >>> try:
        ...     cache.store(key, value)
        ... except CacheError as e:
        ...     print(f"Cache error: {e}")
    """


class MemoryError(ResourceError):
    """Error related to insufficient memory.

    Raised when:
        - Not enough RAM is available
        - Swap memory is exhausted
        - A memory leak is detected

    Example:
        >>> try:
        ...     process_large_dataset()
        ... except MemoryError as e:
        ...     print(f"Out of memory: {e}")
    """


class IOError(VehicleCountingError):
    """General input/output error.

    Raised for I/O issues that don't fit into specific categories.

    Example:
        >>> try:
        ...     read_data_from_disk()
        ... except IOError as e:
        ...     print(f"I/O error: {e}")
    """


class FileNotFoundError(IOError):
    """File not found error.

    Raised when a required file does not exist.

    Example:
        >>> try:
        ...     with open("missing_file.txt") as f:
        ...         data = f.read()
        ... except FileNotFoundError as e:
        ...     print(f"File not found: {e}")
    """


class CameraError(IOError):
    """Error related to the camera or video source.

    Raised when:
        - The camera cannot be opened
        - Camera parameters cannot be configured
        - The camera disconnects unexpectedly

    Example:
        >>> try:
        ...     cap = cv2.VideoCapture(0)
        ...     if not cap.isOpened():
        ...         raise CameraError("Camera not available")
        ... except CameraError as e:
        ...     print(f"Camera error: {e}")
    """


class CountingError(VehicleCountingError):
    """Error in the counting system.

    Raised when:
        - Counting lines are not valid
        - Crossing detection has issues
        - Statistics cannot be updated

    Example:
        >>> try:
        ...     counter.process(frame)
        ... except CountingError as e:
        ...     print(f"Counting failed: {e}")
    """


class ConnectionError(VehicleCountingError):
    """Connection error to external services.

    Raised for network or remote service connection issues.

    Example:
        >>> try:
        ...     response = http_client.get(url)
        ... except ConnectionError as e:
        ...     print(f"Connection failed: {e}")
    """


class TimeoutError(VehicleCountingError):
    """Timeout error in operations.

    Raised when an operation exceeds its time limit.

    Example:
        >>> try:
        ...     result = operation_with_timeout(timeout=5.0)
        ... except TimeoutError as e:
        ...     print(f"Operation timed out: {e}")
    """
