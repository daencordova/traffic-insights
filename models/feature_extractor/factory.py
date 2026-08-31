"""Factory for creating feature extractors.

Provides a unified interface for creating extractors
with different backends.
"""

from models.feature_extractor.backends import (
    FeatureBackend,
    HistogramBackend,
    ResNetBackend,
    SIFTBackend,
)
from models.feature_extractor.base import FeatureExtractor
from utils.logger import LoggerMixin


class FeatureExtractorFactory(LoggerMixin):
    """Factory for feature extractors.

    Creates feature extractors with different backends
    based on configuration and availability.

    This factory provides a unified interface for creating
    feature extractors with various backends:
        - ResNet: Deep learning features (requires PyTorch)
        - SIFT: Traditional computer vision features
        - Histogram: Fast CPU-based features (always available)

    Example:
        >>> # Create with specific backend
        >>> extractor = FeatureExtractorFactory.create(
        ...     backend="resnet", use_gpu=True, cache_size=500
        ... )
        >>>
        >>> # Create the best available
        >>> extractor = FeatureExtractorFactory.create_best_available()
        >>>
        >>> # Create a specific backend
        >>> extractor = FeatureExtractorFactory.create_resnet(use_gpu=True)
        >>> extractor = FeatureExtractorFactory.create_sift(n_features=128)
        >>> extractor = FeatureExtractorFactory.create_histogram()
        >>>
        >>> # Features can now be extracted
        >>> features = extractor.extract_features(image, bbox)
    """

    _backends = {
        "resnet": ResNetBackend,
        "histogram": HistogramBackend,
        "sift": SIFTBackend,
    }

    @classmethod
    def create(
        cls,
        backend: str = "histogram",
        *,
        use_gpu: bool = True,
        cache_size: int = 500,
        feature_dim: int = 2048,
        **kwargs,
    ) -> FeatureExtractor:
        """Creates a feature extractor.

        Args:
            backend: Backend type ('resnet', 'histogram', 'sift').
            use_gpu: Use GPU if available (for ResNet).
            cache_size: Cache size.
            feature_dim: Feature vector dimension.
            **kwargs: Additional arguments for the backend.

        Returns:
            FeatureExtractor: Configured extractor.

        Example:
            >>> # Create ResNet extractor with GPU
            >>> extractor = FeatureExtractorFactory.create(
            ...     backend="resnet", use_gpu=True, cache_size=1000
            ... )
            >>>
            >>> # Create SIFT extractor
            >>> extractor = FeatureExtractorFactory.create(backend="sift", n_features=256)
        """
        logger = cls().logger
        logger.info(
            "Creating FeatureExtractor",
            backend=backend,
            use_gpu=use_gpu,
            cache_size=cache_size,
        )

        backend_instance = cls._create_backend(
            backend_type=backend,
            use_gpu=use_gpu,
            **kwargs,
        )

        return FeatureExtractor(
            backend=backend_instance,
            cache_size=cache_size,
            feature_dim=feature_dim,
        )

    @classmethod
    def _create_backend(
        cls,
        backend_type: str,
        *,
        use_gpu: bool = True,
        **kwargs,
    ) -> FeatureBackend:
        """Creates a specific backend.

        Args:
            backend_type: Backend type string.
            use_gpu: Whether to use GPU.
            **kwargs: Backend-specific arguments.

        Returns:
            FeatureBackend: Created backend instance.

        Note:
            Falls back to HistogramBackend if the requested backend
            is not available.
        """
        backend_class = cls._backends.get(backend_type)

        if backend_class is None:
            logger = cls().logger
            logger.warning(
                f"Backend '{backend_type}' not found, using histogram",
            )
            backend_class = HistogramBackend

        if backend_type == "resnet":
            device = "cuda" if use_gpu else "cpu"
            return backend_class(device=device)
        if backend_type == "sift":
            n_features = kwargs.get("n_features", 128)
            return backend_class(n_features=n_features)
        return backend_class()

    @classmethod
    def create_best_available(
        cls,
        cache_size: int = 500,
        feature_dim: int = 2048,
    ) -> FeatureExtractor:
        """Creates the best available extractor.

        Priority order:
            1. ResNet (if PyTorch and GPU available)
            2. SIFT (if OpenCV supports it)
            3. Histogram (always available)

        Args:
            cache_size: Cache size.
            feature_dim: Feature vector dimension.

        Returns:
            FeatureExtractor: Best available extractor.

        Example:
            >>> # Automatically selects the best backend
            >>> extractor = FeatureExtractorFactory.create_best_available()
            >>>
            >>> # Check which backend was selected
            >>> print(f"Backend: {extractor.backend.name}")
        """
        logger = cls().logger
        logger.info("Creating best available extractor")

        if cls._is_torch_gpu_available():
            logger.info("Using ResNet with GPU")
            return cls.create(
                backend="resnet",
                use_gpu=True,
                cache_size=cache_size,
                feature_dim=feature_dim,
            )

        if cls._is_sift_available():
            logger.info("Using SIFT")
            return cls.create(
                backend="sift",
                cache_size=cache_size,
                feature_dim=128,
            )

        logger.info("Using Histogram (fallback)")
        return cls.create(
            backend="histogram",
            cache_size=cache_size,
            feature_dim=feature_dim,
        )

    @classmethod
    def _is_torch_gpu_available(cls) -> bool:
        """Checks if PyTorch with GPU is available.

        Returns:
            bool: True if PyTorch and CUDA are available.

        Example:
            >>> if FeatureExtractorFactory._is_torch_gpu_available():
            ...     print("GPU acceleration available")
        """
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def _is_sift_available(cls) -> bool:
        """Checks if SIFT is available in OpenCV.

        Returns:
            bool: True if SIFT is available.

        Example:
            >>> if FeatureExtractorFactory._is_sift_available():
            ...     print("SIFT features available")
        """
        try:
            import cv2

            sift = cv2.SIFT_create()
            return sift is not None
        except Exception:
            return False

    @classmethod
    def create_histogram(
        cls,
        cache_size: int = 500,
        feature_dim: int = 2048,
    ) -> FeatureExtractor:
        """Creates a histogram-based extractor.

        Args:
            cache_size: Cache size.
            feature_dim: Feature vector dimension.

        Returns:
            FeatureExtractor: Histogram-based extractor.

        Example:
            >>> extractor = FeatureExtractorFactory.create_histogram()
            >>> # Fast, CPU-only features
        """
        return cls.create(
            backend="histogram",
            cache_size=cache_size,
            feature_dim=feature_dim,
        )

    @classmethod
    def create_resnet(
        cls,
        *,
        use_gpu: bool = True,
        cache_size: int = 500,
        feature_dim: int = 2048,
    ) -> FeatureExtractor:
        """Creates a ResNet-based extractor.

        Args:
            use_gpu: Use GPU if available.
            cache_size: Cache size.
            feature_dim: Feature vector dimension.

        Returns:
            FeatureExtractor: ResNet-based extractor.

        Example:
            >>> # Use GPU if available
            >>> extractor = FeatureExtractorFactory.create_resnet(use_gpu=True)
            >>>
            >>> # Force CPU
            >>> extractor = FeatureExtractorFactory.create_resnet(use_gpu=False)
        """
        return cls.create(
            backend="resnet",
            use_gpu=use_gpu,
            cache_size=cache_size,
            feature_dim=feature_dim,
        )

    @classmethod
    def create_sift(
        cls,
        cache_size: int = 500,
        n_features: int = 128,
    ) -> FeatureExtractor:
        """Creates a SIFT-based extractor.

        Args:
            cache_size: Cache size.
            n_features: Number of features to extract.

        Returns:
            FeatureExtractor: SIFT-based extractor.

        Example:
            >>> # Standard SIFT
            >>> extractor = FeatureExtractorFactory.create_sift()
            >>>
            >>> # More features for better discrimination
            >>> extractor = FeatureExtractorFactory.create_sift(n_features=256)
        """
        return cls.create(
            backend="sift",
            cache_size=cache_size,
            feature_dim=n_features,
            n_features=n_features,
        )
