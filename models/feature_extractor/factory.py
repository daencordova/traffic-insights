"""Fábrica para crear extractores de features.

Proporciona una interfaz unificada para crear extractores
con diferentes backends.
"""

from typing import TYPE_CHECKING

from models.feature_extractor.backends import (
    FeatureBackend,
    HistogramBackend,
    ResNetBackend,
    SIFTBackend,
)
from models.feature_extractor.base import FeatureExtractor
from utils.logger import LoggerMixin

if TYPE_CHECKING:
    import torch  # noqa: F401


class FeatureExtractorFactory(LoggerMixin):
    """Fábrica de extractores de features.

    Crea extractores de features con diferentes backends
    según la configuración y disponibilidad.
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
        """Crea un extractor de features.

        Args:
            backend: Tipo de backend ('resnet', 'histogram', 'sift')
            use_gpu: Usar GPU si está disponible (para ResNet)
            cache_size: Tamaño del caché
            feature_dim: Dimensión del vector de features
            **kwargs: Argumentos adicionales para el backend

        Returns:
            FeatureExtractor: Extractor configurado
        """
        logger = cls().logger
        logger.info(
            "Creando FeatureExtractor",
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
        """Crea un backend específico."""
        backend_class = cls._backends.get(backend_type)

        if backend_class is None:
            logger = cls().logger
            logger.warning(
                f"Backend '{backend_type}' no encontrado, usando histogram",
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
        """Crea el mejor extractor disponible.

        Prioridad:
        1. ResNet (si PyTorch y GPU disponibles)
        2. SIFT (si OpenCV lo soporta)
        3. Histogram (siempre disponible)

        Args:
            cache_size: Tamaño del caché
            feature_dim: Dimensión del vector de features

        Returns:
            FeatureExtractor: Mejor extractor disponible
        """
        logger = cls().logger
        logger.info("Creando el mejor extractor disponible")

        if cls._is_torch_gpu_available():
            logger.info("✅ Usando ResNet con GPU")
            return cls.create(
                backend="resnet",
                use_gpu=True,
                cache_size=cache_size,
                feature_dim=feature_dim,
            )

        if cls._is_sift_available():
            logger.info("✅ Usando SIFT")
            return cls.create(
                backend="sift",
                cache_size=cache_size,
                feature_dim=128,
            )

        logger.info("✅ Usando Histogram (fallback)")
        return cls.create(
            backend="histogram",
            cache_size=cache_size,
            feature_dim=feature_dim,
        )

    @classmethod
    def _is_torch_gpu_available(cls) -> bool:
        """Verifica si PyTorch con GPU está disponible."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def _is_sift_available(cls) -> bool:
        """Verifica si SIFT está disponible en OpenCV."""
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
        """Crea un extractor basado en histogramas."""
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
        """Crea un extractor basado en ResNet."""
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
        """Crea un extractor basado en SIFT."""
        return cls.create(
            backend="sift",
            cache_size=cache_size,
            feature_dim=n_features,
            n_features=n_features,
        )
