"""
Fábrica para crear extractores de features.

Proporciona una interfaz unificada para crear extractores
con diferentes backends.
"""
from models.feature_extractor.base import FeatureExtractor
from models.feature_extractor.backends import (
    FeatureBackend,
    ResNetBackend,
    HistogramBackend,
    SIFTBackend,
)
from utils.logger import LoggerMixin


class FeatureExtractorFactory(LoggerMixin):
    """
    Fábrica de extractores de features.

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
        use_gpu: bool = True,
        cache_size: int = 500,
        feature_dim: int = 2048,
        **kwargs
    ) -> FeatureExtractor:
        """
        Crea un extractor de features.

        Args:
            backend: Tipo de backend ('resnet', 'histogram', 'sift')
            use_gpu: Usar GPU si está disponible (para ResNet)
            cache_size: Tamaño del caché
            feature_dim: Dimensión del vector de features
            **kwargs: Argumentos adicionales para el backend

        Returns:
            FeatureExtractor: Extractor configurado
        """
        logger = LoggerMixin().logger
        logger.info(
            "Creando FeatureExtractor",
            backend=backend,
            use_gpu=use_gpu,
            cache_size=cache_size
        )

        backend_instance = cls._create_backend(backend, use_gpu, **kwargs)

        return FeatureExtractor(
            backend=backend_instance,
            cache_size=cache_size,
            feature_dim=feature_dim
        )

    @classmethod
    def _create_backend(
        cls,
        backend_type: str,
        use_gpu: bool = True,
        **kwargs
    ) -> FeatureBackend:
        """Crea un backend específico."""
        backend_class = cls._backends.get(backend_type)

        if backend_class is None:
            logger = LoggerMixin().logger
            logger.warning(
                f"Backend '{backend_type}' no encontrado, usando histogram"
            )
            backend_class = HistogramBackend

        if backend_type == "resnet":
            device = "cuda" if use_gpu else "cpu"
            return backend_class(device=device)
        elif backend_type == "sift":
            n_features = kwargs.get("n_features", 128)
            return backend_class(n_features=n_features)
        else:
            return backend_class()

    @classmethod
    def create_best_available(
        cls,
        cache_size: int = 500,
        feature_dim: int = 2048
    ) -> FeatureExtractor:
        """
        Crea el mejor extractor disponible.

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
        logger = LoggerMixin().logger
        logger.info("Creando el mejor extractor disponible")

        try:
            import torch
            if torch.cuda.is_available():
                logger.info("✅ Usando ResNet con GPU")
                return cls.create(
                    backend="resnet",
                    use_gpu=True,
                    cache_size=cache_size,
                    feature_dim=feature_dim
                )
        except ImportError:
            pass

        try:
            import cv2
            sift = cv2.SIFT_create()
            if sift is not None:
                logger.info("✅ Usando SIFT")
                return cls.create(
                    backend="sift",
                    cache_size=cache_size,
                    feature_dim=128
                )
        except Exception:
            pass

        logger.info("✅ Usando Histogram (fallback)")
        return cls.create(
            backend="histogram",
            cache_size=cache_size,
            feature_dim=feature_dim
        )

    @classmethod
    def create_histogram(
        cls,
        cache_size: int = 500,
        feature_dim: int = 2048
    ) -> FeatureExtractor:
        """Crea un extractor basado en histogramas."""
        return cls.create(
            backend="histogram",
            cache_size=cache_size,
            feature_dim=feature_dim
        )

    @classmethod
    def create_resnet(
        cls,
        use_gpu: bool = True,
        cache_size: int = 500,
        feature_dim: int = 2048
    ) -> FeatureExtractor:
        """Crea un extractor basado en ResNet."""
        return cls.create(
            backend="resnet",
            use_gpu=use_gpu,
            cache_size=cache_size,
            feature_dim=feature_dim
        )

    @classmethod
    def create_sift(
        cls,
        cache_size: int = 500,
        n_features: int = 128
    ) -> FeatureExtractor:
        """Crea un extractor basado en SIFT."""
        return cls.create(
            backend="sift",
            cache_size=cache_size,
            feature_dim=n_features,
            n_features=n_features
        )
