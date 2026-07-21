"""
Gestor de configuración mejorado con carga desde YAML y soporte de entorno.

Este módulo proporciona un gestor de configuración singleton que:
- Carga configuración desde archivos YAML
- Valida la configuración con Pydantic
- Soporta overrides desde variables de entorno
- Proporciona acceso a la configuración en toda la aplicación
- Guarda la configuración actual a archivo
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Any

from config.settings import Config
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Gestor de configuración singleton con validación Pydantic.

    Esta clase implementa el patrón Singleton para proporcionar
    un único punto de acceso a la configuración del sistema.

    Características:
        - Singleton thread-safe
        - Carga desde YAML con validación
        - Overrides desde variables de entorno
        - Guardado a archivo
        - Acceso por ruta con notación de puntos

    Attributes:
        _instance: Instancia única del gestor.
        _config: Configuración actual del sistema.

    Example:
        >>> config_manager = ConfigManager.get_instance()
        >>> config = config_manager.load_from_file("config.yaml")
        >>> confidence = config_manager.get("model.confidence_threshold")
        >>> config_manager.set("model.confidence_threshold", 0.5)
    """

    _instance: Optional["ConfigManager"] = None
    _config: Optional[Config] = None

    def __new__(cls):
        """Implementación del patrón Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """
        Obtiene la instancia única del gestor de configuración.

        Returns:
            ConfigManager: Instancia única del gestor.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_default(self) -> Config:
        """
        Carga la configuración por defecto.

        Returns:
            Config: Configuración por defecto.

        Note:
            Los valores por defecto están definidos en config/settings.py.
        """
        logger.info("📄 Cargando configuración por defecto")
        self._config = Config()

        logger.info("=" * 60)
        logger.info("📊 PARÁMETROS DE CONFIGURACIÓN POR DEFECTO")
        logger.info("=" * 60)
        logger.info(f"   Modelo: {self._config.model.model_path}")
        logger.info(f"   Device: {self._config.model.device}")
        logger.info(f"   🎯 Confianza: {self._config.model.confidence_threshold}")
        logger.info(f"   📊 IOU: {self._config.model.iou_threshold}")
        logger.info(f"   📐 IMG Size: {self._config.model.imgsz}")
        logger.info("=" * 60)

        return self._config

    def load_from_file(self, path: str) -> Config:
        """
        Carga configuración desde archivo YAML con logging detallado.

        Args:
            path: Ruta al archivo de configuración.

        Returns:
            Config: Objeto de configuración validado.

        Raises:
            ConfigurationError: Si hay errores en la configuración.
            FileNotFoundError: Si el archivo no existe.

        Example:
            >>> config = config_manager.load_from_file("config_prod.yaml")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")

        logger.info(f"📄 Cargando configuración desde: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)

            if raw_data is None:
                raise ConfigurationError("El archivo de configuración está vacío")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parseando YAML: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error leyendo archivo: {e}")

        logger.debug(f"📋 Datos crudos cargados: {list(raw_data.keys())}")

        if 'model' in raw_data:
            model_conf = raw_data['model']
            logger.info(
                f"   📊 Confianza en YAML: {model_conf.get('confidence_threshold', 'No definida')}"
            )
            logger.info(
                f"   📐 IMG Size en YAML: {model_conf.get('imgsz', 'No definido')}"
            )

        try:
            self._config = Config(**raw_data)
            logger.info("✅ Configuración validada correctamente")

            logger.info("=" * 60)
            logger.info("📊 PARÁMETROS DE CONFIGURACIÓN CARGADOS")
            logger.info("=" * 60)
            logger.info(f"   Modelo: {self._config.model.model_path}")
            logger.info(f"   Device: {self._config.model.device}")
            logger.info(f"   🎯 Confianza: {self._config.model.confidence_threshold}")
            logger.info(f"   📊 IOU: {self._config.model.iou_threshold}")
            logger.info(f"   📐 IMG Size: {self._config.model.imgsz}")
            logger.info(f"   🚗 Clases: {self._config.model.vehicle_classes}")
            logger.info(f"   📹 Cámara: {self._config.camera.source}")
            logger.info(f"   📐 Resolución: {self._config.camera.width}x{self._config.camera.height}")
            logger.info(f"   📏 Líneas de conteo: {len(self._config.counting_lines)}")

            for i, line in enumerate(self._config.counting_lines):
                logger.info(
                    f"      Línea {i+1}: {line.get('name', 'Sin nombre')} - "
                    f"{line.get('direction', 'N/A')}"
                )

            logger.info("=" * 60)

        except ValueError as e:
            logger.error(f"❌ Error de validación Pydantic: {e}")
            raise ConfigurationError(f"Datos de configuración inválidos: {e}")
        except Exception as e:
            logger.error(f"❌ Error validando configuración: {e}", exc_info=True)
            raise ConfigurationError(f"Error validando configuración: {e}")

        self._apply_environment_overrides()

        logger.info("✅ Configuración cargada exitosamente")
        logger.info(f"   🎯 Confianza final: {self._config.model.confidence_threshold}")
        logger.info(f"   📐 IMG Size final: {self._config.model.imgsz}")
        logger.info(f"   📏 Líneas finales: {len(self._config.counting_lines)}")

        return self._config

    def _apply_environment_overrides(self):
        """
        Aplica overrides desde variables de entorno.

        Variables de entorno soportadas:
            - MODEL_PATH: Ruta al modelo
            - CAMERA_SOURCE: Fuente de la cámara
            - USE_GPU: Usar GPU (true/false)
            - CONFIDENCE_THRESHOLD: Umbral de confianza

        Note:
            Solo se aplican overrides si las variables de entorno existen.
        """
        if self._config is None:
            return

        env_model_path = os.getenv("MODEL_PATH")
        if env_model_path:
            self._config.model.model_path = env_model_path
            logger.info(f"🔄 Override MODEL_PATH: {self._config.model.model_path}")

        env_camera_source = os.getenv("CAMERA_SOURCE")
        if env_camera_source:
            self._config.camera.source = env_camera_source
            logger.info(f"🔄 Override CAMERA_SOURCE: {self._config.camera.source}")

        env_use_gpu = os.getenv("USE_GPU", "").lower()
        if env_use_gpu == "true":
            self._config.model.device = "cuda"
            logger.info("🔄 Override USE_GPU: cuda")

        env_confidence = os.getenv("CONFIDENCE_THRESHOLD")
        if env_confidence:
            try:
                self._config.model.confidence_threshold = float(env_confidence)
                logger.info(
                    f"🔄 Override CONFIDENCE_THRESHOLD: "
                    f"{self._config.model.confidence_threshold}"
                )
            except ValueError as e:
                logger.warning(f"No se pudo parsear CONFIDENCE_THRESHOLD: {e}")

    @property
    def config(self) -> Config:
        """
        Obtiene la configuración actual.

        Returns:
            Config: Configuración actual o por defecto si no está cargada.

        Note:
            Si no hay configuración cargada, retorna la configuración por defecto.
        """
        if self._config is None:
            logger.warning("Configuración no cargada, usando valores por defecto")
            self._config = Config()
        return self._config

    def save_to_file(self, path: str):
        """
        Guarda la configuración actual a archivo YAML.

        Args:
            path: Ruta donde guardar el archivo.

        Raises:
            ConfigurationError: Si no se puede guardar el archivo.

        Example:
            >>> config_manager.save_to_file("config_backup.yaml")
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config.dict(),
                    f,
                    default_flow_style=False,
                    allow_unicode=True
                )
            logger.info(f"💾 Configuración guardada en: {path}")
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            raise ConfigurationError(f"No se pudo guardar configuración: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor por ruta con notación de puntos.

        Args:
            key: Ruta al valor (ej: "model.confidence_threshold").
            default: Valor por defecto si la clave no existe.

        Returns:
            Any: Valor de la configuración o default.

        Example:
            >>> confidence = config_manager.get("model.confidence_threshold", 0.5)
            >>> imgsz = config_manager.get("model.imgsz", 640)
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Establece un valor por ruta con notación de puntos.

        Args:
            key: Ruta al valor (ej: "model.confidence_threshold").
            value: Nuevo valor a establecer.

        Raises:
            KeyError: Si la clave no existe.

        Example:
            >>> config_manager.set("model.confidence_threshold", 0.6)
            >>> config_manager.set("camera.fps", 30)
        """
        keys = key.split(".")
        target = self.config

        for k in keys[:-1]:
            if hasattr(target, k):
                target = getattr(target, k)
            else:
                raise KeyError(f"Clave no encontrada: {k}")

        setattr(target, keys[-1], value)
        logger.info(f"🔄 Configuración actualizada: {key} = {value}")


config_manager = ConfigManager.get_instance()
config = config_manager.config
