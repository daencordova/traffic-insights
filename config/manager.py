"""
Gestor de configuración mejorado con carga desde YAML y soporte de entorno.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any
import yaml

from .settings import Config
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigManager:
    """Gestor de configuración singleton con validación Pydantic."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[Config] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_from_file(self, path: str) -> Config:
        """
        Carga configuración desde archivo YAML con logging detallado.

        Args:
            path: Ruta al archivo de configuración

        Returns:
            Config: Objeto de configuración validado

        Raises:
            ConfigurationError: Si hay errores en la configuración
            FileNotFoundError: Si el archivo no existe
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
        """Aplica overrides desde variables de entorno (solo si existen)."""
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
        if self._config is None:
            logger.warning("Configuración no cargada, usando valores por defecto")
            self._config = Config()
        return self._config

    def save_to_file(self, path: str):
        """Guarda la configuración actual a archivo."""
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
        """Obtiene un valor por ruta con notación de puntos."""
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
        """Establece un valor por ruta con notación de puntos."""
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
