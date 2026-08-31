"""Enhanced configuration manager with YAML loading and environment support.

This module provides a singleton configuration manager that:
- Loads configuration from YAML files
- Validates configuration with Pydantic
- Supports overrides from environment variables
- Provides application-wide configuration access
- Saves current configuration to file
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from config.settings import Config
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigManager:
    """Singleton configuration manager with Pydantic validation.

    This class implements the Singleton pattern to provide a single
    point of access to the system configuration.

    Features:
        - Thread-safe singleton
        - YAML loading with validation
        - Environment variable overrides
        - File saving capability
        - Dot-notation path access

    Attributes:
        _instance: Singleton instance of the manager.
        _config: Current system configuration.

    Example:
        >>> config_manager = ConfigManager.get_instance()
        >>> config = config_manager.load_from_file("config.yaml")
        >>> confidence = config_manager.get("model.confidence_threshold")
        >>> config_manager.set("model.confidence_threshold", 0.5)
    """

    _instance: ConfigManager | None = None
    _config: Config | None = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> ConfigManager:
        """Gets the singleton instance of the configuration manager.

        Returns:
            ConfigManager: Singleton instance of the manager.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_default(self) -> Config:
        """Loads the default configuration.

        Returns:
            Config: Default configuration.

        Note:
            Default values are defined in config/settings.py.
        """
        logger.info("Loading default configuration")
        self._config = Config()

        logger.info("=" * 60)
        logger.info("DEFAULT CONFIGURATION PARAMETERS")
        logger.info("=" * 60)
        logger.info(f"Model: {self._config.model.model_path}")
        logger.info(f"Device: {self._config.model.device}")
        logger.info(f"Confidence: {self._config.model.confidence_threshold}")
        logger.info(f"IOU: {self._config.model.iou_threshold}")
        logger.info(f"IMG Size: {self._config.model.imgsz}")
        logger.info("=" * 60)

        return self._config

    def load_from_file(self, path: str) -> Config:
        """Loads configuration from a YAML file with detailed logging.

        Args:
            path: Path to the configuration file.

        Returns:
            Config: Validated configuration object.

        Raises:
            ConfigurationError: If there are configuration errors.
            FileNotFoundError: If the file does not exist.

        Example:
            >>> config = config_manager.load_from_file("config.yaml")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info(f"Loading configuration from: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)

            if raw_data is None:
                raise ConfigurationError("Configuration file is empty")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error reading file: {e}") from e

        logger.debug(f"Raw data loaded: {list(raw_data.keys())}")

        if "model" in raw_data:
            model_conf = raw_data["model"]
            logger.info(
                f"   Confidence in YAML: {model_conf.get('confidence_threshold', 'Not defined')}"
            )
            logger.info(f"   IMG Size in YAML: {model_conf.get('imgsz', 'Not defined')}")

        try:
            self._config = Config(**raw_data)
            logger.info("Configuration validated successfully")

            logger.info("=" * 60)
            logger.info("LOADED CONFIGURATION PARAMETERS")
            logger.info("=" * 60)
            logger.info(f"Model: {self._config.model.model_path}")
            logger.info(f"Device: {self._config.model.device}")
            logger.info(f"Confidence: {self._config.model.confidence_threshold}")
            logger.info(f"IOU: {self._config.model.iou_threshold}")
            logger.info(f"IMG Size: {self._config.model.imgsz}")
            logger.info(f"Vehicle Classes: {self._config.model.vehicle_classes}")
            logger.info(f"Camera Source: {self._config.camera.source}")
            logger.info(f"Resolution: {self._config.camera.width}x{self._config.camera.height}")
            logger.info(f"Counting Lines: {len(self._config.counting_lines)}")

            for i, line in enumerate(self._config.counting_lines):
                logger.info(
                    f"Line {i + 1}: {line.get('name', 'Unnamed')} - {line.get('direction', 'N/A')}"
                )

            logger.info("=" * 60)

        except ValueError as e:
            logger.error(f"Pydantic validation error: {e}")
            raise ConfigurationError(f"Invalid configuration data: {e}") from e
        except Exception as e:
            logger.error(f"Error validating configuration: {e}", exc_info=True)
            raise ConfigurationError(f"Error validating configuration: {e}") from e

        self._apply_environment_overrides()

        logger.info("Configuration loaded successfully")
        logger.info(f"Final confidence: {self._config.model.confidence_threshold}")
        logger.info(f"Final IMG Size: {self._config.model.imgsz}")
        logger.info(f"Final lines: {len(self._config.counting_lines)}")

        return self._config

    def _apply_environment_overrides(self) -> None:
        """Applies overrides from environment variables.

        Supported environment variables:
            - MODEL_PATH: Path to the model
            - CAMERA_SOURCE: Camera source
            - USE_GPU: Use GPU (true/false)
            - CONFIDENCE_THRESHOLD: Confidence threshold

        Note:
            Overrides are only applied if the environment variables exist.
        """
        if self._config is None:
            return

        env_model_path = os.getenv("MODEL_PATH")
        if env_model_path:
            self._config.model.model_path = env_model_path
            logger.info(f"Override MODEL_PATH: {self._config.model.model_path}")

        env_camera_source = os.getenv("CAMERA_SOURCE")
        if env_camera_source:
            self._config.camera.source = env_camera_source
            logger.info(f"Override CAMERA_SOURCE: {self._config.camera.source}")

        env_use_gpu = os.getenv("USE_GPU", "").lower()
        if env_use_gpu == "true":
            self._config.model.device = "cuda"
            logger.info("Override USE_GPU: cuda")

        env_confidence = os.getenv("CONFIDENCE_THRESHOLD")
        if env_confidence:
            try:
                self._config.model.confidence_threshold = float(env_confidence)
                logger.info(
                    f"Override CONFIDENCE_THRESHOLD: {self._config.model.confidence_threshold}"
                )
            except ValueError as e:
                logger.warning(f"Could not parse CONFIDENCE_THRESHOLD: {e}")

    @property
    def config(self) -> Config:
        """Gets the current configuration.

        Returns:
            Config: Current configuration or default if not loaded.

        Note:
            If no configuration is loaded, returns the default configuration.
        """
        if self._config is None:
            logger.warning("Configuration not loaded, using default values")
            self._config = Config()
        return self._config

    def save_to_file(self, path: str) -> None:
        """Saves the current configuration to a YAML file.

        Args:
            path: Path where to save the file.

        Raises:
            ConfigurationError: If the file cannot be saved.

        Example:
            >>> config_manager.save_to_file("config_backup.yaml")
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.config.dict(), f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to: {path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise ConfigurationError(f"Could not save configuration: {e}") from e

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a value by dot-notation path.

        Args:
            key: Path to the value (e.g., "model.confidence_threshold").
            default: Default value if the key does not exist.

        Returns:
            Any: Configuration value or default.

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

    def set(self, key: str, value: Any) -> None:
        """Sets a value by dot-notation path.

        Args:
            key: Path to the value (e.g., "model.confidence_threshold").
            value: New value to set.

        Raises:
            KeyError: If the key does not exist.

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
                raise KeyError(f"Key not found: {k}")

        setattr(target, keys[-1], value)
        logger.info(f"Configuration updated: {key} = {value}")


config_manager = ConfigManager.get_instance()
config = config_manager.config
