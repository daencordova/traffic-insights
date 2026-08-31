"""
Configuration module for the application.

This module provides access to application configuration settings through
a unified interface. It exports the Config class, configuration manager,
and a pre-initialized config instance for immediate use.

The configuration system supports:
- Environment-based configuration loading
- Default values with override capabilities
- Type-safe access to configuration parameters
- Centralized configuration management

Example:
    >>> from config import config
    >>> # Access configuration values
    >>> debug_mode = config.get("debug", False)
    >>> database_url = config.get("database.url")

    >>> # Or use the manager for more control
    >>> from config import config_manager
    >>> config_manager.reload()  # Reload configuration from sources
"""

from config.manager import config, config_manager
from config.settings import Config

__all__ = [
    "Config",
    "config_manager",
    "config",
]
