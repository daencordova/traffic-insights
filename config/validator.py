"""
Validador de configuración con validación cruzada.
"""

from pathlib import Path
from typing import List
import logging

from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def validate_config(config) -> List[str]:
    """
    Valida la configuración del sistema.

    Args:
        config: Objeto de configuración Pydantic.

    Returns:
        List[str]: Lista de advertencias (vacío si todo OK).
    """
    warnings = []

    output_dirs = [
        ("screenshots", config.output.screenshots_dir),
        ("exports", config.output.export_dir),
        ("logs", config.output.logs_dir),
    ]

    for name, path in output_dirs:
        dir_path = Path(path)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directorio creado: {path}")
            except OSError as e:
                raise ConfigurationError(
                    f"No se pudo crear el directorio '{name}': {e}",
                    details={"path": str(path), "error": str(e)}
                )

    model_path = Path(config.model.model_path)
    if not model_path.exists():
        if model_path.name.endswith('.pt') and model_path.name.startswith('yolo'):
            try:
                from ultralytics import YOLO
                logger.info(f"Descargando modelo {model_path.name}...")
                YOLO(model_path.name)
                logger.info(f"Modelo {model_path.name} descargado")
            except Exception as e:
                raise ConfigurationError(
                    f"Modelo no encontrado y no se pudo descargar: {model_path}",
                    details={"model_path": str(model_path), "error": str(e)}
                )
        else:
            raise ConfigurationError(
                f"Archivo de modelo no encontrado: {model_path}",
                details={"model_path": str(model_path)}
            )

    if config.model.device == "cpu" and config.tracker.enable_reidentification:
        warnings.append(
            "Re-identificación con features en CPU puede ser muy lenta. "
            "Considere desactivar 'enable_reidentification' o usar GPU."
        )

    memory_per_frame = config.camera.width * config.camera.height * 3
    buffer_memory_mb = (memory_per_frame * config.camera.buffer_size) / (1024 * 1024)
    if buffer_memory_mb > 500:
        warnings.append(
            f"El buffer de {config.camera.buffer_size} frames usa "
            f"~{buffer_memory_mb:.0f} MB de RAM. Considere reducirlo."
        )

    if config.model.imgsz != config.camera.width:
        warnings.append(
            f"El modelo espera {config.model.imgsz}x{config.model.imgsz} pero la "
            f"cámara proporciona {config.camera.width}x{config.camera.height}. "
            "Se redimensionará automáticamente, pero puede afectar el rendimiento."
        )

    if not config.counting_lines:
        warnings.append(
            "No hay líneas de conteo configuradas. El sistema no contará vehículos."
        )

    if config.camera.fps and config.camera.fps > 60:
        warnings.append(
            f"FPS configurado ({config.camera.fps}) es muy alto. "
            "Considere reducirlo para mejor rendimiento."
        )

    return warnings


def validate_config_required_fields(config) -> None:
    """
    Valida que todos los campos requeridos estén presentes.

    Args:
        config: Objeto de configuración Pydantic.

    Raises:
        ConfigurationError: Si falta algún campo requerido.
    """
    required_fields = [
        ("model", "model_path"),
        ("camera", "source"),
        ("tracker", "type"),
    ]

    missing = []
    for section, field in required_fields:
        obj = getattr(config, section, None)
        if obj is None:
            missing.append(f"{section}.{field}")
        elif not hasattr(obj, field) or getattr(obj, field) is None:
            missing.append(f"{section}.{field}")

    if missing:
        raise ConfigurationError(
            f"Campos requeridos faltantes: {', '.join(missing)}",
            details={"missing_fields": missing}
        )
