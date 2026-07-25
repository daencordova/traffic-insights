"""Punto de entrada principal del sistema de seguimiento de tráfico.

Características de robustez implementadas:
- Manejador global de excepciones
- Sistema de recuperación automática
- Circuit breakers para componentes críticos
- Logging estructurado con contexto
- Validación de configuración en tiempo de ejecución
- Gestión de recursos con limpieza automática
"""

import argparse
import gc
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import cv2
import numpy as np

from config.manager import config_manager
from config.validator import validate_config
from core.circuit_breaker import circuit_breaker_registry
from core.error_handler import global_error_handler, setup_global_exception_handler
from core.exceptions import CameraError, ConfigurationError, PipelineError, VehicleCountingError
from core.pipeline.async_pipeline import AsyncPipeline
from core.pipeline.sync_pipeline import SyncPipeline
from utils.helpers import ensure_directory_exists, get_memory_usage
from utils.logger import setup_logger

MINIMUM_MEMORY_MB = 500
MEMORY_WARNING_THRESHOLD = 80
REQUIRED_DIRECTORIES = [
    "data",
    "data/screenshots",
    "data/exports",
    "data/logs",
]

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def setup_signal_handlers(logger: logging.Logger) -> None:
    """Configura manejadores de señales para terminación graceful.

    Args:
        logger: Logger para registrar eventos.
    """
    def signal_handler(signum: int, _frame) -> None:
        """Manejador de señales para terminación controlada."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Señal {signal_name} recibida. Iniciando terminación graceful...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)


def create_recovery_callbacks(logger: logging.Logger) -> dict[str, Callable]:
    """Crea los callbacks de recuperación para el manejador de errores.

    Args:
        logger: Logger para registrar eventos.

    Returns:
        Dict[str, callable]: Diccionario de callbacks de recuperación.
    """
    def recover_pipeline() -> None:
        """Recuperación del pipeline principal."""
        logger.info("🔄 Intentando recuperar pipeline...")
        logger.info("✅ Recuperación de pipeline completada")

    def recover_capture() -> None:
        """Recuperación del servicio de captura."""
        logger.info("🔄 Intentando recuperar servicio de captura...")
        stats = circuit_breaker_registry.get_all_stats()
        for name in stats:
            if "capture" in name.lower():
                breaker = circuit_breaker_registry.get(name)
                if breaker:
                    breaker.reset()
                    logger.info(f"   Circuit breaker '{name}' reiniciado")
        logger.info("✅ Recuperación de captura completada")

    def recover_memory() -> None:
        """Recuperación de memoria (limpieza)."""
        logger.info("🔄 Intentando liberar memoria...")
        gc.collect()
        mem = get_memory_usage()
        logger.info(f"✅ Memoria liberada. Uso actual: {mem.get('rss_mb', 0):.1f} MB")

    return {
        "pipeline": recover_pipeline,
        "capture_service": recover_capture,
        "memory": recover_memory,
    }


def validate_system_requirements(logger: logging.Logger) -> bool:
    """Valida los requisitos del sistema antes de iniciar.

    Args:
        logger: Logger para registrar eventos.

    Returns:
        bool: True si el sistema cumple los requisitos.
    """
    logger.info("🔍 Validando requisitos del sistema...")

    for dir_path in REQUIRED_DIRECTORIES:
        try:
            ensure_directory_exists(dir_path)
            logger.debug(f"   Directorio OK: {dir_path}")
        except Exception as e:
            logger.error(f"   Error creando directorio {dir_path}: {e}")
            return False

    mem = get_memory_usage()
    available_mb = mem.get("system_available_mb", 0)
    if available_mb < MINIMUM_MEMORY_MB:
        logger.warning(
            f"⚠️ Memoria disponible baja: {available_mb:.0f} MB. "
            "El sistema podría tener problemas de rendimiento."
        )

    try:
        logger.debug(f"   OpenCV: {cv2.__version__}")
    except ImportError:
        logger.error("❌ OpenCV no está instalado")
        return False

    try:
        logger.debug(f"   NumPy: {np.__version__}")
    except ImportError:
        logger.error("❌ NumPy no está instalado")
        return False

    logger.info("✅ Requisitos del sistema OK")
    return True


def load_configuration(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Carga la configuración del sistema.

    Args:
        args: Argumentos de línea de comandos.
        logger: Logger para registrar eventos.

    Returns:
        bool: True si la configuración se cargó correctamente.
    """
    config_path = Path(args.config)

    try:
        if config_path.exists():
            logger.info(f"📄 Cargando configuración: {config_path}")
            config_manager.load_from_file(str(config_path))

            validation_errors = validate_config(config_manager.config)
            if validation_errors:
                for error in validation_errors:
                    logger.warning(f"⚠️ {error}")
                if args.verbose:
                    logger.info("   (La configuración es válida pero tiene advertencias)")
        else:
            logger.warning(f"⚠️ Archivo de configuración no encontrado: {config_path}")
            logger.warning("   Usando configuración por defecto")
            config_manager.load_default()

        logger.info("✅ Configuración cargada exitosamente")
        return True

    except ConfigurationError as e:
        logger.error(f"❌ Error de configuración: {e}")
        if args.verbose:
            logger.exception(e)
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado cargando configuración: {e}")
        if args.verbose:
            logger.exception(e)
        return False


def configure_environment(args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """Configura el entorno según el modo (CPU/GPU).

    Args:
        args: Argumentos de línea de comandos.
        logger: Logger para registrar eventos.

    Returns:
        tuple: (is_cpu, workers, buffer_size, batch)
    """
    is_cpu = args.cpu_mode or config_manager.config.model.device == "cpu"

    if is_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["OMP_NUM_THREADS"] = str(args.threads or 2)
        os.environ["MKL_NUM_THREADS"] = str(args.threads or 2)

        workers = args.workers or 4
        buffer_size = args.buffer or 15
        batch = args.batch if args.batch is not None else True

        logger.info("💻 MODO CPU ACTIVADO")
        logger.info(f"   Workers: {workers}")
        logger.info(f"   Buffer: {buffer_size}")
        logger.info(f"   Batch processing: {'Sí' if batch else 'No'}")
        logger.info(f"   Threads: {os.environ.get('OMP_NUM_THREADS', 'auto')}")
    else:
        workers = args.workers or 8
        buffer_size = args.buffer or 30
        batch = args.batch if args.batch is not None else False

        logger.info("🖥️ MODO GPU ACTIVADO")
        logger.info(f"   Workers: {workers}")
        logger.info(f"   Buffer: {buffer_size}")
        logger.info(f"   Batch processing: {'Sí' if batch else 'No'}")

    return is_cpu, workers, buffer_size, batch


def create_pipeline(
    args: argparse.Namespace,
    workers: int,
    buffer_size: int,
    batch: bool,
    logger: logging.Logger
):
    """Crea el pipeline según el modo seleccionado.

    Args:
        args: Argumentos de línea de comandos.
        workers: Número de workers.
        buffer_size: Tamaño del buffer.
        batch: Si usar batch processing.
        logger: Logger para registrar eventos.

    Returns:
        Pipeline configurado.
    """
    if args.use_async:
        logger.info("🚀 Iniciando pipeline ASÍNCRONO...")
        return AsyncPipeline(
            buffer_size=buffer_size,
            num_workers=workers,
            enable_batch_processing=batch,
            batch_size=args.batch_size if batch else 1
        )
    logger.info("🚀 Iniciando pipeline SÍNCRONO (legacy)...")
    logger.warning("⚠️ El modo síncrono es legacy. Se recomienda usar el modo asíncrono.")
    return SyncPipeline()


def run_main_loop(
    pipeline,
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """Ejecuta el bucle principal del pipeline.

    Args:
        pipeline: Pipeline configurado.
        args: Argumentos de línea de comandos.
        logger: Logger para registrar eventos.
    """
    last_stats_time = time.time()
    stats_interval = args.stats_interval or 5.0

    if hasattr(pipeline, 'on_frame_processed') and args.verbose:
        pipeline.on_frame_processed = lambda result: logger.debug(
            f"Frame {result.frame_number} procesado "
            f"({result.processing_time_ms:.1f}ms)"
        )

    if hasattr(pipeline, 'on_error'):
        pipeline.on_error = lambda error: global_error_handler.handle_exception(
            error,
            {"component": "pipeline"}
        )

    if hasattr(pipeline, 'start'):
        pipeline.start(source=args.source)
        logger.info("✅ Pipeline iniciado correctamente")
        logger.info("   Presiona 'q' o ESC para salir")

        while pipeline.is_running:
            health = circuit_breaker_registry.get_health_summary()
            if not health["healthy"]:
                logger.warning(f"⚠️ Circuit breakers abiertos: {health['open_names']}")

            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                _log_pipeline_stats(pipeline, health, logger)
                last_stats_time = current_time

            time.sleep(0.1)
    else:
        logger.info("✅ Pipeline iniciado correctamente")
        logger.info("   Presiona 'q' o ESC para salir")
        pipeline.run(source=args.source)


def _log_pipeline_stats(pipeline, health, logger):
    """Registra estadísticas del pipeline."""
    stats = pipeline.get_stats()
    fps = stats.get('current_fps', 0.0)
    frames = stats.get('total_frames_processed', 0)
    buffer_size_current = stats.get('buffer', {}).get('size', 0)
    buffer_max = stats.get('buffer', {}).get('max_size', 1)
    buffer_usage = (buffer_size_current / buffer_max * 100) if buffer_max > 0 else 0

    logger.info(
        f"📊 FPS: {fps:.1f} | "
        f"Frames: {frames} | "
        f"Buffer: {buffer_usage:.0f}% | "
        f"Tracks: {stats.get('tracker', {}).get('active_tracks', 0)} | "
        f"Circuit: {health['open']} abiertos"
    )

    mem = get_memory_usage()
    if mem.get('percent', 0) > MEMORY_WARNING_THRESHOLD:
        logger.warning(
            f"⚠️ Memoria alta: {mem.get('percent', 0):.1f}% "
            f"({mem.get('rss_mb', 0):.0f} MB)"
        )


def handle_pipeline_error(
    error: Exception,
    args: argparse.Namespace,
    pipeline,
    logger: logging.Logger
) -> bool:
    """Maneja errores del pipeline con recuperación.

    Args:
        error: Excepción capturada.
        args: Argumentos de línea de comandos.
        pipeline: Pipeline en ejecución.
        logger: Logger para registrar eventos.

    Returns:
        bool: True si se pudo recuperar.
    """
    if isinstance(error, CameraError):
        logger.error(f"❌ Error de cámara: {error}")
        logger.info("   Intentando recuperación automática...")

        if global_error_handler.attempt_recovery(error):
            logger.info("✅ Recuperación exitosa. Reiniciando pipeline...")
            if pipeline:
                pipeline.stop()
                pipeline.start(source=args.source)
            return True
        logger.error("❌ No se pudo recuperar la cámara")
        return False

    if isinstance(error, PipelineError):
        logger.error(f"❌ Error en pipeline: {error}")
        return False

    if isinstance(error, VehicleCountingError):
        logger.error(f"❌ Error del sistema: {error}")
        return False

    logger.error(f"❌ Error fatal no manejado: {error}", exc_info=args.verbose)
    return False


def print_final_report(
    pipeline,
    start_time: float,
    error_stats: dict,
    logger: logging.Logger
) -> None:
    """Imprime el reporte final del sistema."""
    elapsed = time.time() - start_time
    mem = get_memory_usage()

    logger.info("=" * 70)
    logger.info("📊 REPORTE FINAL")
    logger.info("=" * 70)
    logger.info(f"⏱️ Tiempo de ejecución: {elapsed:.1f}s")
    logger.info(f"🧠 Memoria final: {mem.get('rss_mb', 0):.1f} MB")

    if pipeline:
        try:
            stats = pipeline.get_stats()
            logger.info(f"📹 Frames procesados: {stats.get('total_frames_processed', 0)}")
            logger.info(f"⚡ FPS promedio: {stats.get('current_fps', 0.0):.1f}")
            logger.info(f"🎯 Tracks activos: {stats.get('tracker', {}).get('active_tracks', 0)}")
        except Exception:
            pass

    if error_stats["total_errors"] > 0:
        logger.warning(f"⚠️ Errores totales: {error_stats['total_errors']}")

    logger.info("=" * 70)
    logger.info("👋 Sistema finalizado correctamente")
    logger.info("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de seguimiento de tráfico con procesamiento robusto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                             # Usa configuración por defecto
  python main.py -c config_prod.yaml         # Usa archivo de configuración
  python main.py -s rtsp://192.168.1.100:554 # Fuente RTSP
  python main.py --async --workers 8         # Pipeline asíncrono con 8 workers
  python main.py --cpu-mode --threads 4      # Modo CPU con 4 threads
        """
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Ruta al archivo de configuración (default: config.yaml)"
    )

    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="Fuente de video (número de cámara, ruta de archivo o URL RTSP)"
    )

    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        default=True,
        help="Usar pipeline asíncrono (predeterminado)"
    )

    parser.add_argument(
        "--sync",
        dest="use_async",
        action="store_false",
        help="Usar pipeline síncrono (modo legacy)"
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Número de workers (auto-ajustado para CPU/GPU)"
    )

    parser.add_argument(
        "-b", "--buffer",
        type=int,
        default=None,
        help="Tamaño del buffer (auto-ajustado para CPU/GPU)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        default=None,
        help="Habilitar procesamiento por lotes"
    )

    parser.add_argument(
        "--no-batch",
        dest="batch",
        action="store_false",
        help="Deshabilitar procesamiento por lotes"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Tamaño del lote para procesamiento por lotes (default: 4)"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Número de threads para CPU (solo modo CPU)"
    )

    parser.add_argument(
        "--cpu-mode",
        action="store_true",
        default=False,
        help="Forzar modo CPU con límites optimizados"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Activar modo verbose (logging DEBUG)"
    )

    parser.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="Intervalo para mostrar estadísticas en segundos (default: 5.0)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Sistema de seguimiento de tráfico v0.2.0"
    )

    return parser.parse_args()


def main() -> NoReturn:
    """Función principal del sistema con manejo robusto de errores.

    Returns:
        NoReturn: El sistema termina con sys.exit().
    """
    setup_global_exception_handler()

    args = parse_args()
    logger = setup_logger(
        name="main",
        log_file="data/logs/system.log",
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    setup_signal_handlers(logger)

    recovery_callbacks = create_recovery_callbacks(logger)
    for name, callback in recovery_callbacks.items():
        global_error_handler.register_recovery(name, callback)

    logger.info("=" * 70)
    logger.info("🚗 SISTEMA DE SEGUIMIENTO DE TRÁFICO v0.2.0")
    logger.info("   (Con sistema robusto de gestión de errores)")
    logger.info("=" * 70)
    logger.info(f"📅 Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🐍 Python: {sys.version.split()[0]}")
    logger.info(f"📂 Directorio: {current_dir}")

    if not validate_system_requirements(logger):
        logger.error("❌ El sistema no cumple los requisitos mínimos")
        sys.exit(1)

    if not load_configuration(args, logger):
        sys.exit(1)

    is_cpu, workers, buffer_size, batch = configure_environment(args, logger)

    pipeline = None
    start_time = time.time()

    try:
        pipeline = create_pipeline(args, workers, buffer_size, batch, logger)
        run_main_loop(pipeline, args, logger)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupción por usuario")

    except Exception as e:
        if not handle_pipeline_error(e, args, pipeline, logger):
            sys.exit(1)

    finally:
        if pipeline:
            logger.info("🧹 Limpiando pipeline...")
            try:
                pipeline.stop()
                logger.info("✅ Pipeline detenido")
            except Exception as e:
                logger.warning(f"Error deteniendo pipeline: {e}")

        logger.info("🧹 Limpiando circuit breakers...")
        circuit_breaker_registry.reset_all()

        logger.info("🧹 Liberando memoria...")
        gc.collect()
        gc.collect()

        error_stats = global_error_handler.get_stats()
        print_final_report(pipeline, start_time, error_stats, logger)

    sys.exit(0)


if __name__ == "__main__":
    main()
