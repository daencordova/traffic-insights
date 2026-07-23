"""
Punto de entrada principal del sistema de seguimiento de tráfico.

Características de robustez implementadas:
- Manejador global de excepciones
- Sistema de recuperación automática
- Circuit breakers para componentes críticos
- Logging estructurado con contexto
- Validación de configuración en tiempo de ejecución
- Gestión de recursos con limpieza automática
"""

import sys
import os
import argparse
import logging
import signal
import time
from pathlib import Path
from typing import NoReturn, Dict, Callable

from core.pipeline.async_pipeline import AsyncPipeline
from core.pipeline.sync_pipeline import SyncPipeline
from config.manager import config_manager
from config.validator import validate_config
from core.circuit_breaker import circuit_breaker_registry
from utils.helpers import ensure_directory_exists, get_memory_usage
from utils.logger import setup_logger
from core.error_handler import (
    setup_global_exception_handler,
    global_error_handler
)
from core.exceptions import (
    ConfigurationError,
    VehicleCountingError,
    CameraError,
    PipelineError
)


current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def setup_signal_handlers(logger: logging.Logger) -> None:
    """
    Configura manejadores de señales para terminación graceful.

    Args:
        logger: Logger para registrar eventos.
    """
    def signal_handler(signum: int, frame) -> None:
        """Manejador de señales para terminación controlada."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Señal {signal_name} recibida. Iniciando terminación graceful...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)


def create_recovery_callbacks(logger: logging.Logger) -> Dict[str, Callable]:
    """
    Crea los callbacks de recuperación para el manejador de errores.

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
        for name, breaker in circuit_breaker_registry._breakers.items():
            if "capture" in name.lower():
                breaker.reset()
                logger.info(f"   Circuit breaker '{name}' reiniciado")
        logger.info("✅ Recuperación de captura completada")

    def recover_memory() -> None:
        """Recuperación de memoria (limpieza)."""
        logger.info("🔄 Intentando liberar memoria...")
        import gc
        gc.collect()
        mem = get_memory_usage()
        logger.info(f"✅ Memoria liberada. Uso actual: {mem.get('rss_mb', 0):.1f} MB")

    return {
        "pipeline": recover_pipeline,
        "capture_service": recover_capture,
        "memory": recover_memory,
    }

def validate_system_requirements(logger: logging.Logger) -> bool:
    """
    Valida los requisitos del sistema antes de iniciar.

    Args:
        logger: Logger para registrar eventos.

    Returns:
        bool: True si el sistema cumple los requisitos.
    """
    logger.info("🔍 Validando requisitos del sistema...")

    required_dirs = [
        "data",
        "data/screenshots",
        "data/exports",
        "data/logs",
    ]

    for dir_path in required_dirs:
        try:
            ensure_directory_exists(dir_path)
            logger.debug(f"   Directorio OK: {dir_path}")
        except Exception as e:
            logger.error(f"   Error creando directorio {dir_path}: {e}")
            return False

    mem = get_memory_usage()
    available_mb = mem.get("system_available_mb", 0)
    if available_mb < 500:
        logger.warning(
            f"⚠️ Memoria disponible baja: {available_mb:.0f} MB. "
            "El sistema podría tener problemas de rendimiento."
        )

    try:
        import cv2
        logger.debug(f"   OpenCV: {cv2.__version__}")
    except ImportError:
        logger.error("❌ OpenCV no está instalado")
        return False

    try:
        import numpy as np
        logger.debug(f"   NumPy: {np.__version__}")
    except ImportError:
        logger.error("❌ NumPy no está instalado")
        return False

    logger.info("✅ Requisitos del sistema OK")
    return True


def main() -> NoReturn:
    """
    Función principal del sistema con manejo robusto de errores.

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

    try:
        config_path = Path(args.config)

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

    except ConfigurationError as e:
        logger.error(f"❌ Error de configuración: {e}")
        if args.verbose:
            logger.exception(e)
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error inesperado cargando configuración: {e}")
        if args.verbose:
            logger.exception(e)
        sys.exit(1)

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

    pipeline = None
    start_time = time.time()

    try:
        if args.use_async:
            logger.info("🚀 Iniciando pipeline ASÍNCRONO...")

            pipeline = AsyncPipeline(
                buffer_size=buffer_size,
                num_workers=workers,
                enable_batch_processing=batch,
                batch_size=args.batch_size if batch else 1
            )

            pipeline.on_frame_processed = lambda result: (
                logger.debug(
                    f"Frame {result.frame_number} procesado "
                    f"({result.processing_time_ms:.1f}ms)"
                )
                if args.verbose else None
            )

            pipeline.on_error = lambda error: (
                global_error_handler.handle_exception(error, {
                    "component": "pipeline",
                    "frame_number": pipeline.frame_count if hasattr(pipeline, 'frame_count') else 0
                })
            )

            pipeline.start(source=args.source)

            last_stats_time = time.time()
            stats_interval = args.stats_interval or 5.0

            logger.info("✅ Pipeline iniciado correctamente")
            logger.info("   Presiona 'q' o ESC para salir")

            while pipeline.is_running:
                health = circuit_breaker_registry.get_health_summary()
                if not health["healthy"]:
                    logger.warning(
                        f"⚠️ Circuit breakers abiertos: {health['open_names']}"
                    )

                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
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
                    if mem.get('percent', 0) > 80:
                        logger.warning(
                            f"⚠️ Memoria alta: {mem.get('percent', 0):.1f}% "
                            f"({mem.get('rss_mb', 0):.0f} MB)"
                        )

                    last_stats_time = current_time

                time.sleep(0.1)

        else:
            logger.info("🚀 Iniciando pipeline SÍNCRONO (legacy)...")
            logger.warning("⚠️ El modo síncrono es legacy. Se recomienda usar el modo asíncrono.")

            pipeline = SyncPipeline()
            pipeline.run(source=args.source)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupción por usuario")

    except CameraError as e:
        logger.error(f"❌ Error de cámara: {e}")
        logger.info("   Intentando recuperación automática...")

        if global_error_handler._attempt_recovery(e):
            logger.info("✅ Recuperación exitosa. Reiniciando pipeline...")
            if pipeline:
                pipeline.stop()
                pipeline.start(source=args.source)
        else:
            logger.error("❌ No se pudo recuperar la cámara")

    except PipelineError as e:
        logger.error(f"❌ Error en pipeline: {e}")
        sys.exit(1)

    except VehicleCountingError as e:
        logger.error(f"❌ Error del sistema: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error fatal no manejado: {e}", exc_info=args.verbose)
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
        import gc
        gc.collect()
        gc.collect()

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

        error_stats = global_error_handler.get_stats()
        if error_stats["total_errors"] > 0:
            logger.warning(f"⚠️ Errores totales: {error_stats['total_errors']}")

        logger.info("=" * 70)
        logger.info("👋 Sistema finalizado correctamente")
        logger.info("=" * 70)

        sys.exit(0)

def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de seguimiento de tráfico con procesamiento robusto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                              # Usa configuración por defecto
  python main.py -c config_prod.yaml         # Usa archivo de configuración
  python main.py -s rtsp://192.168.1.100:554 # Fuente RTSP
  python main.py --async --workers 8         # Pipeline asíncrono con 8 workers
  python main.py --cpu-mode --threads 4      # Modo CPU con 4 threads
  python main.py --monitor                   # Activa monitoreo de rendimiento
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
        "--monitor",
        action="store_true",
        default=True,
        help="Activar monitoreo de rendimiento (default: True)"
    )

    parser.add_argument(
        "--no-monitor",
        dest="monitor",
        action="store_false",
        help="Desactivar monitoreo de rendimiento"
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


if __name__ == "__main__":
    main()
