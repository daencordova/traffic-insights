"""
Punto de entrada principal con soporte para pipeline asíncrono
Optimizado para CPU con límites de recursos
"""

import sys
import argparse
from pathlib import Path
from typing import NoReturn
import logging
import os
import time

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config.manager import config_manager
from core.pipeline.async_pipeline import AsyncVehicleCountingPipeline
from utils.helpers import ensure_directory_exists
from utils.logger import setup_logger
from core.exceptions import ConfigurationError


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Sistema de seguimiento de trafico con procesamiento asíncrono"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="Fuente de video (cámara o archivo)"
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
        help="Número de workers (auto-ajustado para CPU)"
    )
    parser.add_argument(
        "-b", "--buffer",
        type=int,
        default=None,
        help="Tamaño del buffer (auto-ajustado para CPU)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=None,
        help="Habilitar procesamiento por lotes (auto-ajustado para CPU)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Tamaño del lote para procesamiento por lotes"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Activar modo verbose"
    )
    parser.add_argument(
        "--cpu-mode",
        action="store_true",
        default=False,
        help="Forzar modo CPU con límites optimizados"
    )
    return parser.parse_args()


def setup_environment(args):
    """Configura el entorno para el modo CPU."""
    if args.cpu_mode or args.workers is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        logger = logging.getLogger(__name__)
        logger.info("Modo CPU forzado")


def main() -> NoReturn:
    """Función principal con manejo de errores global."""
    args = parse_args()

    logger = setup_logger(
        name="main",
        log_file="data/logs/system.log",
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    logger.info("=" * 60)
    logger.info("🚗 SISTEMA DE SEGUIMIENTO DE TRAFICO v0.1.0")
    logger.info("   (Con procesamiento asíncrono optimizado)")
    logger.info("=" * 60)

    setup_environment(args)

    try:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                config_manager.load_from_file(str(config_path))
                logger.info("✅ Configuración cargada exitosamente")
            except Exception as e:
                logger.error(f"Error cargando configuración: {e}", exc_info=True)
                logger.warning("Usando configuración por defecto")
        else:
            logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            logger.warning("Usando configuración por defecto")

        ensure_directory_exists("data/screenshots")
        ensure_directory_exists("data/exports")
        ensure_directory_exists("data/logs")

        is_cpu = args.cpu_mode or config_manager.config.model.device == "cpu"

        if is_cpu:
            workers = args.workers or 4
            buffer_size = args.buffer or 15
            batch = args.batch if args.batch is not None else True
            logger.info(f"Modo CPU activado (workers={workers}, buffer={buffer_size})")
        else:
            workers = args.workers or 8
            buffer_size = args.buffer or 30
            batch = args.batch if args.batch is not None else False
            logger.info(f"Modo GPU activado (workers={workers}, buffer={buffer_size})")

        if args.use_async:
            logger.info(
                f"✅ Modo asíncrono (workers={workers}, buffer={buffer_size}, "
                f"batch={batch})"
            )

            with AsyncVehicleCountingPipeline(
                buffer_size=buffer_size,
                num_workers=workers,
                enable_batch_processing=batch,
                batch_size=args.batch_size if batch else 1
            ) as pipeline:

                def on_frame_processed(result):
                    if args.verbose:
                        logger.debug(
                            f"Frame {result.frame_number} procesado "
                            f"({result.processing_time_ms:.1f}ms)"
                        )

                pipeline.on_frame_processed = on_frame_processed

                pipeline.start(source=args.source)

                last_stats_time = time.time()
                while pipeline.is_running:
                    time.sleep(0.1)

                    current_time = time.time()
                    if current_time - last_stats_time >= 5.0:
                        stats = pipeline.get_stats()
                        fps_display = f"{stats['current_fps']:.1f}"
                        cpu_mode = "CPU" if stats['cpu_mode'] else "GPU"
                        logger.info(
                            f"📊 FPS: {fps_display} | "
                            f"Frames: {stats['total_frames_processed']} | "
                            f"Buffer: {stats['buffer']['size']}/{stats['buffer']['max_size']} | "
                            f"Modo: {cpu_mode}"
                        )
                        last_stats_time = current_time

        else:
            logger.warning("⚠️ Modo síncrono (legacy) - Se recomienda usar el modo asíncrono")
            from core.pipeline import VehicleCountingPipeline

            pipeline = VehicleCountingPipeline()
            pipeline.run(source=args.source)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupción por usuario")
        sys.exit(0)

    except ConfigurationError as e:
        logger.error(f"❌ Error de configuración: {e}", exc_info=True)
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("👋 Sistema finalizado correctamente")
    logger.info("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
