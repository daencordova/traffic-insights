"""
Pipeline asíncrono para procesamiento de video en tiempo real.

Este módulo implementa el pipeline asíncrono del sistema que permite
el procesamiento paralelo de frames para máximo rendimiento.

El pipeline asíncrono utiliza un orquestador que coordina:
- Captura de video (CaptureService)
- Procesamiento de frames (ProcessingService)
- Renderizado de visualización (RenderService)
- Control de usuario (ControlService)
- Monitoreo de rendimiento (MonitoringService)
"""

from __future__ import annotations

from typing import Optional, Callable

from core.pipeline.orchestrator import PipelineOrchestrator
from utils.logger import LoggerMixin


class AsyncVehicleCountingPipeline(LoggerMixin):
    """
    Pipeline asíncrono para procesamiento de video en tiempo real.

    Este pipeline utiliza múltiples workers y buffers para procesar
    frames de forma paralela, maximizando el rendimiento en sistemas
    multi-núcleo.

    Características:
        - Procesamiento paralelo de frames
        - Buffer circular para captura y procesamiento
        - Múltiples workers para procesamiento
        - Procesamiento por lotes (batch)
        - Monitoreo de rendimiento en tiempo real
        - Manejo robusto de errores

    Attributes:
        config: Configuración del sistema.
        _orchestrator: Orquestador del pipeline.

    Example:
        >>> pipeline = AsyncVehicleCountingPipeline(
        ...     buffer_size=30,
        ...     num_workers=4,
        ...     enable_batch_processing=True
        ... )
        >>> pipeline.start(source="0")
        >>> # Presiona 'q' para salir
        >>> pipeline.stop()
    """

    def __init__(
        self,
        detector=None,
        tracker=None,
        counter=None,
        buffer_size: int = 30,
        num_workers: int = 4,
        enable_batch_processing: bool = False,
        batch_size: int = 4,
        render_callback: Optional[Callable] = None,
    ):
        """
        Inicializa el pipeline asíncrono.

        Args:
            detector: Detector de objetos (opcional).
            tracker: Tracker de objetos (opcional).
            counter: Contador de vehículos (opcional).
            buffer_size: Tamaño del buffer de frames.
            num_workers: Número de workers de procesamiento.
            enable_batch_processing: Habilitar procesamiento por lotes.
            batch_size: Tamaño del lote para batch processing.
            render_callback: Callback para renderizado personalizado.

        Note:
            En modo CPU, los límites se ajustan automáticamente:
            - buffer_size: máximo 20
            - num_workers: máximo 4
        """
        from config.manager import config_manager
        self.config = config_manager.config
        self.logger.info("Inicializando AsyncVehicleCountingPipeline")

        is_cpu = self.config.model.device == "cpu"
        if is_cpu:
            buffer_size = min(buffer_size, 20)
            num_workers = min(num_workers, 4)
            self.logger.info("Modo CPU - límites ajustados")

        self._init_components(detector, tracker, counter, render_callback)

        self._orchestrator = PipelineOrchestrator(
            config=self.config,
            detector=self.detector,
            tracker=self.tracker,
            counter=self.counter,
            renderer=self.renderer,
            controls=self.controls
        )

        self.logger.info("Pipeline asíncrono inicializado")

    def _init_components(self, detector, tracker, counter, render_callback):
        """
        Inicializa los componentes del pipeline.

        Args:
            detector: Detector de objetos (opcional).
            tracker: Tracker de objetos (opcional).
            counter: Contador de vehículos (opcional).
            render_callback: Callback para renderizado personalizado.

        Note:
            Si algún componente no se proporciona, se crea automáticamente
            con la configuración global del sistema.
        """
        from core.detector import YOLODetector
        from core.tracker import AdvancedTracker
        from core.counter import VehicleCounter
        from core.pipeline.renderer import FrameRenderer
        from core.pipeline.controls import ControlHandler

        use_optimized = getattr(
            self.config.optimization,
            "use_optimized_detector",
            True
        )

        if use_optimized:
            try:
                from core.detector import OptimizedYOLODetector
                self.detector = detector or OptimizedYOLODetector()
                self.logger.info("✅ Detector optimizado activado")
            except Exception as e:
                self.logger.warning(f"Detector optimizado no disponible: {e}")
                self.detector = detector or YOLODetector()
        else:
            self.detector = detector or YOLODetector()

        self.tracker = tracker or AdvancedTracker()
        self.counter = counter or VehicleCounter()
        self.renderer = FrameRenderer(self.config)
        self.controls = ControlHandler(self.config)

        if render_callback:
            self.renderer._custom_render = render_callback

    def start(self, source: Optional[str] = None) -> None:
        """
        Inicia el pipeline.

        Args:
            source: Fuente de video (número de cámara, archivo o URL RTSP).
                Si es None, se usa la fuente de la configuración.

        Example:
            >>> pipeline.start("rtsp://192.168.1.100:554/stream")
            >>> # o
            >>> pipeline.start()  # Usa la fuente de config.yaml
        """
        self._orchestrator.start(source)

    def stop(self) -> None:
        """Detiene el pipeline y libera recursos."""
        self._orchestrator.stop()

    def pause(self) -> None:
        """Pausa la ejecución del pipeline."""
        self._orchestrator.pause()

    def resume(self) -> None:
        """Reanuda la ejecución del pipeline."""
        self._orchestrator.resume()

    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del pipeline.

        Returns:
            dict: Estadísticas del pipeline incluyendo:
                - state: Estado del pipeline
                - is_paused: Si está pausado
                - is_running: Si está en ejecución
                - uptime_seconds: Tiempo de ejecución
                - capture: Estadísticas de captura
                - processing: Estadísticas de procesamiento
                - render: Estadísticas de renderizado
                - monitoring: Estadísticas de monitoreo

        Example:
            >>> stats = pipeline.get_stats()
            >>> print(f"FPS: {stats['capture']['fps']:.1f}")
            >>> print(f"Frames procesados: {stats['processing']['processed_count']}")
        """
        return self._orchestrator.get_stats()

    @property
    def is_running(self) -> bool:
        """Indica si el pipeline está en ejecución."""
        return self._orchestrator.is_running

    @property
    def is_paused(self) -> bool:
        """Indica si el pipeline está pausado."""
        return self._orchestrator.is_paused

    @property
    def fps(self) -> float:
        """
        FPS actual del pipeline.

        Returns:
            float: FPS calculado a partir de las estadísticas de captura.
        """
        stats = self.get_stats()
        return stats.get('capture', {}).get('fps', 0.0)
