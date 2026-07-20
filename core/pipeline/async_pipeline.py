"""
Pipeline asíncrono.
"""
from typing import Optional, Callable

from core.pipeline.orchestrator import PipelineOrchestrator
from utils.logger import LoggerMixin

class AsyncVehicleCountingPipeline(LoggerMixin):
    """
    Pipeline asíncrono (wrapper alrededor del orquestador).
    Mantiene la misma interfaz que antes para compatibilidad.
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

    def start(self, source: Optional[str] = None) -> None:
        """Inicia el pipeline."""
        self._orchestrator.start(source)

    def stop(self) -> None:
        """Detiene el pipeline."""
        self._orchestrator.stop()

    def pause(self) -> None:
        """Pausa el pipeline."""
        self._orchestrator.pause()

    def resume(self) -> None:
        """Reanuda el pipeline."""
        self._orchestrator.resume()

    def get_stats(self) -> dict:
        """Obtiene estadísticas del pipeline."""
        return self._orchestrator.get_stats()

    @property
    def is_running(self) -> bool:
        return self._orchestrator.is_running

    @property
    def is_paused(self) -> bool:
        return self._orchestrator.is_paused

    @property
    def fps(self) -> float:
        stats = self.get_stats()
        return stats.get('capture', {}).get('fps', 0.0)
