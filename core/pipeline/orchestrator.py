"""Orquestador del pipeline."""

import time
from typing import TYPE_CHECKING, Any

from core.exceptions import CaptureError, FrameProcessingError, PipelineError
from core.pipeline.services.capture_service import CaptureService
from core.pipeline.services.control_service import ControlService
from core.pipeline.services.monitoring_service import MonitoringService
from core.pipeline.services.processing_service import ProcessingService
from core.pipeline.services.render_service import RenderService
from core.pipeline.state_manager import PipelineStateManager, PipelineStatus
from utils.logger import LoggerMixin

if TYPE_CHECKING:
    from collections.abc import Callable


class PipelineOrchestrator(LoggerMixin):
    """Orquestador del pipeline - responsable de coordinar los servicios."""

    def __init__(
        self,
        config,
        detector=None,
        tracker=None,
        counter=None,
        renderer=None,
        controls=None,
    ):
        self.config = config
        self._state_manager = PipelineStateManager()
        self._services: dict[str, Any] = {}
        self._event_handlers: dict[str, Callable] = {}
        self._is_running = False
        self._last_health_check = 0.0
        self._health_check_interval = 5.0

        self.logger.info("Inicializando PipelineOrchestrator")

        self._init_services(detector, tracker, counter, renderer, controls)
        self._setup_event_handlers()

        self.logger.info("PipelineOrchestrator inicializado")

    def _init_services(self, detector, tracker, counter, renderer, controls):
        """Inicializa todos los servicios."""
        self._services["capture"] = CaptureService(
            config=self.config,
            on_frame_captured=self._on_frame_captured,
            on_frame_dropped=self._on_frame_dropped,
        )

        self._services["processing"] = ProcessingService(
            config=self.config,
            detector=detector,
            tracker=tracker,
            counter=counter,
            on_frame_processed=self._on_frame_processed,
        )

        self._services["render"] = RenderService(
            config=self.config,
            renderer=renderer,
            on_key_pressed=self._on_key_pressed,
        )

        self._services["control"] = ControlService(
            config=self.config,
            controls=controls,
            on_pause_toggle=self._on_pause_toggle,
            on_reset=self._on_reset,
        )

        self._services["monitoring"] = MonitoringService(
            config=self.config,
            interval=5.0,
        )

    def _setup_event_handlers(self):
        """Configura los manejadores de eventos entre servicios."""
        self._event_handlers = {
            "frame_captured": self._services["processing"].enqueue_frame,
            "frame_processed": self._services["render"].enqueue_frame,
            "key_pressed": self._services["control"].handle_key,
            "pause_toggle": self._on_pause_toggle,
            "reset": self._on_reset,
        }

    def start(self, source: str | None = None) -> None:
        """Inicia el pipeline orquestando todos los servicios."""
        if self._is_running:
            self.logger.warning("Pipeline ya está en ejecución")
            return

        self.logger.info("Iniciando pipeline...")
        self._is_running = True

        if not self._state_manager.set_status(PipelineStatus.RUNNING):
            self.logger.error("No se pudo iniciar el pipeline")
            self._is_running = False
            raise PipelineError("Error al iniciar pipeline")

        try:
            self._services["capture"].start(source)
            self._services["processing"].start()
            self._services["render"].start()
            self._services["monitoring"].start()

            self.logger.info("Pipeline iniciado exitosamente")
            self._run_main_loop()

        except Exception as e:
            self.logger.error(f"Error iniciando pipeline: {e}", exc_info=True)
            self._state_manager.set_status(PipelineStatus.ERROR)
            self._is_running = False

            if self._state_manager.can_recover():
                self.logger.info("Intentando recuperación...")
                self._attempt_recovery()
            else:
                self.stop()
                raise PipelineError(f"Fallo crítico en pipeline: {e}") from e

    def _run_main_loop(self) -> None:
        """Bucle principal de supervisión. No bloquea, solo verifica el estado."""
        self.logger.info("Bucle de supervisión iniciado")

        while self._is_running and self._state_manager.is_running():
            try:
                self._apply_flow_control()

                self._check_health()

                self._services["monitoring"].update()

                time.sleep(0.01)

            except KeyboardInterrupt:
                self.logger.info("Interrupción recibida")
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"Error en bucle de supervisión: {e}", exc_info=True)
                self._handle_service_error(e)

    def _apply_flow_control(self) -> None:
        """Aplica control de flujo basado en el estado del pipeline."""
        if self._state_manager.is_paused():
            for service_name in ["capture", "processing", "render"]:
                self._services[service_name].pause()
        elif self._state_manager.is_running():
            for service_name in ["capture", "processing", "render"]:
                self._services[service_name].resume()

    def _check_health(self) -> None:
        """Verifica la salud del sistema periódicamente."""
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return

        self._last_health_check = current_time

        try:
            capture_stats = self._services["capture"].get_stats()

            if capture_stats.get("errors", 0) > 5:
                self.logger.warning("Demasiados errores en captura, intentando recuperar...")
                self._services["capture"].reconnect()

        except Exception as e:
            self.logger.error(f"Error en health check: {e}")

    def _handle_service_error(self, error: Exception) -> None:
        """Maneja errores de servicios."""
        if isinstance(error, CaptureError):
            self.logger.error(f"Error de captura: {error}")
            self._state_manager.record_error()

            if self._state_manager.can_recover():
                self.logger.info("Intentando recuperar captura...")
                self._services["capture"].reconnect()
                if self._state_manager.is_error():
                    self._state_manager.set_status(PipelineStatus.RUNNING)
            else:
                self.stop()

        elif isinstance(error, FrameProcessingError):
            self.logger.error(f"Error en procesamiento: {error}")
            self._services["processing"].reset()

        else:
            self.logger.error(f"Error inesperado: {error}", exc_info=True)

    def _attempt_recovery(self) -> bool:
        """Intenta recuperar el sistema de un error."""
        self._state_manager.mark_recovery_attempt()

        try:
            self.logger.info(f"Intento de recuperación {self._state_manager._recovery_attempts}")

            self._services["capture"].reconnect()
            self._services["processing"].reset()

            self._state_manager.reset_recovery_attempts()
            self._state_manager.set_status(PipelineStatus.RUNNING)

            self.logger.info("Recuperación exitosa")
            return True

        except Exception as e:
            self.logger.error(f"Fallo en recuperación: {e}")
            return False

    def _on_frame_captured(self, frame, metadata):
        """Callback: frame capturado."""
        if self._state_manager.is_running():
            self._services["processing"].enqueue_frame(frame, metadata)

    def _on_frame_processed(self, result):
        """Callback: frame procesado."""
        if self._state_manager.is_running():
            self._services["render"].enqueue_frame(result)

    def _on_frame_dropped(self, frame_number):
        """Callback: frame descartado."""
        self._services["monitoring"].record_dropped_frame()

    def _on_key_pressed(self, key):
        """Callback: tecla presionada."""
        self._services["control"].handle_key(key)

    def _on_pause_toggle(self, is_paused):
        """Callback: toggle de pausa."""
        if is_paused:
            self._state_manager.set_status(PipelineStatus.PAUSED)
        else:
            self._state_manager.set_status(PipelineStatus.RUNNING)
        self.logger.info(f"Pipeline {'pausado' if is_paused else 'reanudado'}")

    def _on_reset(self):
        """Callback: reinicio del sistema."""
        self.logger.info("Reiniciando sistema...")
        self._services["processing"].reset()
        self._services["monitoring"].reset()
        self._state_manager.reset_recovery_attempts()

    def pause(self) -> None:
        """Pausa la ejecución del pipeline."""
        if self._state_manager.can_transition_to(PipelineStatus.PAUSED):
            self._state_manager.set_status(PipelineStatus.PAUSED)
            self.logger.info("Pipeline pausado")

    def resume(self) -> None:
        """Reanuda la ejecución del pipeline."""
        if self._state_manager.can_transition_to(PipelineStatus.RUNNING):
            self._state_manager.set_status(PipelineStatus.RUNNING)
            self.logger.info("Pipeline reanudado")

    def stop(self) -> None:
        """Detiene la ejecución del pipeline."""
        self.logger.info("Deteniendo pipeline...")
        self._is_running = False
        self._state_manager.set_status(PipelineStatus.STOPPED)

        for service_name in ["monitoring", "render", "processing", "capture"]:
            try:
                if service_name in self._services:
                    self._services[service_name].stop()
            except Exception as e:
                self.logger.warning(f"Error deteniendo {service_name}: {e}")

        self.logger.info("Pipeline detenido")

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas combinadas de todos los servicios."""
        stats = {
            "state": self._state_manager.get_stats(),
            "is_running": self._is_running,
            "is_paused": self._state_manager.is_paused(),
        }

        for name, service in self._services.items():
            try:
                if hasattr(service, "get_stats"):
                    stats[name] = service.get_stats()
            except Exception:
                pass

        return stats

    @property
    def is_running(self) -> bool:
        return self._is_running and self._state_manager.is_running()

    @property
    def is_paused(self) -> bool:
        return self._state_manager.is_paused()

    @property
    def state(self) -> PipelineStatus:
        return self._state_manager.get_status()
