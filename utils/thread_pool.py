"""
Thread pool optimizado para procesamiento paralelo de frames
"""

import threading
import queue
import time
from typing import Callable, Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging


class TaskPriority(Enum):
    """Prioridades para tareas"""
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()


@dataclass
class Task:
    """Representa una tarea para ejecutar en el pool"""
    id: int
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None

    @property
    def wait_time_ms(self) -> float:
        """Tiempo de espera en milisegundos"""
        if self.started_at is None:
            return 0.0
        return (self.started_at - self.submitted_at) * 1000

    @property
    def execution_time_ms(self) -> float:
        """Tiempo de ejecución en milisegundos"""
        if self.completed_at is None or self.started_at is None:
            return 0.0
        return (self.completed_at - self.started_at) * 1000


class OptimizedThreadPool:
    """
    Thread pool optimizado con priorización y monitoreo

    Características:
    - Priorización de tareas (HIGH, NORMAL, LOW)
    - Monitoreo de tiempo de espera y ejecución
    - Auto-scaling (opcional)
    - Manejo de errores con callbacks
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 100,
        worker_name_prefix: str = "Worker",
        enable_auto_scale: bool = False,
        min_workers: int = 2,
        max_workers: int = 8,
        idle_timeout: float = 30.0,
        logger: Optional[logging.Logger] = None
    ):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.worker_name_prefix = worker_name_prefix
        self.enable_auto_scale = enable_auto_scale
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.idle_timeout = idle_timeout
        self.logger = logger or logging.getLogger(__name__)

        self._queues: Dict[TaskPriority, queue.Queue] = {
            TaskPriority.HIGH: queue.Queue(maxsize=max_queue_size),
            TaskPriority.NORMAL: queue.Queue(maxsize=max_queue_size),
            TaskPriority.LOW: queue.Queue(maxsize=max_queue_size),
        }

        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._task_counter = 0
        self._total_tasks_completed = 0
        self._active_tasks = 0
        self._task_history: List[Task] = []
        self._max_history = 1000

        self._avg_wait_time_ms = 0.0
        self._avg_execution_time_ms = 0.0

        self._start_workers()

    def _start_workers(self):
        """Inicia los workers del pool"""
        for i in range(self.num_workers):
            self._add_worker()

    def _add_worker(self):
        """Añade un nuevo worker al pool"""
        worker_id = len(self._workers)
        thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.worker_name_prefix}-{worker_id}",
            daemon=True
        )
        self._workers.append(thread)
        thread.start()
        self.logger.debug(f"Worker {worker_id} iniciado")

    def _worker_loop(self):
        """Bucle principal del worker"""
        idle_start = None

        while not self._stop_event.is_set():
            try:
                task = self._get_task()

                if task is None:
                    if self.enable_auto_scale and len(self._workers) > self.min_workers:
                        if idle_start is None:
                            idle_start = time.time()
                        elif time.time() - idle_start > self.idle_timeout:
                            self.logger.debug(f"Worker {threading.current_thread().name} terminado por idle")
                            break
                    time.sleep(0.001)
                    continue

                idle_start = None

                self._execute_task(task)

            except Exception as e:
                self.logger.error(f"Error en worker: {e}", exc_info=True)

        with self._lock:
            if threading.current_thread() in self._workers:
                self._workers.remove(threading.current_thread())

    def _get_task(self) -> Optional[Task]:
        """Obtiene la siguiente tarea de la cola de mayor prioridad"""
        try:
            return self._queues[TaskPriority.HIGH].get_nowait()
        except queue.Empty:
            pass

        try:
            return self._queues[TaskPriority.NORMAL].get_nowait()
        except queue.Empty:
            pass

        try:
            return self._queues[TaskPriority.LOW].get_nowait()
        except queue.Empty:
            pass

        return None

    def _execute_task(self, task: Task):
        """Ejecuta una tarea con monitoreo"""
        try:
            task.started_at = time.time()

            with self._lock:
                self._active_tasks += 1

            result = task.func(*task.args, **task.kwargs)
            task.result = result

            task.completed_at = time.time()
            self._total_tasks_completed += 1

            self._update_averages(task)

            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    self.logger.error(f"Error en callback de tarea {task.id}: {e}")

        except Exception as e:
            task.error = e
            task.completed_at = time.time()
            self.logger.error(f"Error ejecutando tarea {task.id}: {e}", exc_info=True)

            if task.error_callback:
                try:
                    task.error_callback(e)
                except Exception as cb_error:
                    self.logger.error(f"Error en error_callback de tarea {task.id}: {cb_error}")

        finally:
            with self._lock:
                self._active_tasks -= 1

            if len(self._task_history) >= self._max_history:
                self._task_history.pop(0)
            self._task_history.append(task)

    def _update_averages(self, task: Task):
        """Actualiza métricas promedio"""
        alpha = 0.1

        if task.wait_time_ms > 0:
            self._avg_wait_time_ms = (
                alpha * task.wait_time_ms +
                (1 - alpha) * self._avg_wait_time_ms
            )

        if task.execution_time_ms > 0:
            self._avg_execution_time_ms = (
                alpha * task.execution_time_ms +
                (1 - alpha) * self._avg_execution_time_ms
            )

    def submit(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        **kwargs
    ) -> Task:
        """
        Envía una tarea al pool

        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            priority: Prioridad de la tarea
            callback: Callback al completar (recibe el resultado)
            error_callback: Callback en caso de error (recibe la excepción)
            **kwargs: Argumentos nombrados

        Returns:
            Task: Objeto que representa la tarea enviada
        """
        if self._stop_event.is_set():
            raise RuntimeError("Thread pool está detenido")

        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1

            task = Task(
                id=task_id,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                callback=callback,
                error_callback=error_callback,
                submitted_at=time.time()
            )

            try:
                self._queues[priority].put(task, timeout=0.1)
            except queue.Full:
                self.logger.warning(f"Cola de prioridad {priority.name} llena, descartando tarea {task_id}")
                return None

            if self.enable_auto_scale and len(self._workers) < self.max_workers:
                total_queue_size = sum(q.qsize() for q in self._queues.values())
                if total_queue_size > self.num_workers * 2:
                    self._add_worker()

            return task

    def submit_batch(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> List[Task]:
        """Envía un lote de tareas"""
        results = []
        for func, args, kwargs in tasks:
            task = self.submit(func, *args, priority=priority, **kwargs)
            if task is not None:
                results.append(task)
        return results

    def wait_all(self, timeout: float = None):
        """Espera a que todas las tareas se completen"""
        start = time.time()
        while True:
            with self._lock:
                total_queue_size = sum(q.qsize() for q in self._queues.values())
                if total_queue_size == 0 and self._active_tasks == 0:
                    break

            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError("Timeout esperando tareas")

            time.sleep(0.01)

    def stop(self, wait: bool = True, timeout: float = 30.0):
        """Detiene el pool"""
        self.logger.info("Deteniendo thread pool...")
        self._stop_event.set()

        if wait:
            start = time.time()
            for worker in self._workers:
                remaining = timeout - (time.time() - start)
                if remaining > 0:
                    worker.join(timeout=remaining)
                else:
                    self.logger.warning("Timeout esperando workers")
                    break

        self.logger.info(f"Thread pool detenido. Tareas completadas: {self._total_tasks_completed}")

    def get_stats(self) -> dict:
        """Obtiene estadísticas del pool"""
        with self._lock:
            total_queue_size = sum(q.qsize() for q in self._queues.values())

            return {
                "num_workers": len(self._workers),
                "active_tasks": self._active_tasks,
                "queue_size": total_queue_size,
                "total_tasks_completed": self._total_tasks_completed,
                "avg_wait_time_ms": self._avg_wait_time_ms,
                "avg_execution_time_ms": self._avg_execution_time_ms,
                "queues": {
                    priority.name: q.qsize()
                    for priority, q in self._queues.items()
                },
                "is_running": not self._stop_event.is_set(),
                "auto_scaling_enabled": self.enable_auto_scale,
            }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(wait=True)
