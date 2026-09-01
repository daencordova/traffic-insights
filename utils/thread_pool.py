"""Optimized thread pool for parallel frame processing.

This module provides a thread pool with task prioritization,
monitoring, and auto-scaling capabilities.

Features:
    - Task prioritization (HIGH, NORMAL, LOW)
    - Wait time and execution time monitoring
    - Auto-scaling (optional)
    - Error handling with callbacks
    - Batch task submission
    - Statistics and monitoring

Example:
    >>> from utils.thread_pool import OptimizedThreadPool, ThreadPoolConfig, TaskPriority
    >>>
    >>> # Create configuration
    >>> config = ThreadPoolConfig(num_workers=4, max_queue_size=100, enable_auto_scale=True)
    >>>
    >>> # Create thread pool
    >>> pool = OptimizedThreadPool(config)
    >>>
    >>> # Submit a task
    >>> def process_frame(frame):
    ...     return detector.detect(frame)
    >>>
    >>> task = pool.submit(
    ...     process_frame,
    ...     frame,
    ...     priority=TaskPriority.HIGH,
    ...     callback=lambda result: print(f"Detected {len(result)} objects"),
    ... )
    >>>
    >>> # Submit batch of tasks
    >>> tasks = [(detect_frame, (f,), {}) for f in frames]
    >>> results = pool.submit_batch(tasks, priority=TaskPriority.NORMAL)
    >>>
    >>> # Wait for all tasks
    >>> pool.wait_all(timeout=5.0)
    >>>
    >>> # Get statistics
    >>> stats = pool.get_stats()
    >>> print(f"Active workers: {stats['num_workers']}")
    >>> print(f"Queue size: {stats['queue_size']}")
    >>>
    >>> # Use as context manager
    >>> with OptimizedThreadPool() as pool:
    ...     pool.submit(process, data)
    >>> # Pool is automatically stopped
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import queue
import threading
import time
from typing import Any


class TaskPriority(Enum):
    """Task priorities for scheduling.

    Attributes:
        HIGH: High priority tasks are executed first.
        NORMAL: Normal priority tasks (default).
        LOW: Low priority tasks are executed last.

    Example:
        >>> priority = TaskPriority.HIGH
        >>> task = pool.submit(func, args, priority=priority)
    """

    HIGH = auto()
    NORMAL = auto()
    LOW = auto()


@dataclass(slots=True)
class Task:
    """Represents a task to be executed in the pool.

    This class stores all information about a submitted task,
    including execution metrics and callbacks.

    Attributes:
        id: Unique task identifier.
        func: Function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        priority: Task priority level.
        callback: Callback on completion (receives result).
        error_callback: Callback on error (receives exception).
        submitted_at: Submission timestamp.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        result: Task result.
        error: Task error (if any).

    Example:
        >>> task = Task(id=42, func=process_frame, args=(frame,), priority=TaskPriority.HIGH)
        >>> print(f"Wait time: {task.wait_time_ms:.2f}ms")
    """

    id: int
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    callback: Callable | None = None
    error_callback: Callable | None = None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: Exception | None = None

    @property
    def wait_time_ms(self) -> float:
        """Wait time in milliseconds before execution."""
        if self.started_at is None:
            return 0.0
        return (self.started_at - self.submitted_at) * 1000

    @property
    def execution_time_ms(self) -> float:
        """Execution time in milliseconds."""
        if self.completed_at is None or self.started_at is None:
            return 0.0
        return (self.completed_at - self.started_at) * 1000


@dataclass
class ThreadPoolConfig:
    """Configuration for the thread pool.

    Attributes:
        num_workers: Initial number of workers.
        max_queue_size: Maximum queue size per priority.
        worker_name_prefix: Prefix for worker thread names.
        enable_auto_scale: Whether to enable auto-scaling.
        min_workers: Minimum number of workers (auto-scaling).
        max_workers: Maximum number of workers (auto-scaling).
        idle_timeout: Idle timeout for workers in seconds.
        logger: Logger for the pool.

    Example:
        >>> config = ThreadPoolConfig(
        ...     num_workers=8,
        ...     max_queue_size=200,
        ...     enable_auto_scale=True,
        ...     min_workers=4,
        ...     max_workers=16,
        ...     idle_timeout=60.0,
        ... )
    """

    num_workers: int = 4
    max_queue_size: int = 100
    worker_name_prefix: str = "Worker"
    enable_auto_scale: bool = False
    min_workers: int = 2
    max_workers: int = 8
    idle_timeout: float = 30.0
    logger: logging.Logger | None = None


class OptimizedThreadPool:
    """Optimized thread pool with prioritization and monitoring.

    This thread pool supports task prioritization, auto-scaling,
    and comprehensive monitoring.

    Features:
        - Task prioritization (HIGH, NORMAL, LOW)
        - Wait time and execution time monitoring
        - Auto-scaling (optional)
        - Error handling with callbacks
        - Batch task submission
        - Statistics and monitoring

    Example:
        >>> pool = OptimizedThreadPool()
        >>>
        >>> # Submit a task with callback
        >>> pool.submit(
        ...     process_data,
        ...     data,
        ...     priority=TaskPriority.HIGH,
        ...     callback=lambda r: print(f"Result: {r}"),
        ...     error_callback=lambda e: print(f"Error: {e}"),
        ... )
        >>>
        >>> # Check stats
        >>> stats = pool.get_stats()
        >>> print(f"Avg wait: {stats['avg_wait_time_ms']:.2f}ms")
    """

    def __init__(self, config: ThreadPoolConfig | None = None):
        """Initializes the thread pool.

        Args:
            config: Pool configuration. If None, uses default values.

        Example:
            >>> # Default configuration
            >>> pool = OptimizedThreadPool()
            >>>
            >>> # Custom configuration
            >>> config = ThreadPoolConfig(num_workers=8, enable_auto_scale=True)
            >>> pool = OptimizedThreadPool(config)
        """
        if config is None:
            config = ThreadPoolConfig()

        self.config = config
        self.num_workers = config.num_workers
        self.max_queue_size = config.max_queue_size
        self.worker_name_prefix = config.worker_name_prefix
        self.enable_auto_scale = config.enable_auto_scale
        self.min_workers = config.min_workers
        self.max_workers = config.max_workers
        self.idle_timeout = config.idle_timeout
        self.logger = config.logger or logging.getLogger(__name__)

        self._queues: dict[TaskPriority, queue.Queue] = {
            TaskPriority.HIGH: queue.Queue(maxsize=self.max_queue_size),
            TaskPriority.NORMAL: queue.Queue(maxsize=self.max_queue_size),
            TaskPriority.LOW: queue.Queue(maxsize=self.max_queue_size),
        }

        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._task_counter = 0
        self._total_tasks_completed = 0
        self._active_tasks = 0
        self._task_history: list[Task] = []
        self._max_history = 1000

        self._avg_wait_time_ms = 0.0
        self._avg_execution_time_ms = 0.0

        self._start_workers()

    def _start_workers(self) -> None:
        """Starts the pool workers."""
        for _ in range(self.num_workers):
            self._add_worker()

    def _add_worker(self) -> None:
        """Adds a new worker to the pool."""
        worker_id = len(self._workers)
        thread = threading.Thread(
            target=self._worker_loop, name=f"{self.worker_name_prefix}-{worker_id}", daemon=True
        )
        self._workers.append(thread)
        thread.start()
        self.logger.debug(f"Worker {worker_id} started")

    def _worker_loop(self) -> None:
        """Main worker loop."""
        idle_start = None

        while not self._stop_event.is_set():
            try:
                task = self._get_task()

                if task is None:
                    if self.enable_auto_scale and len(self._workers) > self.min_workers:
                        if idle_start is None:
                            idle_start = time.time()
                        elif time.time() - idle_start > self.idle_timeout:
                            self.logger.debug(
                                f"Worker {threading.current_thread().name} terminated due to idle"
                            )
                            break
                    time.sleep(0.001)
                    continue

                idle_start = None

                self._execute_task(task)

            except Exception as e:
                self.logger.error(f"Worker error: {e}", exc_info=True)

        with self._lock:
            if threading.current_thread() in self._workers:
                self._workers.remove(threading.current_thread())

    def _get_task(self) -> Task | None:
        """Gets the next task from the highest priority queue."""
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

    def _execute_task(self, task: Task) -> None:
        """Executes a task with monitoring."""
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
                    self.logger.error(f"Error in callback for task {task.id}: {e}")

        except Exception as e:
            task.error = e
            task.completed_at = time.time()
            self.logger.error(f"Error executing task {task.id}: {e}", exc_info=True)

            if task.error_callback:
                try:
                    task.error_callback(e)
                except Exception as cb_error:
                    self.logger.error(f"Error in error_callback for task {task.id}: {cb_error}")

        finally:
            with self._lock:
                self._active_tasks -= 1

            if len(self._task_history) >= self._max_history:
                self._task_history.pop(0)
            self._task_history.append(task)

    def _update_averages(self, task: Task) -> None:
        """Updates average metrics."""
        alpha = 0.1

        if task.wait_time_ms > 0:
            self._avg_wait_time_ms = (
                alpha * task.wait_time_ms + (1 - alpha) * self._avg_wait_time_ms
            )

        if task.execution_time_ms > 0:
            self._avg_execution_time_ms = (
                alpha * task.execution_time_ms + (1 - alpha) * self._avg_execution_time_ms
            )

    def submit(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Callable | None = None,
        error_callback: Callable | None = None,
        **kwargs,
    ) -> Task | None:
        """Submits a task to the pool.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            priority: Task priority.
            callback: Callback on completion (receives result).
            error_callback: Callback on error (receives exception).
            **kwargs: Keyword arguments.

        Returns:
            Task: Submitted task object, or None if queue is full.

        Raises:
            RuntimeError: If the thread pool is stopped.

        Example:
            >>> def process(data):
            ...     return data * 2
            >>>
            >>> task = pool.submit(
            ...     process,
            ...     42,
            ...     priority=TaskPriority.HIGH,
            ...     callback=lambda r: print(f"Result: {r}"),
            ... )
        """
        if self._stop_event.is_set():
            raise RuntimeError("Thread pool is stopped")

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
                submitted_at=time.time(),
            )

            try:
                self._queues[priority].put(task, timeout=0.1)
            except queue.Full:
                self.logger.warning(
                    f"Queue for priority {priority.name} is full, discarding task {task_id}"
                )
                return None

            if self.enable_auto_scale and len(self._workers) < self.max_workers:
                total_queue_size = sum(q.qsize() for q in self._queues.values())
                if total_queue_size > self.num_workers * 2:
                    self._add_worker()

            return task

    def submit_batch(
        self,
        tasks: list[tuple[Callable, tuple, dict]],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> list[Task]:
        """Submits a batch of tasks.

        Args:
            tasks: List of tasks, each as (func, args, kwargs) tuple.
            priority: Priority for all tasks.

        Returns:
            List[Task]: List of successfully submitted tasks.

        Example:
            >>> tasks = [(process, (d,), {}) for d in data]
            >>> results = pool.submit_batch(tasks, priority=TaskPriority.NORMAL)
        """
        results = []
        for func, args, kwargs in tasks:
            task = self.submit(func, *args, priority=priority, **kwargs)
            if task is not None:
                results.append(task)
        return results

    def wait_all(self, timeout: float | None = None) -> None:
        """Waits for all tasks to complete.

        Args:
            timeout: Timeout in seconds (optional).

        Raises:
            TimeoutError: If timeout expires.

        Example:
            >>> pool.wait_all(timeout=5.0)
            >>> print("All tasks completed")
        """
        start = time.time()
        while True:
            with self._lock:
                total_queue_size = sum(q.qsize() for q in self._queues.values())
                if total_queue_size == 0 and self._active_tasks == 0:
                    break

            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for tasks")

            time.sleep(0.01)

    def stop(self, *, wait: bool = True, timeout: float = 30.0) -> None:
        """Stops the pool.

        Args:
            wait: Whether to wait for tasks to finish.
            timeout: Timeout for waiting in seconds.

        Example:
            >>> pool.stop(wait=True, timeout=10.0)
            >>> print("Pool stopped")
        """
        self.logger.info("Stopping thread pool...")
        self._stop_event.set()

        if wait:
            start = time.time()
            for worker in self._workers:
                remaining = timeout - (time.time() - start)
                if remaining > 0:
                    worker.join(timeout=remaining)
                else:
                    self.logger.warning("Timeout waiting for workers")
                    break

        self.logger.info(f"Thread pool stopped. Completed tasks: {self._total_tasks_completed}")

    def get_stats(self) -> dict:
        """Gets pool statistics.

        Returns:
            dict: Pool statistics including:
                - num_workers: Number of active workers
                - active_tasks: Currently executing tasks
                - queue_size: Total queue size
                - total_tasks_completed: Completed tasks
                - avg_wait_time_ms: Average wait time
                - avg_execution_time_ms: Average execution time
                - queues: Size by priority queue
                - is_running: Whether pool is running
                - auto_scaling_enabled: Whether auto-scaling is enabled

        Example:
            >>> stats = pool.get_stats()
            >>> print(f"Workers: {stats['num_workers']}")
            >>> print(f"Queue: {stats['queue_size']}")
            >>> print(f"Avg wait: {stats['avg_wait_time_ms']:.2f}ms")
        """
        with self._lock:
            total_queue_size = sum(q.qsize() for q in self._queues.values())

            return {
                "num_workers": len(self._workers),
                "active_tasks": self._active_tasks,
                "queue_size": total_queue_size,
                "total_tasks_completed": self._total_tasks_completed,
                "avg_wait_time_ms": self._avg_wait_time_ms,
                "avg_execution_time_ms": self._avg_execution_time_ms,
                "queues": {priority.name: q.qsize() for priority, q in self._queues.items()},
                "is_running": not self._stop_event.is_set(),
                "auto_scaling_enabled": self.enable_auto_scale,
            }

    def __enter__(self):
        """Enters the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the context manager and stops the pool."""
        self.stop(wait=True)
