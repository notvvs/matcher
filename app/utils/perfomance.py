"""
Утилиты для измерения производительности
"""

import time
import functools
import logging
from typing import Callable, Any, Dict
from contextlib import contextmanager


class PerformanceMonitor:
    """Мониторинг производительности"""

    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def measure(self, operation_name: str):
        """Контекстный менеджер для измерения времени операции"""

        start_time = time.time()

        try:
            yield
        finally:
            elapsed = time.time() - start_time

            # Сохраняем метрику
            if operation_name not in self.metrics:
                self.metrics[operation_name] = {
                    'count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0
                }

            metric = self.metrics[operation_name]
            metric['count'] += 1
            metric['total_time'] += elapsed
            metric['min_time'] = min(metric['min_time'], elapsed)
            metric['max_time'] = max(metric['max_time'], elapsed)

            self.logger.debug(
                f"{operation_name}: {elapsed:.3f}с"
            )

    def get_stats(self, operation_name: str = None) -> Dict[str, Any]:
        """Получить статистику по операциям"""

        if operation_name:
            metric = self.metrics.get(operation_name)
            if not metric:
                return {}

            return {
                'operation': operation_name,
                'count': metric['count'],
                'total_time': metric['total_time'],
                'avg_time': metric['total_time'] / metric['count'],
                'min_time': metric['min_time'],
                'max_time': metric['max_time']
            }

        # Возвращаем статистику по всем операциям
        stats = {}

        for name, metric in self.metrics.items():
            stats[name] = {
                'count': metric['count'],
                'total_time': metric['total_time'],
                'avg_time': metric['total_time'] / metric['count'],
                'min_time': metric['min_time'],
                'max_time': metric['max_time']
            }

        return stats

    def reset(self):
        """Сбросить все метрики"""
        self.metrics.clear()

    def print_summary(self):
        """Вывести сводку по производительности"""

        if not self.metrics:
            print("Нет данных о производительности")
            return

        print("\nСводка по производительности:")
        print("-" * 60)

        # Сортируем по общему времени
        sorted_metrics = sorted(
            self.metrics.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )

        for name, metric in sorted_metrics:
            avg_time = metric['total_time'] / metric['count']

            print(f"{name}:")
            print(f"  Вызовов: {metric['count']}")
            print(f"  Общее время: {metric['total_time']:.3f}с")
            print(f"  Среднее время: {avg_time:.3f}с")
            print(f"  Мин/Макс: {metric['min_time']:.3f}с / {metric['max_time']:.3f}с")
            print()


def timeit(func: Callable) -> Callable:
    """Декоратор для измерения времени выполнения функции"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            logger = logging.getLogger(func.__module__)
            logger.debug(
                f"{func.__name__} выполнена за {elapsed:.3f}с"
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger = logging.getLogger(func.__module__)
            logger.error(
                f"{func.__name__} завершилась с ошибкой за {elapsed:.3f}с: {e}"
            )
            raise

    return wrapper


class Timer:
    """Простой таймер для измерения времени"""

    def __init__(self, name: str = None, logger: logging.Logger = None):
        self.name = name or "Timer"
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.elapsed_time = None

    def start(self):
        """Запустить таймер"""
        self.start_time = time.time()
        self.elapsed_time = None

    def stop(self) -> float:
        """Остановить таймер и вернуть прошедшее время"""

        if self.start_time is None:
            raise RuntimeError("Таймер не был запущен")

        self.elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"{self.name}: {self.elapsed_time:.3f}с"
        )

        return self.elapsed_time

    def __enter__(self):
        """Вход в контекстный менеджер"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера"""
        self.stop()


def format_time(seconds: float) -> str:
    """Форматирует время в читаемый вид"""

    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}мкс"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}мс"
    elif seconds < 60:
        return f"{seconds:.2f}с"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}м {secs:.1f}с"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}ч {minutes}м"