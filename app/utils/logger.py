import logging
import sys
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """Настройка базового логгера"""

    # Создаем директорию для логов
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Убираем дублирование
    if logger.handlers:
        return logger

    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Консольный вывод
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файл для всех логов
    file_handler = logging.FileHandler(
        log_dir / 'matcher.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Файл для ошибок
    error_handler = logging.FileHandler(
        log_dir / 'errors.log',
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger