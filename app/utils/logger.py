import logging
import sys
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """Настройка базового логгера"""

    logger = logging.getLogger(name)

    # Избегаем дублирования обработчиков
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

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

    # Файловый вывод (опционально)
    log_dir = Path('logs')
    if log_dir.exists():
        file_handler = logging.FileHandler(
            log_dir / 'app.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger