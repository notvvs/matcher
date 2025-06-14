import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class Settings:
    """Основные настройки приложения"""

    # Пути
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'

    # Elasticsearch
    ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    ELASTICSEARCH_PORT = int(os.getenv('ELASTICSEARCH_PORT', 9200))
    ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'products')
    ELASTICSEARCH_TIMEOUT = int(os.getenv('ELASTICSEARCH_TIMEOUT', 30))

    # Поиск
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', 2000))
    MAX_FINAL_RESULTS = int(os.getenv('MAX_FINAL_RESULTS', 20))
    MIN_WEIGHT_THRESHOLD = float(os.getenv('MIN_WEIGHT_THRESHOLD', 1.0))

    # Семантический поиск
    EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'intfloat/multilingual-e5-base')
    SEMANTIC_THRESHOLD = float(os.getenv('SEMANTIC_THRESHOLD', 0.35))
    SEMANTIC_MAX_CANDIDATES = int(os.getenv('SEMANTIC_MAX_CANDIDATES', 500))
    SEMANTIC_BATCH_SIZE = int(os.getenv('SEMANTIC_BATCH_SIZE', 64))

    # Веса для комбинирования скоров
    ES_SCORE_WEIGHT = float(os.getenv('ES_SCORE_WEIGHT', 0.4))
    SEMANTIC_SCORE_WEIGHT = float(os.getenv('SEMANTIC_SCORE_WEIGHT', 0.6))

    # Логирование
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = BASE_DIR / 'logs'

    # Режим отладки
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

    @classmethod
    def get_elasticsearch_config(cls):
        """Конфигурация для подключения к Elasticsearch"""
        return {
            'hosts': [{
                'host': cls.ELASTICSEARCH_HOST,
                'port': cls.ELASTICSEARCH_PORT,
                'scheme': 'http'
            }],
            'verify_certs': False,
            'ssl_show_warn': False,
            'request_timeout': cls.ELASTICSEARCH_TIMEOUT
        }


# Экземпляр для импорта
settings = Settings()