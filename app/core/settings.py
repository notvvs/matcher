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

    # Семантический поиск (оставляем для совместимости)
    EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'intfloat/multilingual-e5-base')
    SEMANTIC_THRESHOLD = float(os.getenv('SEMANTIC_THRESHOLD', 0.35))
    SEMANTIC_MAX_CANDIDATES = int(os.getenv('SEMANTIC_MAX_CANDIDATES', 500))
    SEMANTIC_BATCH_SIZE = int(os.getenv('SEMANTIC_BATCH_SIZE', 64))

    # Веса для комбинирования скоров (семантический подход)
    ES_SCORE_WEIGHT = float(os.getenv('ES_SCORE_WEIGHT', 0.4))
    SEMANTIC_SCORE_WEIGHT = float(os.getenv('SEMANTIC_SCORE_WEIGHT', 0.6))

    # LLM настройки
    LLM_MODEL = os.getenv('LLM_MODEL', 'mistral:7b')
    LLM_API_URL = os.getenv('LLM_API_URL', 'http://localhost:11434/api/generate')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.1))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 200))
    LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', 30))
    LLM_MAX_WORKERS = int(os.getenv('LLM_MAX_WORKERS', 4))
    LLM_THRESHOLD = float(os.getenv('LLM_THRESHOLD', 0.7))

    # Веса для LLM комбинирования
    ES_SCORE_WEIGHT_LLM = float(os.getenv('ES_SCORE_WEIGHT_LLM', 0.3))
    LLM_SCORE_WEIGHT = float(os.getenv('LLM_SCORE_WEIGHT', 0.7))

    # Веса для извлечения терминов
    WEIGHTS = {
        'required_values': {
            'start': 4.0,
            'step': 0.2,
            'count': 5
        },
        'optional_values': {
            'start': 2.5,
            'step': 0.3,
            'count': 3
        },
        'char_names': {
            'start': 1.8,
            'step': 0.2,
            'count': 4
        },
        'synonym_penalty': 0.7,
        'es_field_multipliers': {
            'title': 2.5,
            'category': 1.8,
            'brand': 1.5,
            'attr_value': 2.0,
            'attr_name': 1.0
        }
    }

    # Логирование
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = BASE_DIR / 'logs'

    # Режим отладки
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

    # Файлы конфигурации
    CONFIG_DIR = os.getenv('CONFIG_DIR', 'config')
    STOPWORDS_FILE = os.getenv('STOPWORDS_FILE', str(DATA_DIR / 'stopwords.txt'))
    SYNONYMS_FILE = os.getenv('SYNONYMS_FILE', str(DATA_DIR / 'synonyms.txt'))
    IMPORTANT_CHARS_FILE = os.getenv('IMPORTANT_CHARS_FILE', str(DATA_DIR / 'important_chars.txt'))

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