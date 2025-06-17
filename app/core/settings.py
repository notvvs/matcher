import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class Settings:
    """Основные настройки приложения"""

    # Пути
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = Path(os.getenv('DATA_DIR', str(BASE_DIR / 'data')))
    LOG_DIR = Path(os.getenv('LOG_DIR', 'logs'))

    # Elasticsearch
    ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    ELASTICSEARCH_PORT = int(os.getenv('ELASTICSEARCH_PORT', 9200))
    ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'products')
    ELASTICSEARCH_TIMEOUT = int(os.getenv('ELASTICSEARCH_TIMEOUT', 30))

    # Поиск
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', 50))
    MAX_FINAL_RESULTS = int(os.getenv('MAX_FINAL_RESULTS', 20))
    MIN_WEIGHT_THRESHOLD = float(os.getenv('MIN_WEIGHT_THRESHOLD', 1.0))

    # LLM настройки
    LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2:1b')
    LLM_API_URL = os.getenv('LLM_API_URL', 'http://localhost:11434/api/generate')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.1))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 200))
    LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', 20))
    LLM_MAX_WORKERS = int(os.getenv('LLM_MAX_WORKERS', 8))
    LLM_BATCH_SIZE = int(os.getenv('LLM_BATCH_SIZE', 10))
    LLM_THRESHOLD = float(os.getenv('LLM_THRESHOLD', 0.5))

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

    # API настройки
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))
    API_RELOAD = os.getenv('API_RELOAD', 'True').lower() == 'true'

    # Режим отладки
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

    # Файлы конфигурации
    STOPWORDS_FILE = os.getenv('STOPWORDS_FILE', 'stopwords.txt')
    SYNONYMS_FILE = os.getenv('SYNONYMS_FILE', 'synonyms.txt')
    IMPORTANT_CHARS_FILE = os.getenv('IMPORTANT_CHARS_FILE', 'important_chars.txt')

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