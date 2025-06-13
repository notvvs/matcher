"""
Централизованные настройки системы поиска товаров
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения из .env файла
load_dotenv()


class Settings:
    """Основные настройки"""

    # Elasticsearch
    ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    ELASTICSEARCH_PORT = int(os.getenv('ELASTICSEARCH_PORT', 9200))
    ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'ipointer_index')
    ELASTICSEARCH_TIMEOUT = int(os.getenv('ELASTICSEARCH_TIMEOUT', 30))

    # MongoDB (для будущего использования)
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'products_db')
    MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'products')

    # Пути к файлам конфигурации
    CONFIG_DIR = Path(os.getenv('CONFIG_DIR', 'config'))
    STOPWORDS_FILE = os.getenv('STOPWORDS_FILE', 'stopwords.txt')
    SYNONYMS_FILE = os.getenv('SYNONYMS_FILE', 'synonyms.txt')
    IMPORTANT_CHARS_FILE = os.getenv('IMPORTANT_CHARS_FILE', 'important_chars.txt')

    # Настройки поиска
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', 2000))
    MAX_FINAL_RESULTS = int(os.getenv('MAX_FINAL_RESULTS', 20))
    MIN_WEIGHT_THRESHOLD = float(os.getenv('MIN_WEIGHT_THRESHOLD', 1.0))

    # Настройки весов
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
        'synonym_penalty': 0.7,  # Множитель для синонимов (30% штраф)

        # Множители для полей Elasticsearch
        'es_field_multipliers': {
            'title': 2.5,
            'category': 1.8,
            'brand': 1.5,
            'attr_value': 2.0,
            'attr_name': 1.0
        }
    }

    # Семантический поиск
    EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'intfloat/multilingual-e5-base')
    SEMANTIC_THRESHOLD = float(os.getenv('SEMANTIC_THRESHOLD', 0.35))  # Повышено с 0.3 до 0.35
    SEMANTIC_MAX_CANDIDATES = int(os.getenv('SEMANTIC_MAX_CANDIDATES', 500))
    SEMANTIC_BATCH_SIZE = int(os.getenv('SEMANTIC_BATCH_SIZE', 64))

    # Веса для комбинирования скоров
    ES_SCORE_WEIGHT = float(os.getenv('ES_SCORE_WEIGHT', 0.4))
    SEMANTIC_SCORE_WEIGHT = float(os.getenv('SEMANTIC_SCORE_WEIGHT', 0.6))

    # Логирование
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'matcher.log')

    # Режим отладки
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

    @classmethod
    def get_config_path(cls, filename):
        """Получить полный путь к конфигурационному файлу"""
        return cls.CONFIG_DIR / filename

    @classmethod
    def get_elasticsearch_config(cls):
        """Получить конфигурацию для Elasticsearch"""
        return {
            'hosts': [{'host': cls.ELASTICSEARCH_HOST, 'port': cls.ELASTICSEARCH_PORT, 'scheme': 'http'}],
            'verify_certs': False,
            'ssl_show_warn': False,
            'request_timeout': cls.ELASTICSEARCH_TIMEOUT
        }

    @classmethod
    def get_mongodb_config(cls):
        """Получить конфигурацию для MongoDB"""
        return {
            'uri': cls.MONGODB_URI,
            'database': cls.MONGODB_DATABASE,
            'collection': cls.MONGODB_COLLECTION
        }

    @classmethod
    def print_config(cls):
        """Вывести текущую конфигурацию"""
        print("📋 Текущая конфигурация:")
        print(f"   Elasticsearch: {cls.ELASTICSEARCH_HOST}:{cls.ELASTICSEARCH_PORT}")
        print(f"   Индекс: {cls.ELASTICSEARCH_INDEX}")
        print(f"   MongoDB: {cls.MONGODB_URI}")
        print(f"   Конфиг директория: {cls.CONFIG_DIR}")
        print(f"   Режим отладки: {cls.DEBUG_MODE}")
        print(f"   Макс. результатов поиска: {cls.MAX_SEARCH_RESULTS}")
        print(f"   Макс. финальных результатов: {cls.MAX_FINAL_RESULTS}")


# Для удобства импорта
settings = Settings()