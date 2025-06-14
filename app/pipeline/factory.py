from pathlib import Path

from typing import Dict

from app.core.settings import settings
from app.core.config_loader import ConfigLoader
from app.core.stopwords import StopwordsManager
from app.core.synonyms import SynonymsManager

from app.services.extraction.text_cleaner import TextCleaner
from app.services.extraction.term_weighter import TermWeighter
from app.services.extraction.extractor import TermExtractor

from app.services.search.query_builder import ElasticsearchQueryBuilder
from app.services.search.elasticsearch_client import ElasticsearchClient

from app.services.filtering.semantic_filter import SemanticFilter
from app.services.filtering.score_combiner import ScoreCombiner

from app.pipeline.tender_pipeline import TenderProcessingPipeline


def create_pipeline() -> TenderProcessingPipeline:
    """Создает и конфигурирует пайплайн обработки тендеров"""

    # Загружаем конфигурационные данные
    config_loader = ConfigLoader(settings.DATA_DIR)

    # Создаем менеджеры
    stopwords_manager = StopwordsManager.from_file(
        settings.DATA_DIR,
        'stopwords.txt'
    )

    synonyms_manager = SynonymsManager.from_file(
        settings.DATA_DIR,
        'synonyms.txt'
    )

    # Загружаем важные характеристики
    important_chars = config_loader.load_set('important_chars.txt')

    # Создаем компоненты извлечения
    text_cleaner = TextCleaner(stopwords_manager)
    term_weighter = TermWeighter(synonyms_manager)

    term_extractor = TermExtractor(
        text_cleaner=text_cleaner,
        term_weighter=term_weighter,
        synonyms_manager=synonyms_manager,
        important_chars=important_chars
    )

    # Создаем компоненты поиска
    query_builder = ElasticsearchQueryBuilder()
    es_client = ElasticsearchClient(query_builder)

    # Создаем компоненты фильтрации
    semantic_filter = SemanticFilter()
    score_combiner = ScoreCombiner()

    # Создаем пайплайн
    pipeline = TenderProcessingPipeline(
        term_extractor=term_extractor,
        es_client=es_client,
        semantic_filter=semantic_filter,
        score_combiner=score_combiner
    )

    return pipeline


def create_pipeline_with_config(config: Dict) -> TenderProcessingPipeline:
    """
    Создает пайплайн с пользовательской конфигурацией

    Args:
        config: Словарь с настройками
    """

    # Применяем пользовательские настройки
    if 'data_dir' in config:
        data_dir = Path(config['data_dir'])
    else:
        data_dir = settings.DATA_DIR

    # Загружаем конфигурационные данные
    config_loader = ConfigLoader(data_dir)

    # Создаем менеджеры с возможностью переопределения файлов
    stopwords_file = config.get('stopwords_file', 'stopwords.txt')
    stopwords_manager = StopwordsManager.from_file(data_dir, stopwords_file)

    synonyms_file = config.get('synonyms_file', 'synonyms.txt')
    synonyms_manager = SynonymsManager.from_file(data_dir, synonyms_file)

    important_chars_file = config.get('important_chars_file', 'important_chars.txt')
    important_chars = config_loader.load_set(important_chars_file)

    # Создаем компоненты
    text_cleaner = TextCleaner(stopwords_manager)
    term_weighter = TermWeighter(synonyms_manager)

    term_extractor = TermExtractor(
        text_cleaner=text_cleaner,
        term_weighter=term_weighter,
        synonyms_manager=synonyms_manager,
        important_chars=important_chars
    )

    # Компоненты поиска
    field_multipliers = config.get('es_field_multipliers')
    query_builder = ElasticsearchQueryBuilder(field_multipliers)
    es_client = ElasticsearchClient(query_builder)

    # Компоненты фильтрации
    model_name = config.get('embeddings_model')
    semantic_filter = SemanticFilter(model_name)

    es_weight = config.get('es_score_weight')
    semantic_weight = config.get('semantic_score_weight')
    score_combiner = ScoreCombiner(es_weight, semantic_weight)

    # Создаем пайплайн
    pipeline = TenderProcessingPipeline(
        term_extractor=term_extractor,
        es_client=es_client,
        semantic_filter=semantic_filter,
        score_combiner=score_combiner
    )

    return pipeline