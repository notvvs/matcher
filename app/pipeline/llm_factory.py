from pathlib import Path
from typing import Dict, Optional

from app.core.settings import settings
from app.core.config_loader import ConfigLoader
from app.core.stopwords import StopwordsManager
from app.core.synonyms import SynonymsManager

from app.services.extraction.text_cleaner import TextCleaner
from app.services.extraction.term_weighter import TermWeighter
from app.services.extraction.extractor import TermExtractor

from app.services.search.query_builder import ElasticsearchQueryBuilder
from app.services.search.elasticsearch_client import ElasticsearchClient

from app.services.filtering.llm_filter import LLMFilter, LLMConfig
from app.services.filtering.llm_score_combiner import LLMScoreCombiner

from app.pipeline.llm_tender_pipeline import LLMTenderProcessingPipeline


def create_llm_pipeline(llm_config: Optional[LLMConfig] = None) -> LLMTenderProcessingPipeline:
    """Создает и конфигурирует LLM пайплайн обработки тендеров"""

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

    # Создаем компоненты LLM фильтрации
    if llm_config is None:
        llm_config = LLMConfig(
            model_name=settings.LLM_MODEL,
            api_url=settings.LLM_API_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            max_workers=settings.LLM_MAX_WORKERS,
            batch_size=settings.LLM_BATCH_SIZE  # Добавлено
        )

    llm_filter = LLMFilter(llm_config)

    score_combiner = LLMScoreCombiner(
        es_weight=settings.ES_SCORE_WEIGHT_LLM,
        llm_weight=settings.LLM_SCORE_WEIGHT
    )

    # Создаем пайплайн
    pipeline = LLMTenderProcessingPipeline(
        term_extractor=term_extractor,
        es_client=es_client,
        llm_filter=llm_filter,
        score_combiner=score_combiner
    )

    return pipeline


def create_hybrid_pipeline() -> LLMTenderProcessingPipeline:
    """Создает гибридный пайплайн с настраиваемой LLM моделью"""

    # Можно выбрать разные модели
    model_name = getattr(settings, 'LLM_MODEL', 'mistral:7b')

    # Конфигурация LLM
    llm_config = LLMConfig(
        model_name=model_name,
        temperature=0.1,  # Низкая температура для детерминированности
        max_tokens=200,
        timeout=30,
        max_workers=4  # Параллельная обработка
    )

    return create_llm_pipeline(llm_config)