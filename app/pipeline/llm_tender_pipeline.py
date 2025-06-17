import time
import logging
from typing import Dict, List

from app.core.settings import settings
from app.services.extraction.extractor import TermExtractor
from app.services.search.elasticsearch_client import ElasticsearchClient
from app.services.filtering.llm_filter import LLMFilter
from app.services.filtering.llm_score_combiner import LLMScoreCombiner


class LLMTenderProcessingPipeline:
    """Пайплайн для обработки тендеров с использованием LLM"""

    def __init__(
            self,
            term_extractor: TermExtractor,
            es_client: ElasticsearchClient,
            llm_filter: LLMFilter,
            score_combiner: LLMScoreCombiner
    ):
        self.term_extractor = term_extractor
        self.es_client = es_client
        self.llm_filter = llm_filter
        self.score_combiner = score_combiner
        self.logger = logging.getLogger(__name__)

    def process(self, tender: Dict) -> Dict:
        """Обрабатывает тендер через все этапы с LLM анализом"""

        start_time = time.time()
        tender_name = tender.get('name', 'Без названия')

        self.logger.info(f"Обработка тендера: {tender_name}")

        # Инициализируем результат
        result = {
            'tender': tender,
            'stages': {},
            'final_products': [],
            'statistics': {},
            'execution_time': 0
        }

        try:
            # Этап 1: Извлечение терминов
            stage1_result = self._execute_extraction(tender)
            result['stages']['extraction'] = stage1_result

            if 'error' in stage1_result:
                result['error'] = stage1_result['error']
                return result

            # Этап 2: Поиск в Elasticsearch
            stage2_result = self._execute_search(stage1_result['search_terms'])
            result['stages']['elasticsearch'] = stage2_result

            if 'error' in stage2_result:
                result['error'] = stage2_result['error']
                return result

            candidates = stage2_result['candidates']

            if not candidates:
                self.logger.warning("Поиск не дал результатов")
                result['statistics'] = {
                    'total_products_found': 0,
                    'total_time': time.time() - start_time
                }
                return result

            # Этап 3: LLM анализ и фильтрация
            stage3_result = self._execute_llm_filtering(tender, candidates)
            result['stages']['llm_analysis'] = stage3_result

            if 'error' in stage3_result:
                result['error'] = stage3_result['error']
                return result

            # Финальные результаты
            final_products = stage3_result['filtered_products'][:settings.MAX_FINAL_RESULTS]
            result['final_products'] = final_products

            # Общая статистика
            total_time = time.time() - start_time
            result['execution_time'] = total_time
            result['statistics'] = self._calculate_statistics(result, total_time)

            self.logger.info(
                f"Обработка завершена: найдено {len(final_products)} товаров "
                f"за {total_time:.2f}с"
            )

        except Exception as e:
            self.logger.error(f"Ошибка в пайплайне: {e}", exc_info=True)
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time

        return result

    def _execute_extraction(self, tender: Dict) -> Dict:
        """Этап 1: Извлечение терминов"""
        start_time = time.time()

        try:
            search_terms = self.term_extractor.extract_from_tender(tender)

            return {
                'search_terms': search_terms,
                'search_query': search_terms['search_query'],
                'boost_terms_count': len(search_terms['boost_terms']),
                'execution_time': time.time() - start_time
            }

        except Exception as e:
            self.logger.error(f"Ошибка извлечения терминов: {e}")
            return {'error': str(e), 'execution_time': time.time() - start_time}

    def _execute_search(self, search_terms: Dict) -> Dict:
        """Этап 2: Поиск в Elasticsearch"""
        start_time = time.time()

        try:
            max_candidates = min(settings.MAX_SEARCH_RESULTS, 100)
            es_results = self.es_client.search_products(
                search_terms,
                size=max_candidates
            )

            if 'error' in es_results:
                return {
                    'error': es_results['error'],
                    'execution_time': time.time() - start_time
                }

            candidates = es_results.get('candidates', [])

            return {
                'candidates': candidates,
                'total_found': es_results['total_found'],
                'candidates_retrieved': len(candidates),
                'max_score': es_results.get('max_score', 0),
                'execution_time': time.time() - start_time
            }

        except Exception as e:
            self.logger.error(f"Ошибка поиска: {e}")
            return {'error': str(e), 'execution_time': time.time() - start_time}

    def _execute_llm_filtering(self, tender: Dict, candidates: List[Dict]) -> Dict:
        """Этап 3: LLM анализ и фильтрация"""
        start_time = time.time()

        try:
            threshold = settings.LLM_THRESHOLD

            filtered = self.llm_filter.filter_by_llm_analysis(
                tender,
                candidates,
                threshold=threshold,
                top_k=50
            )

            if not filtered:
                return {
                    'filtered_products': [],
                    'input_products': len(candidates),
                    'filtered_products_count': 0,
                    'threshold_used': threshold,
                    'execution_time': time.time() - start_time
                }

            # Комбинирование скоров
            filtered = self.score_combiner.combine_scores(filtered)

            return {
                'filtered_products': filtered,
                'input_products': len(candidates),
                'filtered_products_count': len(filtered),
                'threshold_used': threshold,
                'execution_time': time.time() - start_time
            }

        except Exception as e:
            self.logger.error(f"Ошибка LLM фильтрации: {e}")
            return {'error': str(e), 'execution_time': time.time() - start_time}

    def _calculate_statistics(self, result: Dict, total_time: float) -> Dict:
        """Рассчитывает общую статистику"""
        stages = result.get('stages', {})

        stats = {
            'total_products_found': len(result.get('final_products', [])),
            'stages_timing': {
                'extraction': stages.get('extraction', {}).get('execution_time', 0),
                'elasticsearch': stages.get('elasticsearch', {}).get('execution_time', 0),
                'llm_analysis': stages.get('llm_analysis', {}).get('execution_time', 0)
            },
            'total_time': total_time,
            'products_per_stage': {
                'after_es': stages.get('elasticsearch', {}).get('candidates_retrieved', 0),
                'after_llm': stages.get('llm_analysis', {}).get('filtered_products_count', 0),
                'final': len(result.get('final_products', []))
            }
        }

        return stats