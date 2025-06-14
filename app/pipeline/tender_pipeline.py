import time
import logging
from typing import Dict, Any, List, Optional

from app.core.settings import settings
from app.services.extraction.extractor import TermExtractor
from app.services.search.elasticsearch_client import ElasticsearchClient
from app.services.filtering.semantic_filter import SemanticFilter
from app.services.filtering.score_combiner import ScoreCombiner


class TenderProcessingPipeline:
    """Пайплайн для обработки тендеров через все этапы"""

    def __init__(
            self,
            term_extractor: TermExtractor,
            es_client: ElasticsearchClient,
            semantic_filter: SemanticFilter,
            score_combiner: ScoreCombiner
    ):
        self.term_extractor = term_extractor
        self.es_client = es_client
        self.semantic_filter = semantic_filter
        self.score_combiner = score_combiner
        self.logger = logging.getLogger(__name__)

    def process(self, tender: Dict) -> Dict:
        """
        Обрабатывает тендер через все этапы

        Returns:
            Dict с результатами обработки
        """

        start_time = time.time()
        tender_name = tender.get('name', 'Без названия')

        self.logger.info(f"Начало обработки тендера: {tender_name}")

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
                return result

            # Этап 3: Семантическая фильтрация
            stage3_result = self._execute_filtering(tender, candidates)
            result['stages']['semantic'] = stage3_result

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

            # Логируем топ результаты
            self._log_top_results(final_products)

        except Exception as e:
            self.logger.error(f"Критическая ошибка в пайплайне: {e}", exc_info=True)
            result['error'] = str(e)

        return result

    def _execute_extraction(self, tender: Dict) -> Dict:
        """Этап 1: Извлечение терминов"""

        self.logger.info("Этап 1: Извлечение терминов...")
        start_time = time.time()

        try:
            search_terms = self.term_extractor.extract_from_tender(tender)

            execution_time = time.time() - start_time

            self.logger.info(
                f"Извлечено терминов: {len(search_terms['boost_terms'])}, "
                f"время: {execution_time:.2f}с"
            )

            return {
                'search_terms': search_terms,
                'search_query': search_terms['search_query'],
                'boost_terms_count': len(search_terms['boost_terms']),
                'execution_time': execution_time
            }

        except Exception as e:
            self.logger.error(f"Ошибка извлечения терминов: {e}", exc_info=True)
            return {'error': str(e), 'execution_time': time.time() - start_time}

    def _execute_search(self, search_terms: Dict) -> Dict:
        """Этап 2: Поиск в Elasticsearch"""

        self.logger.info("Этап 2: Поиск в Elasticsearch...")
        start_time = time.time()

        try:
            es_results = self.es_client.search_products(
                search_terms,
                size=settings.MAX_SEARCH_RESULTS
            )

            execution_time = time.time() - start_time

            if 'error' in es_results:
                return {
                    'error': es_results['error'],
                    'execution_time': execution_time
                }

            candidates = es_results.get('candidates', [])

            self.logger.info(
                f"Найдено: {len(candidates)} из {es_results['total_found']}, "
                f"время: {execution_time:.2f}с"
            )

            return {
                'candidates': candidates,
                'total_found': es_results['total_found'],
                'candidates_retrieved': len(candidates),
                'max_score': es_results.get('max_score', 0),
                'execution_time': execution_time
            }

        except Exception as e:
            self.logger.error(f"Ошибка поиска: {e}", exc_info=True)
            return {'error': str(e), 'execution_time': time.time() - start_time}

    def _execute_filtering(self, tender: Dict, candidates: List[Dict]) -> Dict:
        """Этап 3: Семантическая фильтрация и комбинирование скоров"""

        self.logger.info("Этап 3: Семантическая фильтрация...")
        start_time = time.time()

        try:
            # Семантическая фильтрация
            filtered = self.semantic_filter.filter_by_similarity(
                tender,
                candidates,
                threshold=settings.SEMANTIC_THRESHOLD,
                top_k=settings.SEMANTIC_MAX_CANDIDATES
            )

            # Комбинирование скоров
            filtered = self.score_combiner.combine_scores(filtered)

            execution_time = time.time() - start_time

            self.logger.info(
                f"Отфильтровано: {len(filtered)} из {len(candidates)}, "
                f"время: {execution_time:.2f}с"
            )

            return {
                'filtered_products': filtered,
                'input_products': len(candidates),
                'filtered_products_count': len(filtered),
                'threshold_used': settings.SEMANTIC_THRESHOLD,
                'execution_time': execution_time
            }

        except Exception as e:
            self.logger.error(f"Ошибка фильтрации: {e}", exc_info=True)
            return {'error': str(e), 'execution_time': time.time() - start_time}

    def _calculate_statistics(self, result: Dict, total_time: float) -> Dict:
        """Рассчитывает общую статистику"""

        stages = result.get('stages', {})

        return {
            'total_products_found': len(result.get('final_products', [])),
            'stages_timing': {
                'extraction': f"{stages.get('extraction', {}).get('execution_time', 0):.2f}s",
                'elasticsearch': f"{stages.get('elasticsearch', {}).get('execution_time', 0):.2f}s",
                'semantic': f"{stages.get('semantic', {}).get('execution_time', 0):.2f}s"
            },
            'total_time': f"{total_time:.2f}s"
        }

    def _log_top_results(self, products: List[Dict], top_n: int = 3):
        """Логирует топ результаты"""

        if not products:
            return

        self.logger.info(f"Топ-{top_n} результата:")

        for i, product in enumerate(products[:top_n], 1):
            self.logger.info(
                f"  {i}. {product.get('title', 'Без названия')} "
                f"(скор: {product.get('combined_score', 0):.3f})"
            )