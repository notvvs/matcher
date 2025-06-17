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

        self.logger.info(f"Начало обработки тендера: {tender_name}")
        self.logger.info(f"Характеристик: {len(tender.get('characteristics', []))}")

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
                    'total_time': f"{time.time() - start_time:.2f}s"
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

            # Логируем топ результаты с обоснованием
            self._log_top_results_with_reasoning(final_products)

        except Exception as e:
            self.logger.error(f"Критическая ошибка в пайплайне: {e}", exc_info=True)
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time

        return result

    def _execute_extraction(self, tender: Dict) -> Dict:
        """Этап 1: Извлечение терминов"""

        self.logger.info("=" * 50)
        self.logger.info("ЭТАП 1: Извлечение терминов")
        start_time = time.time()

        try:
            search_terms = self.term_extractor.extract_from_tender(tender)

            execution_time = time.time() - start_time

            # Логируем детали
            self.logger.info(f"Основной запрос: '{search_terms['search_query']}'")
            self.logger.info(f"Извлечено boost-терминов: {len(search_terms['boost_terms'])}")

            if search_terms['boost_terms']:
                self.logger.info("Топ-5 терминов с весами:")
                sorted_terms = sorted(
                    search_terms['boost_terms'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for term, weight in sorted_terms[:5]:
                    self.logger.info(f"  - {term}: {weight:.2f}")

            self.logger.info(f"Время выполнения: {execution_time:.2f}с")

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

        self.logger.info("=" * 50)
        self.logger.info("ЭТАП 2: Поиск в Elasticsearch")
        start_time = time.time()

        try:
            # Для LLM анализа берем меньше кандидатов чем для семантического
            max_candidates = min(settings.MAX_SEARCH_RESULTS, 100)

            self.logger.info(f"Максимум кандидатов для LLM: {max_candidates}")

            es_results = self.es_client.search_products(
                search_terms,
                size=max_candidates
            )

            execution_time = time.time() - start_time

            if 'error' in es_results:
                return {
                    'error': es_results['error'],
                    'execution_time': execution_time
                }

            candidates = es_results.get('candidates', [])

            self.logger.info(f"Найдено всего: {es_results['total_found']}")
            self.logger.info(f"Получено кандидатов: {len(candidates)}")
            self.logger.info(f"Максимальный ES скор: {es_results.get('max_score', 0):.2f}")
            self.logger.info(f"Время выполнения: {execution_time:.2f}с")

            # Логируем топ-3 ES результата
            if candidates:
                self.logger.info("Топ-3 по ES скору:")
                for i, candidate in enumerate(candidates[:3], 1):
                    self.logger.info(
                        f"  {i}. {candidate.get('title', '')[:60]}... "
                        f"(ES скор: {candidate.get('elasticsearch_score', 0):.2f})"
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

    def _execute_llm_filtering(self, tender: Dict, candidates: List[Dict]) -> Dict:
        """Этап 3: LLM анализ и фильтрация"""

        self.logger.info("=" * 50)
        self.logger.info("ЭТАП 3: LLM анализ соответствия")
        start_time = time.time()

        try:
            self.logger.info(f"Товаров для LLM анализа: {len(candidates)}")

            # LLM анализ
            threshold = settings.LLM_THRESHOLD
            self.logger.info(f"Порог LLM скора: {threshold}")

            filtered = self.llm_filter.filter_by_llm_analysis(
                tender,
                candidates,
                threshold=threshold,
                top_k=50  # Ограничиваем для производительности
            )

            self.logger.info(f"После LLM фильтрации: {len(filtered)} товаров")

            if not filtered:
                self.logger.warning("LLM не нашла подходящих товаров")
                return {
                    'filtered_products': [],
                    'input_products': len(candidates),
                    'filtered_products_count': 0,
                    'threshold_used': threshold,
                    'execution_time': time.time() - start_time
                }

            # Комбинирование скоров
            self.logger.info("Комбинирование ES и LLM скоров...")
            filtered = self.score_combiner.combine_scores(filtered)

            execution_time = time.time() - start_time

            self.logger.info(f"Финальная фильтрация: {len(filtered)} из {len(candidates)}")
            self.logger.info(f"Время выполнения: {execution_time:.2f}с")

            return {
                'filtered_products': filtered,
                'input_products': len(candidates),
                'filtered_products_count': len(filtered),
                'threshold_used': threshold,
                'execution_time': execution_time
            }

        except Exception as e:
            self.logger.error(f"Ошибка LLM фильтрации: {e}", exc_info=True)
            return {'error': str(e), 'execution_time': time.time() - start_time}

    def _calculate_statistics(self, result: Dict, total_time: float) -> Dict:
        """Рассчитывает общую статистику"""

        stages = result.get('stages', {})

        stats = {
            'total_products_found': len(result.get('final_products', [])),
            'stages_timing': {
                'extraction': f"{stages.get('extraction', {}).get('execution_time', 0):.2f}s",
                'elasticsearch': f"{stages.get('elasticsearch', {}).get('execution_time', 0):.2f}s",
                'llm_analysis': f"{stages.get('llm_analysis', {}).get('execution_time', 0):.2f}s"
            },
            'total_time': f"{total_time:.2f}s",
            'products_per_stage': {
                'after_es': stages.get('elasticsearch', {}).get('candidates_retrieved', 0),
                'after_llm': stages.get('llm_analysis', {}).get('filtered_products_count', 0),
                'final': len(result.get('final_products', []))
            }
        }

        # Расчет эффективности фильтрации
        if stats['products_per_stage']['after_es'] > 0:
            llm_filter_rate = (
                                      1 - stats['products_per_stage']['after_llm'] /
                                      stats['products_per_stage']['after_es']
                              ) * 100
            stats['llm_filter_rate'] = f"{llm_filter_rate:.1f}%"

        return stats

    def _log_top_results_with_reasoning(self, products: List[Dict], top_n: int = 5):
        """Логирует топ результаты с обоснованием от LLM"""

        if not products:
            return

        self.logger.info("=" * 50)
        self.logger.info(f"ТОП-{min(top_n, len(products))} ФИНАЛЬНЫХ РЕЗУЛЬТАТОВ:")

        for i, product in enumerate(products[:top_n], 1):
            self.logger.info(f"\n{i}. {product.get('title', 'Без названия')}")
            self.logger.info(f"   Категория: {product.get('category', 'Н/Д')}")
            if product.get('brand'):
                self.logger.info(f"   Бренд: {product['brand']}")

            self.logger.info(f"   Комбинированный скор: {product.get('combined_score', 0):.3f}")
            self.logger.info(f"   LLM скор: {product.get('llm_score', 0):.3f}")
            self.logger.info(f"   ES скор (норм.): {product.get('normalized_es_score', 0):.3f}")

            reasoning = product.get('llm_reasoning', '')
            if reasoning:
                self.logger.info(f"   LLM обоснование: {reasoning}")

        self.logger.info("=" * 50)