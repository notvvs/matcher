"""
Главный модуль системы поиска товаров для тендеров
"""

import time
import json
from typing import Dict, Any, List

from app.services.extractor import ConfigurableTermExtractor
from app.services.elasticsearch_service import ElasticsearchService
from app.services.semantic_search import SemanticSearchService

from app.config.settings import settings
from app.utils.logger import setup_logger


class TenderMatcher:
    """Основной класс для поиска товаров по тендеру"""

    def __init__(self):
        self.logger = setup_logger(__name__)

        self.logger.info("Инициализация системы...")

        try:
            self.extractor = ConfigurableTermExtractor()
            self.es_service = ElasticsearchService()
            self.semantic_service = SemanticSearchService()

            self.logger.info("Система готова к работе")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {e}")
            raise

    def process_tender(self, tender: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод обработки тендера

        Этапы:
        1. Извлечение терминов из тендера
        2. Поиск в Elasticsearch (500k → 2k)
        3. Семантическая фильтрация (2k → финальные результаты)
        """

        start_time = time.time()
        tender_name = tender.get('name', 'Без названия')

        self.logger.info(f"Обработка тендера: {tender_name}")

        results = {
            'tender': tender,
            'stages': {},
            'final_products': [],
            'statistics': {},
            'execution_time': 0
        }

        try:
            # Этап 1: Извлечение терминов
            self.logger.info("Извлечение терминов...")
            stage1_start = time.time()

            search_terms = self.extractor.extract_from_tender(tender)

            stage1_time = time.time() - stage1_start

            results['stages']['extraction'] = {
                'search_query': search_terms['search_query'],
                'boost_terms_count': len(search_terms['boost_terms']),
                'execution_time': stage1_time
            }

            self.logger.info(f"Извлечено терминов: {len(search_terms['boost_terms'])}, время: {stage1_time:.2f}с")

            # Этап 2: Поиск в Elasticsearch
            self.logger.info("Поиск в Elasticsearch...")
            stage2_start = time.time()

            es_results = self.es_service.search_products(
                search_terms,
                size=settings.MAX_SEARCH_RESULTS
            )

            stage2_time = time.time() - stage2_start

            if 'error' in es_results:
                self.logger.error(f"Ошибка ES: {es_results['error']}")
                results['error'] = es_results['error']
                return results

            candidates = es_results.get('candidates', [])

            results['stages']['elasticsearch'] = {
                'total_found': es_results['total_found'],
                'candidates_retrieved': len(candidates),
                'max_score': es_results.get('max_score', 0),
                'execution_time': stage2_time
            }

            self.logger.info(f"Найдено: {len(candidates)} из {es_results['total_found']}, время: {stage2_time:.2f}с")

            if not candidates:
                self.logger.warning("Товары не найдены")
                return results

            # Этап 3: Семантическая фильтрация
            self.logger.info("Семантическая фильтрация...")
            stage3_start = time.time()

            semantic_filtered = self.semantic_service.filter_by_similarity(
                tender,
                candidates,
                threshold=settings.SEMANTIC_THRESHOLD,
                top_k=settings.SEMANTIC_MAX_CANDIDATES
            )

            semantic_filtered = self.semantic_service.combine_with_es_scores(semantic_filtered)
            stage3_time = time.time() - stage3_start

            results['stages']['semantic'] = {
                'input_products': len(candidates),
                'filtered_products': len(semantic_filtered),
                'threshold_used': settings.SEMANTIC_THRESHOLD,
                'execution_time': stage3_time
            }

            self.logger.info(f"Отфильтровано: {len(semantic_filtered)} из {len(candidates)}, время: {stage3_time:.2f}с")

            if not semantic_filtered:
                self.logger.warning("Все товары отфильтрованы")
                return results

            # Семантически отфильтрованные товары становятся финальными
            final_products = semantic_filtered[:settings.MAX_FINAL_RESULTS]
            results['final_products'] = final_products

            # Общая статистика
            total_time = time.time() - start_time
            results['execution_time'] = total_time

            results['statistics'] = {
                'total_products_found': len(final_products),
                'stages_timing': {
                    'extraction': f"{stage1_time:.2f}s",
                    'elasticsearch': f"{stage2_time:.2f}s",
                    'semantic': f"{stage3_time:.2f}s"
                },
                'total_time': f"{total_time:.2f}s"
            }

            # Итоговая информация
            self.logger.info(f"Обработка завершена: найдено {len(final_products)} товаров за {total_time:.2f}с")

            # Топ результаты
            if final_products:
                self.logger.info("Топ-3 результата:")
                for i, product in enumerate(final_products[:3], 1):
                    self.logger.info(f"  {i}. {product['title']} (скор: {product.get('combined_score', 0):.3f})")

        except Exception as e:
            self.logger.error(f"Критическая ошибка: {e}", exc_info=True)
            results['error'] = str(e)

        return results


def main():
    """Основная функция"""

    logger = setup_logger(__name__)

    # Пример тендера
    tender_example = {
        "name": "Блокнот",
        "characteristics": [
            {
                "name": "Вид линовки",
                "value": "Клетка",
                "type": "Качественная",
                "required": True
            },
            {
                "name": "Количество листов",
                "value": "≥ 40 ШТ",
                "type": "Количественная",
                "required": True
            },
            {
                "name": "Материал обложки",
                "value": "Картон",
                "type": "Качественная",
                "required": True
            }
        ]
    }

    try:
        # Инициализация и обработка
        matcher = TenderMatcher()
        results = matcher.process_tender(tender_example)

        # Сохранение результатов
        output_file = f"results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты сохранены: {output_file}")

    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    main()