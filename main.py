"""
Главный модуль системы поиска товаров для тендеров
"""

import time
import json
from typing import Dict, Any, List

from app.services.extractor import ConfigurableTermExtractor
from app.services.elasticsearch_service import ElasticsearchService
from app.services.semantic_search import SemanticSearchService
from app.services.attribute_matcher import AttributeMatcher
from app.config.settings import settings
from app.utils.logger import setup_logger


class TenderMatcher:
    """Основной класс для поиска товаров по тендеру"""

    def __init__(self):
        self.logger = setup_logger(__name__)

        self.logger.info("=" * 80)
        self.logger.info("Инициализация системы поиска товаров для тендеров")
        self.logger.info("=" * 80)

        # Инициализация сервисов
        self.logger.info("Инициализация сервисов...")

        self.extractor = ConfigurableTermExtractor()
        self.es_service = ElasticsearchService()
        self.semantic_service = SemanticSearchService()
        self.attribute_matcher = AttributeMatcher()

        self.logger.info("Все сервисы инициализированы")

    def process_tender(self, tender: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод обработки тендера

        Этапы:
        1. Извлечение терминов из тендера
        2. Поиск в Elasticsearch (500k → 2k)
        3. Семантическая фильтрация (2k → 500)
        4. Точное сопоставление характеристик (500 → финальные)
        """

        start_time = time.time()
        tender_name = tender.get('name', 'Без названия')

        self.logger.info("=" * 80)
        self.logger.info(f"НАЧАЛО ОБРАБОТКИ ТЕНДЕРА: {tender_name}")
        self.logger.info("=" * 80)

        results = {
            'tender': tender,
            'stages': {},
            'final_products': [],
            'statistics': {},
            'execution_time': 0
        }

        try:
            # ЭТАП 1: Извлечение терминов
            self.logger.info("\n📋 ЭТАП 1: ИЗВЛЕЧЕНИЕ ТЕРМИНОВ")
            self.logger.info("-" * 40)

            stage1_start = time.time()
            search_terms = self.extractor.extract_from_tender(tender)
            stage1_time = time.time() - stage1_start

            results['stages']['extraction'] = {
                'search_query': search_terms['search_query'],
                'boost_terms_count': len(search_terms['boost_terms']),
                'execution_time': stage1_time
            }

            self.logger.info(f"Извлечение завершено за {stage1_time:.2f} сек")

            # ЭТАП 2: Поиск в Elasticsearch
            self.logger.info("\n🔍 ЭТАП 2: ПОИСК В ELASTICSEARCH")
            self.logger.info("-" * 40)

            stage2_start = time.time()
            es_results = self.es_service.search_products(
                search_terms,
                size=settings.MAX_SEARCH_RESULTS
            )
            stage2_time = time.time() - stage2_start

            if 'error' in es_results:
                self.logger.error(f"Ошибка Elasticsearch: {es_results['error']}")
                results['error'] = es_results['error']
                return results

            candidates = es_results.get('candidates', [])
            results['stages']['elasticsearch'] = {
                'total_found': es_results['total_found'],
                'candidates_retrieved': len(candidates),
                'max_score': es_results.get('max_score', 0),
                'execution_time': stage2_time
            }

            self.logger.info(f"Elasticsearch завершен за {stage2_time:.2f} сек")
            self.logger.info(f"Найдено кандидатов: {len(candidates)} из {es_results['total_found']}")

            if not candidates:
                self.logger.warning("Elasticsearch не нашел подходящих товаров")
                results['final_products'] = []
                return results

            # ЭТАП 3: Семантическая фильтрация
            self.logger.info("\n🧠 ЭТАП 3: СЕМАНТИЧЕСКАЯ ФИЛЬТРАЦИЯ")
            self.logger.info("-" * 40)

            stage3_start = time.time()
            semantic_filtered = self.semantic_service.filter_by_similarity(
                tender,
                candidates,
                threshold=settings.SEMANTIC_THRESHOLD,
                top_k=settings.SEMANTIC_MAX_CANDIDATES
            )

            # Комбинируем скоры
            semantic_filtered = self.semantic_service.combine_with_es_scores(semantic_filtered)
            stage3_time = time.time() - stage3_start

            results['stages']['semantic'] = {
                'input_products': len(candidates),
                'filtered_products': len(semantic_filtered),
                'threshold_used': settings.SEMANTIC_THRESHOLD,
                'execution_time': stage3_time
            }

            self.logger.info(f"Семантическая фильтрация завершена за {stage3_time:.2f} сек")
            self.logger.info(f"После фильтрации: {len(semantic_filtered)} товаров")

            if not semantic_filtered:
                self.logger.warning("Семантическая фильтрация отсеяла все товары")
                results['final_products'] = []
                return results

            # ЭТАП 4: Точное сопоставление характеристик
            self.logger.info("\n✅ ЭТАП 4: СОПОСТАВЛЕНИЕ ХАРАКТЕРИСТИК")
            self.logger.info("-" * 40)

            stage4_start = time.time()
            final_products = []

            self.logger.info(f"Проверка {len(semantic_filtered)} товаров...")

            for i, product in enumerate(semantic_filtered):
                # Логируем прогресс каждые 10 товаров
                if i > 0 and i % 10 == 0:
                    self.logger.debug(f"Обработано {i}/{len(semantic_filtered)} товаров...")

                match_result = self.attribute_matcher.match_product(tender, product)

                if match_result['is_suitable']:
                    product['match_details'] = match_result
                    final_products.append(product)

                    self.logger.info(f"✓ Товар #{i+1} '{product['title'][:50]}...' ПОДХОДИТ")

            stage4_time = time.time() - stage4_start

            results['stages']['attribute_matching'] = {
                'input_products': len(semantic_filtered),
                'matched_products': len(final_products),
                'execution_time': stage4_time
            }

            self.logger.info(f"Сопоставление характеристик завершено за {stage4_time:.2f} сек")
            self.logger.info(f"Найдено подходящих товаров: {len(final_products)}")

            # Сортируем по комбинированному скору
            final_products.sort(
                key=lambda x: x.get('combined_score', 0),
                reverse=True
            )

            # Ограничиваем количество финальных результатов
            final_products = final_products[:settings.MAX_FINAL_RESULTS]

            results['final_products'] = final_products

            # Собираем общую статистику
            total_time = time.time() - start_time
            results['execution_time'] = total_time

            results['statistics'] = {
                'total_products_found': len(final_products),
                'stages_timing': {
                    'extraction': f"{stage1_time:.2f}s",
                    'elasticsearch': f"{stage2_time:.2f}s",
                    'semantic': f"{stage3_time:.2f}s",
                    'matching': f"{stage4_time:.2f}s"
                },
                'total_time': f"{total_time:.2f}s"
            }

            # Финальное логирование
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"ОБРАБОТКА ЗАВЕРШЕНА")
            self.logger.info(f"Найдено подходящих товаров: {len(final_products)}")
            self.logger.info(f"Общее время выполнения: {total_time:.2f} секунд")
            self.logger.info("=" * 80 + "\n")

            # Логируем топ-3 результата
            if final_products:
                self.logger.info("ТОП-3 РЕЗУЛЬТАТА:")
                for i, product in enumerate(final_products[:3]):
                    self.logger.info(f"{i+1}. {product['title']}")
                    self.logger.info(f"   Категория: {product.get('category', 'Н/Д')}")
                    self.logger.info(f"   Комбинированный скор: {product.get('combined_score', 0):.3f}")
                    self.logger.info(f"   Совпадение характеристик: {product['match_details']['match_percentage']:.1f}%")

        except Exception as e:
            self.logger.error(f"Критическая ошибка при обработке тендера: {e}", exc_info=True)
            results['error'] = str(e)

        return results


def main():
    """Основная функция"""

    logger = setup_logger(__name__)

    # Пример тендера
    tender_example = {
        "name": "Блоки для записей",
        "characteristics": [
            {
                "name": "Ширина",
                "value": "> 80 и ≤ 90 ММ",
                "type": "Количественная",
                "required": True
            },
            {
                "name": "Длина ",
                "value": "> 80 и ≤ 90 ММ",
                "type": "Качественная",
                "required": True
            },
            {
                "name": "Количество листов в блоке",
                "value": "≥ 500 ШТ",
                "type": "Качественная",
                "required": False
            }
        ]
    }

    try:
        # Инициализация системы
        matcher = TenderMatcher()

        # Обработка тендера
        results = matcher.process_tender(tender_example)

        # Сохранение результатов
        output_file = f"results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты сохранены в файл: {output_file}")

        # Вывод статистики матчера
        logger.info("\nСТАТИСТИКА ATTRIBUTE MATCHER:")
        stats = matcher.attribute_matcher.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Ошибка в основной программе: {e}", exc_info=True)


if __name__ == "__main__":
    main()