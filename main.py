"""
Основной файл для поиска товаров по тендеру с attribute matching
"""

import sys
import json
import time
from typing import List, Dict, Any, Optional

from app.config.settings import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.services.extractor import ConfigurableTermExtractor
from app.services.semantic_search import SemanticSearchService
from app.services.attribute_matcher import AttributeMatcher


class TenderSearchPipeline:
    """Полный pipeline поиска товаров для тендера"""

    def __init__(self):
        self.extractor = ConfigurableTermExtractor()
        self.es_service = ElasticsearchService()
        self.semantic_service = SemanticSearchService()
        self.attribute_matcher = AttributeMatcher()

    def search_products(self,
                       tender: Dict[str, Any],
                       use_semantic: bool = True,
                       use_attribute_matching: bool = True,
                       relevance_threshold: float = None,
                       max_results: int = None) -> Dict[str, Any]:
        """
        Полный поиск товаров для тендера

        Args:
            tender: Данные тендера
            use_semantic: Использовать ли семантическую фильтрацию
            use_attribute_matching: Использовать ли сопоставление характеристик
            relevance_threshold: Минимальный порог релевантности (None = автоматически)
            max_results: Максимальное количество результатов

        Returns:
            {
                'products': [...],
                'stats': {...},
                'timings': {...},
                'tender_info': {...}
            }
        """

        start_time = time.time()

        # Инициализация результатов
        result = {
            'products': [],
            'stats': {
                'initial_candidates': 0,
                'after_elasticsearch': 0,
                'after_semantic': 0,
                'after_attribute_matching': 0,
                'suitable_products': 0,
                'avg_match_percentage': 0.0
            },
            'timings': {},
            'tender_info': {
                'name': tender.get('name', 'Без названия'),
                'total_characteristics': len(tender.get('characteristics', [])),
                'required_characteristics': sum(
                    1 for c in tender.get('characteristics', [])
                    if c.get('required', False)
                )
            }
        }

        print(f"\n{'='*80}")
        print(f"🔍 ПОИСК ТОВАРОВ: {result['tender_info']['name']}")
        print(f"   Характеристик: {result['tender_info']['total_characteristics']} "
              f"(обязательных: {result['tender_info']['required_characteristics']})")
        print(f"{'='*80}")

        # ЭТАП 1: Извлечение терминов и поиск в Elasticsearch
        print("\n📝 Этап 1: Поиск в Elasticsearch...")
        stage_start = time.time()

        search_terms = self.extractor.extract_from_tender(tender)
        es_results = self.es_service.search_products(
            search_terms,
            size=settings.MAX_SEARCH_RESULTS
        )

        if 'error' in es_results:
            result['error'] = es_results['error']
            return result

        products = es_results.get('candidates', [])
        result['stats']['after_elasticsearch'] = len(products)
        result['timings']['elasticsearch'] = time.time() - stage_start

        print(f"✅ Найдено: {len(products)} товаров за {result['timings']['elasticsearch']:.2f}с")

        if not products:
            return result

        # ЭТАП 2: Семантическая фильтрация (опционально)
        if use_semantic and len(products) > 20:
            print("\n🧠 Этап 2: Семантическая фильтрация...")
            stage_start = time.time()

            products = self.semantic_service.filter_by_similarity(
                tender,
                products,
                threshold=settings.SEMANTIC_THRESHOLD
            )

            products = self.semantic_service.combine_with_es_scores(products)

            result['stats']['after_semantic'] = len(products)
            result['timings']['semantic'] = time.time() - stage_start

            print(f"✅ После семантики: {len(products)} товаров за {result['timings']['semantic']:.2f}с")
        else:
            # Нормализуем ES скоры если семантика не использовалась
            for product in products:
                es_score = product.get('elasticsearch_score', 0)
                product['combined_score'] = min(es_score / 10.0, 1.0)

            result['stats']['after_semantic'] = len(products)
            result['timings']['semantic'] = 0

        # Определяем порог релевантности
        if relevance_threshold is None:
            relevance_threshold = self._calculate_relevance_threshold(products, tender)

        # Фильтруем по релевантности
        products = [
            p for p in products
            if p.get('combined_score', 0) >= relevance_threshold
        ]

        print(f"\n📊 Порог релевантности: {relevance_threshold:.3f}")
        print(f"   Товаров для детальной проверки: {len(products)}")

        # ЭТАП 3: Сопоставление характеристик (опционально)
        if use_attribute_matching and products and tender.get('characteristics'):
            print(f"\n🎯 Этап 3: Проверка характеристик...")
            stage_start = time.time()

            matched_products = []
            total_match_percentage = 0

            for i, product in enumerate(products):
                # Показываем прогресс для больших списков
                if i > 0 and i % 100 == 0:
                    print(f"   Проверено: {i}/{len(products)} товаров...")

                # Сопоставляем характеристики
                match_result = self.attribute_matcher.match_product(tender, product)

                # Добавляем результаты к товару
                product['attribute_match'] = match_result

                # Считаем финальный скор
                product['final_score'] = self._calculate_final_score(
                    product.get('combined_score', 0),
                    match_result
                )

                # Добавляем только подходящие товары
                if match_result['is_suitable']:
                    matched_products.append(product)
                    total_match_percentage += match_result['match_percentage']
                    result['stats']['suitable_products'] += 1

            # Сортируем по финальному скору
            matched_products.sort(key=lambda x: x['final_score'], reverse=True)

            products = matched_products
            result['stats']['after_attribute_matching'] = len(products)
            result['timings']['attribute_matching'] = time.time() - stage_start

            # Средний процент совпадения
            if result['stats']['suitable_products'] > 0:
                result['stats']['avg_match_percentage'] = (
                    total_match_percentage / result['stats']['suitable_products']
                )

            print(f"✅ Подходящих товаров: {len(products)} из {len(products) + (result['stats']['after_semantic'] - len(products))}")
            print(f"   Средний процент совпадения: {result['stats']['avg_match_percentage']:.1f}%")
            print(f"   Время проверки: {result['timings']['attribute_matching']:.2f}с")
        else:
            result['stats']['after_attribute_matching'] = len(products)
            result['timings']['attribute_matching'] = 0

        # Ограничиваем количество результатов
        if max_results and len(products) > max_results:
            products = products[:max_results]

        # Финальная статистика
        result['products'] = products
        result['timings']['total'] = time.time() - start_time

        print(f"\n📈 ИТОГО:")
        print(f"   Финальных результатов: {len(products)}")
        print(f"   Общее время: {result['timings']['total']:.2f}с")

        # Показываем статистику матчера
        if use_attribute_matching:
            matcher_stats = self.attribute_matcher.get_stats()
            if matcher_stats['total_matches'] > 0:
                print(f"\n📊 Статистика attribute matching:")
                print(f"   Успешных совпадений: {matcher_stats['success_rate']:.1%}")
                print(f"   Частичных совпадений: {matcher_stats['partial_rate']:.1%}")

        return result

    def _calculate_relevance_threshold(self, products: List[Dict],
                                     tender: Dict[str, Any]) -> float:
        """Автоматически определяет порог релевантности"""

        if not products:
            return 0.2

        scores = [p.get('combined_score', 0) for p in products]
        scores.sort(reverse=True)

        # Простая эвристика на основе количества характеристик
        char_count = len(tender.get('characteristics', []))

        if char_count <= 3:
            # Простой тендер - берем топ 25%
            threshold_index = len(scores) // 4
        elif char_count <= 7:
            # Средний тендер - берем топ 35%
            threshold_index = int(len(scores) * 0.35)
        else:
            # Сложный тендер - берем топ 50%
            threshold_index = len(scores) // 2

        # Ограничиваем количество для производительности
        threshold_index = min(threshold_index, 500)

        if threshold_index < len(scores):
            return max(0.15, scores[threshold_index])
        else:
            return 0.15

    def _calculate_final_score(self, relevance_score: float,
                             match_result: Dict[str, Any]) -> float:
        """Вычисляет финальный скор товара"""

        # Базовая формула: 40% поисковая релевантность + 60% совпадение характеристик
        base_score = 0.4 * relevance_score + 0.6 * match_result['score']

        # Бонус за высокую уверенность
        if match_result['confidence'] > 0.9:
            base_score *= 1.1

        # Бонус за 100% совпадение
        if match_result['match_percentage'] == 100:
            base_score *= 1.15

        return min(1.0, base_score)


def print_results(results: Dict[str, Any], max_show: int = 20, detailed: bool = True):
    """Красивый вывод результатов поиска"""

    products = results['products']

    if not products:
        print("\n❌ Подходящие товары не найдены")
        return

    print(f"\n📋 РЕЗУЛЬТАТЫ ПОИСКА (топ-{min(len(products), max_show)} из {len(products)}):")
    print("="*80)

    for i, product in enumerate(products[:max_show], 1):
        print(f"\n{i}. {product['title']}")
        print(f"   Категория: {product.get('category', 'Не указана')}")

        if product.get('brand') and product['brand'] != '-':
            print(f"   Бренд: {product['brand']}")

        # Скоры
        print(f"\n   📊 Оценки:")
        print(f"   • Итоговая релевантность: {product.get('final_score', 0):.3f}")

        if 'combined_score' in product:
            print(f"   • Поисковая релевантность: {product['combined_score']:.3f}")

        # Результаты attribute matching
        if 'attribute_match' in product:
            match = product['attribute_match']
            print(f"   • Совпадение характеристик: {match['score']:.3f} "
                  f"({match['match_percentage']:.0f}%)")
            print(f"   • Уверенность: {match['confidence']:.3f}")

            if match['total_required'] > 0:
                print(f"\n   ✅ Обязательные характеристики: "
                      f"{match['matched_required']}/{match['total_required']}")

            if match['total_optional'] > 0:
                print(f"   ➕ Опциональные характеристики: "
                      f"{match['matched_optional']}/{match['total_optional']}")

            # Детали по характеристикам
            if detailed and match['details']:
                print("\n   📝 Детали проверки:")

                # Сначала показываем несовпавшие обязательные
                for detail in match['details']:
                    if detail['required'] and not detail['matched']:
                        char = detail['characteristic']
                        print(f"   ❌ {char['name']}: {char['value']} - {detail['reason']}")

                # Потом совпавшие
                shown_matched = 0
                for detail in match['details']:
                    if detail['matched'] and shown_matched < 3:
                        char = detail['characteristic']
                        print(f"   ✅ {char['name']}: {char['value']} - {detail['reason']}")
                        shown_matched += 1

        # Атрибуты товара
        if product.get('attributes') and detailed:
            print(f"\n   📦 Основные атрибуты:")
            for attr in product['attributes'][:5]:
                print(f"   • {attr['attr_name']}: {attr['attr_value']}")

            if len(product['attributes']) > 5:
                print(f"   ... и еще {len(product['attributes']) - 5} атрибутов")

    if len(products) > max_show:
        print(f"\n... и еще {len(products) - max_show} товаров")


def main():
    """Основная функция с примером использования"""

    # Пример тендера
    tender = {
        "name": "Блоки для записей",
        "okpd2Code": "17.23.13.199",
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
            },

        ]
    }

    try:
        # Создаем pipeline
        pipeline = TenderSearchPipeline()

        # Запускаем поиск
        results = pipeline.search_products(
            tender,
            use_semantic=True,
            use_attribute_matching=True,
            relevance_threshold=None,  # Автоматическое определение
            max_results=50
        )

        # Выводим результаты
        print_results(results, max_show=20, detailed=True)

        # Сохраняем результаты если указан флаг
        if len(sys.argv) > 1 and sys.argv[1] == '--save':
            filename = f"search_results_{int(time.time())}.json"

            # Подготавливаем данные для сохранения
            save_data = {
                'tender': tender,
                'results': {
                    'products': [
                        {
                            'id': p.get('id', ''),
                            'title': p.get('title', ''),
                            'category': p.get('category', ''),
                            'brand': p.get('brand', ''),
                            'final_score': p.get('final_score', 0),
                            'match_percentage': p.get('attribute_match', {}).get('match_percentage', 0),
                            'matched_required': p.get('attribute_match', {}).get('matched_required', 0),
                            'total_required': p.get('attribute_match', {}).get('total_required', 0)
                        }
                        for p in results['products']
                    ],
                    'stats': results['stats'],
                    'timings': results['timings']
                }
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f"\n💾 Результаты сохранены в {filename}")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()