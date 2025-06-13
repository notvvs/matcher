"""
Основной файл для поиска товаров по тендеру
"""


import sys
import json

from app.config.settings import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.services.extractor import ConfigurableTermExtractor
from app.services.semantic_search import SemanticSearchService


def search_products(tender_data):
    """
    Поиск товаров по тендеру

    Args:
        tender_data: Данные тендера

    Returns:
        Список подходящих товаров, отсортированных по релевантности
    """

    print(f"\n🔍 Поиск товаров для тендера: {tender_data.get('name', 'Без названия')}")
    print("=" * 80)

    # 1. Извлечение терминов
    print("\n1️⃣ Извлечение терминов...")
    extractor = ConfigurableTermExtractor()
    search_terms = extractor.extract_from_tender(tender_data)

    # 2. Поиск в Elasticsearch
    print("\n2️⃣ Поиск в Elasticsearch...")
    es_service = ElasticsearchService()
    es_results = es_service.search_products(search_terms, size=settings.MAX_SEARCH_RESULTS)

    if 'error' in es_results:
        print(f"❌ Ошибка поиска: {es_results['error']}")
        return []

    es_candidates = es_results.get('candidates', [])
    print(f"✅ Найдено в ES: {len(es_candidates)} товаров")

    if not es_candidates:
        print("❌ Товары не найдены")
        return []

    # 3. Семантическая фильтрация
    print("\n3️⃣ Семантическая фильтрация...")
    semantic_service = SemanticSearchService()

    semantic_results = semantic_service.filter_by_similarity(
        tender_data,
        es_candidates,
        threshold=settings.SEMANTIC_THRESHOLD,
        top_k=-1  # Без ограничений
    )


    # 4. Комбинирование скоров
    final_products = semantic_service.combine_with_es_scores(semantic_results)
    print(f"✅ После семантической фильтрации: {len(final_products)} товаров")

    return final_products


def print_results(products, max_show=20):
    """
    Вывод результатов поиска

    Args:
        products: Список найденных товаров
        max_show: Максимальное количество товаров для показа
    """

    if not products:
        print("\n❌ Подходящие товары не найдены")
        return

    print(f"\n📋 НАЙДЕННЫЕ ТОВАРЫ (топ-{min(len(products), max_show)} из {len(products)}):")
    print("=" * 80)

    for i, product in enumerate(products[:max_show], 1):
        if product['combined_score'] > 0.5:
            print(f"\n{i}. {product['title']}")
            print(f"   Категория: {product['category']}")

            if product.get('brand') and product['brand'] != '-':
                print(f"   Бренд: {product['brand']}")

            # Скоры
            print(f"   Релевантность: {product['combined_score']:.3f}")
            print(f"   ├─ Семантическая близость: {product['semantic_score']:.3f}")
            print(f"   └─ Совпадение по ключевым словам: {product['normalized_es_score']:.3f}")

            # Основные атрибуты
            if product.get('attributes'):
                print("   Атрибуты:")
                for attr in product['attributes'][:5]:
                    print(f"      • {attr['attr_name']}: {attr['attr_value']}")
                if len(product['attributes']) > 5:
                    print(f"      ... и еще {len(product['attributes']) - 5} атрибутов")

        if len(products) > max_show:
            print(f"\n... и еще {len(products) - max_show} товаров")


def main():
    """Основная функция"""

    # Пример тендера (можно заменить на загрузку из файла или API)
    tender = {
        "name": "Флеш-накопитель",
        "okpd2Code": "26.20.22.110",
        "characteristics": [
            {
                "name": "Объем памяти",
                "value": "≥ 32 ГБАЙТ",
                "type": "Качественная",
                "required": True
            },
            {
                "name": "Тип",
                "value": "USB Flash",
                "type": "Качественная",
                "required": True
            },
        ]
    }

    try:
        # Поиск товаров
        products = search_products(tender)

        # Вывод результатов
        print_results(products, max_show=1000)

        # Опционально: сохранение результатов
        if len(sys.argv) > 1 and sys.argv[1] == '--save':
            filename = f"search_results_{int(time.time())}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'tender': tender,
                    'products': products,
                    'total_found': len(products)
                }, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Результаты сохранены в {filename}")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import time

    main()