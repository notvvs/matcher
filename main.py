import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from app.pipeline.factory import create_pipeline
from app.utils.logger import setup_logger


def process_tender_example():
    """Обработка примера тендера"""

    logger = setup_logger(__name__)

    # Пример тендера
    tender = {
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
        # Создаем пайплайн
        logger.info("Инициализация системы...")
        pipeline = create_pipeline()

        # Обрабатываем тендер
        logger.info("Начало обработки тендера...")
        result = pipeline.process(tender)

        # Сохраняем результаты
        output_file = f"results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты сохранены в файл: {output_file}")

        # Выводим краткую статистику
        if 'error' not in result:
            print_results_summary(result)
        else:
            logger.error(f"Ошибка обработки: {result['error']}")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)


def process_tender_file(filename: str):
    """Обработка тендера из файла"""

    logger = setup_logger(__name__)

    try:
        # Загружаем тендер из файла
        with open(filename, 'r', encoding='utf-8') as f:
            tender = json.load(f)

        logger.info(f"Загружен тендер из файла: {filename}")

        # Создаем пайплайн
        pipeline = create_pipeline()

        # Обрабатываем
        result = pipeline.process(tender)

        # Сохраняем результаты
        output_file = f"results_{Path(filename).stem}_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты сохранены в файл: {output_file}")

        if 'error' not in result:
            print_results_summary(result)

    except FileNotFoundError:
        logger.error(f"Файл не найден: {filename}")
    except json.JSONDecodeError:
        logger.error(f"Ошибка парсинга JSON в файле: {filename}")
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}", exc_info=True)


def print_results_summary(result: Dict[str, Any]):
    """Выводит краткую сводку результатов"""

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОБРАБОТКИ ТЕНДЕРА")
    print("="*60)

    # Информация о тендере
    tender = result.get('tender', {})
    print(f"Тендер: {tender.get('name', 'Без названия')}")
    print(f"Характеристик: {len(tender.get('characteristics', []))}")

    # Статистика по этапам
    stages = result.get('stages', {})

    if 'extraction' in stages:
        extraction = stages['extraction']
        print(f"\nИзвлечение терминов:")
        print(f"  - Boost терминов: {extraction.get('boost_terms_count', 0)}")
        print(f"  - Время: {extraction.get('execution_time', 0):.2f}с")

    if 'elasticsearch' in stages:
        es_stage = stages['elasticsearch']
        print(f"\nПоиск в Elasticsearch:")
        print(f"  - Найдено всего: {es_stage.get('total_found', 0)}")
        print(f"  - Получено кандидатов: {es_stage.get('candidates_retrieved', 0)}")
        print(f"  - Время: {es_stage.get('execution_time', 0):.2f}с")

    if 'semantic' in stages:
        semantic = stages['semantic']
        print(f"\nСемантическая фильтрация:")
        print(f"  - Входных товаров: {semantic.get('input_products', 0)}")
        print(f"  - После фильтрации: {semantic.get('filtered_products_count', 0)}")
        print(f"  - Время: {semantic.get('execution_time', 0):.2f}с")

    # Финальные результаты
    final_products = result.get('final_products', [])
    print(f"\nФинальные результаты:")
    print(f"  - Найдено товаров: {len(final_products)}")
    print(f"  - Общее время: {result.get('execution_time', 0):.2f}с")

    # Топ-5 товаров
    if final_products:
        print(f"\nТоп-5 товаров:")
        for i, product in enumerate(final_products[:5], 1):
            print(f"  {i}. {product.get('title', 'Без названия')}")
            print(f"     Категория: {product.get('category', 'Н/Д')}")
            print(f"     Скор: {product.get('combined_score', 0):.3f}")

    print("="*60 + "\n")


def main():
    """Главная функция"""

    parser = argparse.ArgumentParser(
        description="Система поиска товаров для тендеров"
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Путь к JSON файлу с тендером'
    )

    parser.add_argument(
        '--example', '-e',
        action='store_true',
        help='Запустить с примером тендера'
    )

    args = parser.parse_args()

    if args.file:
        process_tender_file(args.file)
    elif args.example:
        process_tender_example()
    else:
        # По умолчанию запускаем пример
        process_tender_example()


if __name__ == "__main__":
    main()