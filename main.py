import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from app.utils.logger import setup_logger


def process_tender_with_llm(tender: Dict, use_llm: bool = True, model: str = "mistral:7b"):
    """Обработка тендера с опциональным использованием LLM"""

    logger = setup_logger(__name__)

    try:
        # Выбираем тип пайплайна
        if use_llm:
            logger.info(f"Инициализация LLM системы с моделью {model}...")

            # Импортируем LLM компоненты
            from app.pipeline.llm_factory import create_llm_pipeline
            from app.services.filtering.llm_filter import LLMConfig

            # Конфигурация LLM
            llm_config = LLMConfig(
                model_name=model,
                temperature=0.1,
                max_tokens=200,
                timeout=30,
                max_workers=4
            )

            pipeline = create_llm_pipeline(llm_config)
            pipeline_type = "LLM"
        else:
            logger.info("Инициализация системы с семантическим поиском...")

            # Импортируем стандартные компоненты
            from app.pipeline.factory import create_pipeline

            pipeline = create_pipeline()
            pipeline_type = "Semantic"

        # Обрабатываем тендер
        logger.info(f"Начало обработки тендера ({pipeline_type})...")
        result = pipeline.process(tender)

        # Сохраняем результаты
        output_file = f"results_{pipeline_type.lower()}_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты сохранены в файл: {output_file}")

        # Выводим статистику
        if 'error' not in result:
            print_results_summary(result, pipeline_type)
        else:
            logger.error(f"Ошибка обработки: {result['error']}")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)


def print_results_summary(result: Dict[str, Any], pipeline_type: str):
    """Выводит краткую сводку результатов"""

    print(f"\nРЕЗУЛЬТАТЫ ОБРАБОТКИ ТЕНДЕРА ({pipeline_type})")
    print("=" * 60)

    # Информация о тендере
    tender = result.get('tender', {})
    print(f"Тендер: {tender.get('name', 'Без названия')}")
    print(f"Характеристик: {len(tender.get('characteristics', []))}")

    # Обязательные характеристики
    required_chars = [c for c in tender.get('characteristics', []) if c.get('required', False)]
    if required_chars:
        print(f"\nОбязательные характеристики:")
        for char in required_chars:
            print(f"  - {char['name']}: {char['value']}")

    # Статистика по этапам
    stages = result.get('stages', {})

    if 'extraction' in stages:
        extraction = stages['extraction']
        print(f"\nИзвлечение терминов:")
        print(f"  Поисковый запрос: '{extraction.get('search_query', '')}'")
        print(f"  Boost терминов: {extraction.get('boost_terms_count', 0)}")
        print(f"  Время: {extraction.get('execution_time', 0):.2f}с")

    if 'elasticsearch' in stages:
        es_stage = stages['elasticsearch']
        print(f"\nПоиск в Elasticsearch:")
        print(f"  Найдено всего: {es_stage.get('total_found', 0)}")
        print(f"  Получено кандидатов: {es_stage.get('candidates_retrieved', 0)}")
        print(f"  Время: {es_stage.get('execution_time', 0):.2f}с")

    # В зависимости от типа пайплайна
    if pipeline_type == "LLM" and 'llm_analysis' in stages:
        llm_stage = stages['llm_analysis']
        print(f"\nLLM анализ:")
        print(f"  Входных товаров: {llm_stage.get('input_products', 0)}")
        print(f"  После анализа: {llm_stage.get('filtered_products_count', 0)}")
        print(f"  Порог: {llm_stage.get('threshold_used', 0)}")
        print(f"  Время: {llm_stage.get('execution_time', 0):.2f}с")
    elif 'semantic' in stages:
        semantic = stages['semantic']
        print(f"\nСемантическая фильтрация:")
        print(f"  Входных товаров: {semantic.get('input_products', 0)}")
        print(f"  После фильтрации: {semantic.get('filtered_products_count', 0)}")
        print(f"  Время: {semantic.get('execution_time', 0):.2f}с")

    # Финальные результаты
    final_products = result.get('final_products', [])
    print(f"\nФинальные результаты:")
    print(f"  Найдено товаров: {len(final_products)}")
    print(f"  Общее время: {result.get('execution_time', 0):.2f}с")

    # Топ товаров
    if final_products:
        print(f"\nТоп-10 товаров:")
        for i, product in enumerate(final_products[:10], 1):
            print(f"\n  {i}. {product.get('title', 'Без названия')}")
            print(f"     Категория: {product.get('category', 'Н/Д')}")

            if product.get('brand'):
                print(f"     Бренд: {product['brand']}")

            print(f"     Комбинированный скор: {product.get('combined_score', 0):.3f}")

            # Детали скоров
            if pipeline_type == "LLM":
                print(f"     LLM скор: {product.get('llm_score', 0):.3f}")
                reasoning = product.get('llm_reasoning', '')
                if reasoning:
                    # Ограничиваем длину обоснования
                    if len(reasoning) > 150:
                        reasoning = reasoning[:150] + "..."
                    print(f"     Обоснование: {reasoning}")
            else:
                print(f"     Семантический скор: {product.get('semantic_score', 0):.3f}")

            print(f"     ES скор (норм.): {product.get('normalized_es_score', 0):.3f}")

            # Показываем ключевые атрибуты
            attrs = product.get('attributes', [])
            if attrs:
                print("     Ключевые характеристики:")
                for attr in attrs[:3]:  # Первые 3 атрибута
                    attr_name = attr.get('attr_name', '')
                    attr_value = attr.get('attr_value', '')
                    if attr_name and attr_value:
                        print(f"       - {attr_name}: {attr_value}")

    print()


def process_tender_file(filename: str, use_llm: bool = True, model: str = "mistral:7b"):
    """Обработка тендера из файла"""

    logger = setup_logger(__name__)

    try:
        # Загружаем тендер из файла
        with open(filename, 'r', encoding='utf-8') as f:
            tender = json.load(f)

        logger.info(f"Загружен тендер из файла: {filename}")

        # Обрабатываем
        process_tender_with_llm(tender, use_llm=use_llm, model=model)

    except FileNotFoundError:
        logger.error(f"Файл не найден: {filename}")
    except json.JSONDecodeError:
        logger.error(f"Ошибка парсинга JSON в файле: {filename}")
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}", exc_info=True)


def main():
    """Главная функция с поддержкой LLM"""

    parser = argparse.ArgumentParser(
        description="Система поиска товаров для тендеров с LLM анализом"
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Путь к JSON файлу с тендером'
    )

    parser.add_argument(
        '--llm',
        action='store_true',
        help='Использовать LLM для анализа (требуется Ollama)'
    )

    parser.add_argument(
        '--semantic',
        action='store_true',
        help='Использовать семантический поиск (старый метод)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='mistral:7b',
        help='Модель LLM для использования (по умолчанию: mistral:7b)'
    )

    parser.add_argument(
        '--example', '-e',
        action='store_true',
        help='Запустить с примером тендера'
    )

    args = parser.parse_args()

    # Определяем метод
    # По умолчанию используем LLM, если явно не указан semantic
    use_llm = not args.semantic if not args.llm else True

    # Пример тендера
    example_tender = {
        "name": "Точилка канцелярская для карандашей",
        "characteristics": [
            {
                "name": "Тип",
                "value": "Ручная",
                "type": "Качественная",
                "required": True
            },
            {
                "name": "Наличие контейнера для стружки",
                "value": "Да",
                "type": "Качественная",
                "required": True
            },
            {
                "name": "Количество отверстий",
                "value": "1",
                "type": "Качественная",
                "required": True
            },
        ]
    }

    if args.file:
        # Загружаем из файла
        process_tender_file(args.file, use_llm=use_llm, model=args.model)
    else:
        # Используем пример
        process_tender_with_llm(example_tender, use_llm=use_llm, model=args.model)


if __name__ == "__main__":
    main()