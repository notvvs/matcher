import logging
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from app.pipeline.llm_factory import create_llm_pipeline, create_hybrid_pipeline
from app.services.filtering.llm_filter import LLMConfig

logger = logging.getLogger(__name__)

def process_tender_with_llm(tender: Dict, use_llm: bool = True, model: str = "mistral:7b"):
    """Обработка тендера с опциональным использованием LLM"""


    try:
        # Выбираем тип пайплайна
        if use_llm:
            logger.info(f"Инициализация LLM системы с моделью {model}...")

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
            pipeline = create_llm_pipeline()
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

    # Статистика по этапам
    stages = result.get('stages', {})

    if 'extraction' in stages:
        extraction = stages['extraction']
        print(f"\nИзвлечение терминов:")
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

    # Топ-5 товаров
    if final_products:
        print(f"\nТоп-5 товаров:")
        for i, product in enumerate(final_products[:5], 1):
            print(f"\n  {i}. {product.get('title', 'Без названия')}")
            print(f"     Категория: {product.get('category', 'Н/Д')}")
            print(f"     Комбинированный скор: {product.get('combined_score', 0):.3f}")

            # Детали скоров
            if pipeline_type == "LLM":
                print(f"     LLM скор: {product.get('llm_score', 0):.3f}")
                reasoning = product.get('llm_reasoning', '')
                if reasoning:
                    print(f"     Обоснование: {reasoning[:100]}...")
            else:
                print(f"     Семантический скор: {product.get('semantic_score', 0):.3f}")

            print(f"     ES скор: {product.get('normalized_es_score', 0):.3f}")

    print()


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

    # Пример тендера
    example_tender = {
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
            },
            {
                "name": "Формат",
                "value": "А4",
                "type": "Качественная",
                "required": False
            }
        ]
    }

    if args.file:
        # Загружаем из файла
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                tender = json.load(f)
            process_tender_with_llm(tender, use_llm=args.llm, model=args.model)
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
    else:
        # Используем пример
        process_tender_with_llm(example_tender, use_llm=args.llm, model=args.model)


if __name__ == "__main__":
    main()