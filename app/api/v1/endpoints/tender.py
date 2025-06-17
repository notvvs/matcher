from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging
from datetime import datetime

from app.models.api import (
    TenderRequest,
    TenderResponse,
    ItemResult,
    ProcessingStats,
    ErrorResponse
)
from app.pipeline.llm_factory import create_llm_pipeline
from app.services.filtering.llm_filter import LLMConfig
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# Глобальный пайплайн (создается один раз)
pipeline = None


def get_pipeline():
    """Получить или создать пайплайн"""
    global pipeline
    if pipeline is None:
        logger.info("Инициализация LLM пайплайна...")
        llm_config = LLMConfig(
            model_name="mistral:7b",
            temperature=0.1,
            max_tokens=300,
            timeout=30,
            max_workers=4
        )
        pipeline = create_llm_pipeline(llm_config)
    return pipeline


@router.post("/process", response_model=TenderResponse)
async def process_tender(
        tender_request: TenderRequest,
        background_tasks: BackgroundTasks
):
    """
    Обработка тендера и поиск подходящих товаров

    - **tender_request**: Данные тендера в формате JSON
    - **return**: Результаты поиска товаров для каждой позиции тендера
    """
    start_time = datetime.now()

    try:
        # Получаем пайплайн
        pipeline = get_pipeline()

        # Результаты для каждого товара
        items_results = []
        total_products_found = 0
        errors = []

        # Обрабатываем каждый товар из тендера
        for item in tender_request.items:
            try:
                # Преобразуем в формат для пайплайна
                tender_dict = {
                    "name": item.name,
                    "characteristics": [
                        {
                            "name": char.name,
                            "value": char.value,
                            "type": char.type,
                            "required": char.required
                        }
                        for char in item.characteristics
                    ]
                }

                # Обрабатываем через пайплайн
                logger.info(f"Обработка позиции: {item.name}")
                result = pipeline.process(tender_dict)

                # Преобразуем результаты
                products = []
                if 'final_products' in result:
                    for product in result['final_products']:
                        products.append({
                            "id": product.get('id', ''),
                            "title": product.get('title', ''),
                            "category": product.get('category', ''),
                            "brand": product.get('brand'),
                            "elasticsearch_score": product.get('elasticsearch_score', 0.0),
                            "semantic_score": product.get('llm_score', 0.0),  # Используем LLM скор
                            "combined_score": product.get('combined_score', 0.0),
                            "attributes": product.get('attributes', [])
                        })

                # Добавляем результат для позиции
                items_results.append(ItemResult(
                    item_id=item.id,
                    item_name=item.name,
                    products_found=len(products),
                    products=products,
                    processing_time=result.get('execution_time', 0.0),
                    error=result.get('error')
                ))

                total_products_found += len(products)

            except Exception as e:
                logger.error(f"Ошибка обработки позиции {item.id}: {e}")
                errors.append(f"Позиция {item.id}: {str(e)}")

                # Добавляем результат с ошибкой
                items_results.append(ItemResult(
                    item_id=item.id,
                    item_name=item.name,
                    products_found=0,
                    products=[],
                    processing_time=0.0,
                    error=str(e)
                ))

        # Общее время обработки
        total_time = (datetime.now() - start_time).total_seconds()

        # Статистика
        stats = ProcessingStats(
            total_items=len(tender_request.items),
            processed_items=len([r for r in items_results if r.error is None]),
            total_products_found=total_products_found,
            total_processing_time=total_time,
            average_time_per_item=total_time / len(tender_request.items) if tender_request.items else 0
        )

        # Формируем ответ
        response = TenderResponse(
            success=len(errors) == 0,
            tender_number=tender_request.tenderInfo.tenderNumber,
            tender_name=tender_request.tenderInfo.tenderName,
            items_results=items_results,
            statistics=stats,
            errors=errors if errors else None
        )

        # Логируем результат в фоне
        background_tasks.add_task(
            log_processing_result,
            tender_request.tenderInfo.tenderNumber,
            stats,
            len(errors) == 0
        )

        return response

    except Exception as e:
        logger.error(f"Критическая ошибка обработки тендера: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки тендера: {str(e)}"
        )


@router.post("/process_simple")
async def process_simple_tender(tender_data: dict):
    """
    Упрощенный endpoint для обработки тендера

    Принимает простой формат:
    ```json
    {
        "name": "Блокнот",
        "characteristics": [
            {
                "name": "Вид линовки",
                "value": "Клетка",
                "type": "Качественная",
                "required": true
            }
        ]
    }
    ```
    """
    try:
        # Валидация входных данных
        if 'name' not in tender_data:
            raise HTTPException(status_code=400, detail="Отсутствует поле 'name'")

        if 'characteristics' not in tender_data or not tender_data['characteristics']:
            raise HTTPException(status_code=400, detail="Отсутствуют характеристики")

        # Получаем пайплайн
        pipeline = get_pipeline()

        # Обрабатываем
        logger.info(f"Обработка простого тендера: {tender_data['name']}")
        result = pipeline.process(tender_data)

        # Формируем упрощенный ответ
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])

        products = []
        if 'final_products' in result:
            for product in result['final_products'][:10]:  # Топ-10
                products.append({
                    "id": product.get('id', ''),
                    "title": product.get('title', ''),
                    "category": product.get('category', ''),
                    "brand": product.get('brand'),
                    "combined_score": product.get('combined_score', 0.0),
                    "llm_score": product.get('llm_score', 0.0),
                    "llm_reasoning": product.get('llm_reasoning', ''),
                    "key_attributes": [
                        {
                            "name": attr.get('attr_name'),
                            "value": attr.get('attr_value')
                        }
                        for attr in product.get('attributes', [])[:5]
                        if attr.get('attr_name') and attr.get('attr_value')
                    ]
                })

        return {
            "success": True,
            "tender_name": tender_data['name'],
            "products_found": len(products),
            "products": products,
            "processing_time": result.get('execution_time', 0),
            "statistics": result.get('statistics', {})
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки простого тендера: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки: {str(e)}"
        )


def log_processing_result(tender_number: str, stats: ProcessingStats, success: bool):
    """Логирование результатов обработки в фоне"""
    logger.info(
        f"Тендер {tender_number} обработан. "
        f"Успех: {success}, "
        f"Товаров найдено: {stats.total_products_found}, "
        f"Время: {stats.total_processing_time:.2f}с"
    )