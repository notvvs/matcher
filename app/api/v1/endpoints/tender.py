from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime

from app.models.api import (
    TenderRequest,
    TenderResponse,
    ItemResult,
    ProcessingStats
)
from app.core.settings import settings
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

        # Все настройки берутся из settings
        llm_config = LLMConfig(
            model_name=settings.LLM_MODEL,
            api_url=settings.LLM_API_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            max_workers=settings.LLM_MAX_WORKERS,
            batch_size=settings.LLM_BATCH_SIZE
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
                            "semantic_score": product.get('llm_score', 0.0),
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

def log_processing_result(tender_number: str, stats: ProcessingStats, success: bool):
    """Логирование результатов обработки в фоне"""
    logger.info(
        f"Тендер {tender_number} обработан. "
        f"Успех: {success}, "
        f"Товаров найдено: {stats.total_products_found}, "
        f"Время: {stats.total_processing_time:.2f}с"
    )