from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.v1.router import api_router
from app.models.api import HealthCheckResponse
from app.services.search.elasticsearch_client import ElasticsearchClient
from app.services.search.query_builder import ElasticsearchQueryBuilder
from app.utils.logger import setup_logger

# Настройка логирования
logger = setup_logger(__name__)

# Глобальные ресурсы
resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Запуск приложения...")

    # Инициализация Elasticsearch клиента
    try:
        query_builder = ElasticsearchQueryBuilder()
        es_client = ElasticsearchClient(query_builder)
        resources['es_client'] = es_client
        logger.info("Elasticsearch клиент инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации Elasticsearch: {e}")

    yield

    # Shutdown
    logger.info("Остановка приложения...")


# Создание приложения
app = FastAPI(
    title="Tender Products Matcher API",
    description="API для поиска товаров по требованиям тендера с использованием LLM",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Корневой endpoint"""
    return {
        "message": "Tender Products Matcher API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "api_v1": "/api/v1",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Проверка состояния сервиса"""

    # Проверка Elasticsearch
    es_status = "error"
    if 'es_client' in resources:
        try:
            if resources['es_client'].ping():
                es_status = "ok"
        except:
            pass

    # Проверка пайплайна
    pipeline_status = "ok"  # Пайплайн создается лениво

    return HealthCheckResponse(
        status="ok" if es_status == "ok" else "degraded",
        elasticsearch=es_status,
        pipeline=pipeline_status
    )


# Подключение API роутеров
app.include_router(
    api_router,
    prefix="/api/v1"
)


# Обработчик ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="Внутренняя ошибка сервера"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )