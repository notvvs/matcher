from fastapi import APIRouter

from app.api.v1.endpoints import tender

api_router = APIRouter()

# Подключаем роутеры
api_router.include_router(
    tender.router,
    prefix="/tender",
    tags=["tender"]
)