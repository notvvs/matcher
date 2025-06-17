from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime


# Модели для входящего тендера
class Price(BaseModel):
    amount: float
    currency: str


class DeliveryInfo(BaseModel):
    deliveryAddress: str
    deliveryTerm: str
    deliveryConditions: str


class PaymentInfo(BaseModel):
    paymentTerm: str
    paymentMethod: str
    paymentConditions: Optional[str] = None


class TenderInfo(BaseModel):
    tenderName: str
    tenderNumber: str
    customerName: str
    description: Optional[str] = None
    purchaseType: str
    financingSource: str
    maxPrice: Price
    deliveryInfo: DeliveryInfo
    paymentInfo: PaymentInfo


class Characteristic(BaseModel):
    id: int
    name: str
    value: str
    unit: Optional[str] = None
    type: str
    required: bool
    changeable: bool
    fillInstruction: str


class TenderItem(BaseModel):
    id: int
    name: str
    okpd2Code: str
    ktruCode: Optional[str] = None
    quantity: float
    unitOfMeasurement: str
    unitPrice: Price
    totalPrice: Price
    characteristics: List[Characteristic]
    additionalRequirements: Optional[str] = None


class GeneralRequirements(BaseModel):
    qualityRequirements: Optional[str] = None
    packagingRequirements: Optional[str] = None
    markingRequirements: Optional[str] = None
    warrantyRequirements: Optional[str] = None
    safetyRequirements: Optional[str] = None
    regulatoryRequirements: Optional[str] = None


class Attachment(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    url: str


class TenderRequest(BaseModel):
    """Модель входящего тендера"""
    tenderInfo: TenderInfo
    items: List[TenderItem]
    generalRequirements: GeneralRequirements
    attachments: List[Attachment]


# Модели для ответа
class ProductAttribute(BaseModel):
    """Атрибут товара"""
    model_config = ConfigDict(extra='allow')

    attr_name: str = Field(..., description="Название атрибута")
    attr_value: Optional[str] = Field(None, description="Значение атрибута")
    attr_value_numeric: Optional[float] = Field(None, description="Числовое значение атрибута")


class ProductMatch(BaseModel):
    """Найденное соответствие товара"""
    id: str
    title: str
    category: str
    brand: Optional[str] = None
    elasticsearch_score: float
    semantic_score: float
    combined_score: float
    attributes: Optional[List[ProductAttribute]] = None


class ItemResult(BaseModel):
    """Результат обработки одного товара из тендера"""
    item_id: int
    item_name: str
    products_found: int
    products: List[ProductMatch]
    processing_time: float
    error: Optional[str] = None


class ProcessingStats(BaseModel):
    """Статистика обработки"""
    total_items: int
    processed_items: int
    total_products_found: int
    total_processing_time: float
    average_time_per_item: float


class TenderResponse(BaseModel):
    """Ответ на обработку тендера"""
    success: bool
    tender_number: str
    tender_name: str
    items_results: List[ItemResult]
    statistics: ProcessingStats
    timestamp: datetime = Field(default_factory=datetime.now)
    errors: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    """Модель ошибки"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheckResponse(BaseModel):
    """Ответ healthcheck"""
    status: str
    elasticsearch: str
    pipeline: str
    timestamp: datetime = Field(default_factory=datetime.now)