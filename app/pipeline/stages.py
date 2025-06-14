"""
Определение этапов обработки тендера
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional


class StageType(Enum):
    """Типы этапов обработки"""
    EXTRACTION = "extraction"
    SEARCH = "search"
    FILTERING = "filtering"


@dataclass
class StageResult:
    """Результат выполнения этапа"""

    stage_type: StageType
    success: bool
    execution_time: float
    data: Dict[str, Any]
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        """Проверка наличия ошибки"""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        result = {
            'stage_type': self.stage_type.value,
            'success': self.success,
            'execution_time': self.execution_time,
            'data': self.data
        }

        if self.error:
            result['error'] = self.error

        return result


class PipelineStages:
    """Определения этапов пайплайна"""

    # Названия этапов
    EXTRACTION = "Извлечение терминов"
    ELASTICSEARCH = "Поиск в Elasticsearch"
    SEMANTIC_FILTER = "Семантическая фильтрация"

    # Описания этапов
    DESCRIPTIONS = {
        StageType.EXTRACTION: "Анализ тендера и извлечение ключевых терминов с весами",
        StageType.SEARCH: "Полнотекстовый поиск товаров в индексе Elasticsearch",
        StageType.FILTERING: "Фильтрация по семантической близости и комбинирование скоров"
    }

    @classmethod
    def get_description(cls, stage_type: StageType) -> str:
        """Получить описание этапа"""
        return cls.DESCRIPTIONS.get(stage_type, "Неизвестный этап")