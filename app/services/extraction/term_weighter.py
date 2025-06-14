from typing import Dict, List, Set, Tuple
from collections import defaultdict

from app.core.constants import TERM_WEIGHTS, RANGE_INDICATORS
from app.core.synonyms import SynonymsManager


class TermWeighter:
    """Расчет весов для терминов"""

    def __init__(self, synonyms_manager: SynonymsManager):
        self.synonyms_manager = synonyms_manager

    def calculate_weights(
            self,
            tender_data: Dict,
            classified_terms: Dict[str, List[str]],
            original_terms: Set[str]
    ) -> Dict[str, float]:
        """Рассчитывает веса для всех терминов"""

        weights = {}

        # 1. Веса для значений обязательных характеристик
        self._add_required_weights(
            weights,
            tender_data,
            classified_terms.get('secondary', [])
        )

        # 2. Веса для значений опциональных характеристик
        self._add_optional_weights(
            weights,
            tender_data,
            classified_terms.get('secondary', [])
        )

        # 3. Веса для названий важных характеристик
        self._add_char_name_weights(
            weights,
            tender_data,
            classified_terms.get('tertiary', [])
        )

        # 4. Применяем штраф для синонимов
        self._apply_synonym_penalty(weights, original_terms)

        # 5. Фильтруем по минимальному порогу
        return self._filter_by_threshold(weights)

    def _add_required_weights(
            self,
            weights: Dict[str, float],
            tender_data: Dict,
            secondary_terms: List[str]
    ):
        """Добавляет веса для обязательных характеристик"""

        required_values = []
        characteristics = tender_data.get('characteristics', [])

        # Собираем значения обязательных характеристик
        for char in characteristics:
            if char.get('required', False):
                value = char.get('value', '')
                if value and not self._is_range_value(value):
                    required_values.append(value.lower())

        # Назначаем веса
        config = TERM_WEIGHTS['required_values']
        for i, term in enumerate(secondary_terms[:config['count']]):
            if any(term in val for val in required_values):
                weight = config['start'] - (i * config['step'])
                weights[term] = weight

    def _add_optional_weights(
            self,
            weights: Dict[str, float],
            tender_data: Dict,
            secondary_terms: List[str]
    ):
        """Добавляет веса для опциональных характеристик"""

        optional_values = []
        characteristics = tender_data.get('characteristics', [])

        # Собираем значения опциональных характеристик
        for char in characteristics:
            if not char.get('required', False):
                value = char.get('value', '')
                if value and not self._is_range_value(value):
                    optional_values.append(value.lower())

        # Назначаем веса
        config = TERM_WEIGHTS['optional_values']
        for i, term in enumerate(secondary_terms[:config['count']]):
            if term not in weights and any(term in val for val in optional_values):
                weight = config['start'] - (i * config['step'])
                weights[term] = weight

    def _add_char_name_weights(
            self,
            weights: Dict[str, float],
            tender_data: Dict,
            tertiary_terms: List[str]
    ):
        """Добавляет веса для названий характеристик"""

        char_names = []
        characteristics = tender_data.get('characteristics', [])

        # Собираем названия важных характеристик
        for char in characteristics[:3]:  # Берем первые 3
            if char.get('required', False):
                name = char.get('name', '')
                if name:
                    char_names.append(name.lower())

        # Назначаем веса
        config = TERM_WEIGHTS['char_names']
        for i, term in enumerate(tertiary_terms[:config['count']]):
            if term not in weights and any(term in name for name in char_names):
                weight = config['start'] - (i * config['step'])
                weights[term] = weight

    def _apply_synonym_penalty(
            self,
            weights: Dict[str, float],
            original_terms: Set[str]
    ):
        """Применяет штраф для синонимов"""

        penalty = TERM_WEIGHTS['synonym_penalty']

        for term in list(weights.keys()):
            # Если термин не из оригинального текста - это синоним
            if term not in original_terms:
                weights[term] = round(weights[term] * penalty, 2)

    def _filter_by_threshold(
            self,
            weights: Dict[str, float],
            threshold: float = 1.0
    ) -> Dict[str, float]:
        """Фильтрует веса по минимальному порогу"""

        return {
            term: weight
            for term, weight in weights.items()
            if weight >= threshold
        }

    def _is_range_value(self, value: str) -> bool:
        """Проверяет, является ли значение диапазоном"""
        return any(indicator in value for indicator in RANGE_INDICATORS)