from typing import Dict, List, Set
import logging

from app.core.settings import settings
from app.core.stopwords import StopwordsManager
from app.core.synonyms import SynonymsManager
from app.core.constants import IGNORE_VALUES
from app.services.extraction.term_weighter import TermWeighter
from app.services.extraction.text_cleaner import TextCleaner


class TermExtractor:
    """Извлечение и обработка терминов из тендера"""

    def __init__(
            self,
            text_cleaner: TextCleaner,
            term_weighter: TermWeighter,
            synonyms_manager: SynonymsManager,
            important_chars: Set[str]
    ):
        self.text_cleaner = text_cleaner
        self.term_weighter = term_weighter
        self.synonyms_manager = synonyms_manager
        self.important_chars = important_chars
        self.logger = logging.getLogger(__name__)

    def extract_from_tender(self, tender: Dict) -> Dict:
        """Главный метод извлечения терминов"""

        # 1. Извлекаем сырые термины
        raw_terms = self._extract_raw_terms(tender)

        # 2. Классифицируем термины
        classified = self._classify_terms(raw_terms)

        # 3. Расширяем синонимами
        expanded = self._expand_with_synonyms(classified)

        # 4. Сохраняем оригинальные термины для определения синонимов
        original_terms = self._get_original_terms(tender)

        # 5. Рассчитываем веса
        boost_terms = self.term_weighter.calculate_weights(
            tender,
            expanded,
            original_terms
        )

        # 6. Формируем результат
        result = {
            'search_query': self._build_search_query(raw_terms),
            'boost_terms': boost_terms,
            'must_match_terms': expanded.get('primary', []),
            'all_terms': self._get_all_terms(expanded),
            'debug_info': {
                'tender_name': tender.get('name', ''),
                'required_characteristics': len([
                    c for c in tender.get('characteristics', [])
                    if c.get('required', False)
                ]),
                'optional_characteristics': len([
                    c for c in tender.get('characteristics', [])
                    if not c.get('required', False)
                ]),
                'boost_terms_count': len(boost_terms)
            }
        }

        return result

    def _extract_raw_terms(self, tender: Dict) -> Dict[str, List[str]]:
        """Извлекает сырые термины из тендера"""

        terms = {
            'name_terms': [],
            'char_names': [],
            'char_values': []
        }

        # Из названия тендера
        tender_name = tender.get('name', '')
        if tender_name:
            terms['name_terms'] = self.text_cleaner.clean_and_filter(tender_name)

        # Из характеристик
        for char in tender.get('characteristics', []):
            # Названия характеристик
            char_name = char.get('name', '')
            if char_name:
                name_words = self.text_cleaner.clean_and_filter(char_name)
                terms['char_names'].extend(name_words)

            # Значения характеристик
            char_value = char.get('value', '')
            if char_value and not self._is_range_value(char_value):
                value_words = self.text_cleaner.clean_and_filter(str(char_value))
                terms['char_values'].extend(value_words)

        return terms

    def _classify_terms(self, raw_terms: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Классифицирует термины по важности"""

        classified = {
            'primary': raw_terms['name_terms'][:3],  # Первые 3 слова из названия
            'secondary': [],  # Значения характеристик
            'tertiary': []  # Названия важных характеристик
        }

        # Значения характеристик (кроме игнорируемых)
        for value in raw_terms['char_values']:
            if value not in IGNORE_VALUES:
                classified['secondary'].append(value)

        # Важные названия характеристик
        for char_name in raw_terms['char_names']:
            if self._is_important_characteristic(char_name):
                classified['tertiary'].append(char_name)

        return classified

    def _expand_with_synonyms(self, classified: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Расширяет термины синонимами"""

        expanded = {}

        for category, terms in classified.items():
            expanded[category] = self.synonyms_manager.expand_with_synonyms(terms)

        return expanded

    def _is_important_characteristic(self, word: str) -> bool:
        """Проверяет, является ли слово важной характеристикой"""

        word_lower = word.lower()
        return any(important in word_lower for important in self.important_chars)

    def _is_range_value(self, value: str) -> bool:
        """Проверяет, является ли значение диапазоном"""

        from app.core.constants import RANGE_INDICATORS
        return any(indicator in str(value) for indicator in RANGE_INDICATORS)

    def _build_search_query(self, raw_terms: Dict[str, List[str]]) -> str:
        """Формирует основной поисковый запрос"""

        # Берем первые 2 слова из названия
        primary_terms = raw_terms['name_terms'][:2]
        return ' '.join(primary_terms)

    def _get_original_terms(self, tender: Dict) -> Set[str]:
        """Получает все оригинальные термины из тендера"""

        original = set()

        # Из названия
        name = tender.get('name', '')
        if name:
            original.update(self.text_cleaner.clean_and_filter(name))

        # Из характеристик
        for char in tender.get('characteristics', []):
            char_name = char.get('name', '')
            char_value = char.get('value', '')

            if char_name:
                original.update(self.text_cleaner.clean_and_filter(char_name))
            if char_value:
                original.update(self.text_cleaner.clean_and_filter(str(char_value)))

        return original

    def _get_all_terms(self, expanded: Dict[str, List[str]]) -> List[str]:
        """Собирает все уникальные термины"""

        all_terms = set()

        for terms in expanded.values():
            all_terms.update(terms)

        return list(all_terms)