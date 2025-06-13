import os
import re

from app.config.settings import settings
from app.utils.logger import setup_logger


class ConfigurableTermExtractor:
    """Экстрактор с оптимальной логикой для тендеров"""

    def __init__(self, config_dir=None):
        self.logger = setup_logger(__name__)
        self.config_dir = config_dir or settings.CONFIG_DIR

        # Загружаем конфигурации
        self.stop_words = self._load_config_file(settings.STOPWORDS_FILE)
        self.important_chars = self._load_config_file(settings.IMPORTANT_CHARS_FILE)
        self.synonyms_dict = self._load_synonyms()

        self.logger.info(f"Конфигурация загружена: стоп-слов={len(self.stop_words)}, "
                         f"важных характеристик={len(self.important_chars)}, "
                         f"синонимов={len(self.synonyms_dict)}")

    def _load_config_file(self, filename):
        """Загружаем конфигурационный файл"""
        filepath = os.path.join(self.config_dir, filename)
        config_set = set()

        if not os.path.exists(filepath):
            self.logger.warning(f"Файл {filepath} не найден")
            return config_set

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        config_set.add(line.lower())

            self.logger.debug(f"Загружен {filename}: {len(config_set)} записей")
            return config_set

        except Exception as e:
            self.logger.error(f"Ошибка загрузки {filename}: {e}")
            return config_set

    def _load_synonyms(self):
        """Загружаем синонимы"""
        filepath = os.path.join(self.config_dir, settings.SYNONYMS_FILE)
        synonyms_dict = {}

        if not os.path.exists(filepath):
            self.logger.warning(f"Файл синонимов не найден: {filepath}")
            return synonyms_dict

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and ',' in line:
                        synonyms = [s.strip().lower() for s in line.split(',')]
                        for word in synonyms:
                            if word not in synonyms_dict:
                                synonyms_dict[word] = set()
                            synonyms_dict[word].update(synonyms)
                            synonyms_dict[word].discard(word)

            self.logger.debug(f"Загружено синонимов: {len(synonyms_dict)}")
            return synonyms_dict

        except Exception as e:
            self.logger.error(f"Ошибка загрузки синонимов: {e}")
            return {}

    def is_stop_word(self, word):
        """Проверяем стоп-слово"""
        return word.lower() in self.stop_words

    def is_important_characteristic(self, word):
        """Проверяем важную характеристику"""
        word_lower = word.lower()
        return any(important in word_lower for important in self.important_chars)

    def get_synonyms(self, word):
        """Получаем синонимы"""
        return list(self.synonyms_dict.get(word.lower(), []))

    def expand_with_synonyms(self, words):
        """Расширяем синонимами"""
        expanded = set(words)
        for word in words:
            synonyms = self.get_synonyms(word)
            expanded.update(synonyms)
        return list(expanded)

    def extract_from_tender(self, tender_item):
        """ГЛАВНАЯ ФУНКЦИЯ с оптимальной логикой"""

        tender_name = tender_item.get('name', 'Без названия')
        self.logger.info(f"=== Начало анализа тендера: {tender_name} ===")

        # 1. Извлекаем сырые термины
        self.logger.debug("Шаг 1: Извлечение сырых терминов")
        raw_terms = self._extract_raw_terms(tender_item)

        # 2. Классифицируем
        self.logger.debug("Шаг 2: Классификация терминов")
        classified = self._classify_terms(raw_terms)

        # 3. Расширяем синонимами
        self.logger.debug("Шаг 3: Расширение синонимами")
        expanded = self._expand_classified_terms(classified)

        # 4. ПРИМЕНЯЕМ ОПТИМАЛЬНУЮ ЛОГИКУ ВЕСОВ
        self.logger.debug("Шаг 4: Применение оптимальной логики весов")
        result = self._build_optimal_tender_weights(expanded, tender_item, raw_terms)

        self.logger.info(f"=== Завершен анализ тендера: {tender_name} ===")

        return result

    def _extract_raw_terms(self, tender_item):
        """Извлекаем сырые термины"""
        terms = {
            'name_terms': [],
            'char_names': [],
            'char_values': [],
            'all_text': ''
        }

        # Из названия
        tender_name = tender_item.get('name', '')
        if tender_name:
            terms['name_terms'] = self._clean_and_filter_words(tender_name)
            terms['all_text'] += f" {tender_name}"
            self.logger.debug(f"Из названия извлечено терминов: {len(terms['name_terms'])} - {terms['name_terms']}")

        # Из характеристик
        if 'characteristics' in tender_item:
            for char in tender_item['characteristics']:
                char_name = char.get('name', '')
                char_value = char.get('value', '')

                terms['all_text'] += f" {char_name} {char_value}"

                if char_name:
                    char_name_words = self._clean_and_filter_words(char_name)
                    terms['char_names'].extend(char_name_words)

                if char_value and not self._is_range_value(char_value):
                    char_value_words = self._clean_and_filter_words(str(char_value))
                    terms['char_values'].extend(char_value_words)

        self.logger.debug(f"Итого сырых терминов: название={len(terms['name_terms'])}, "
                          f"хар-ки={len(terms['char_names'])}, значения={len(terms['char_values'])}")

        return terms

    def _clean_and_filter_words(self, text):
        """Очистка и фильтрация"""
        if not text:
            return []

        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()

        filtered = []
        for word in words:
            if (len(word) > 2 and
                    not self.is_stop_word(word) and
                    word.isalpha() and
                    not word.isdigit()):
                filtered.append(word)

        return filtered

    def _is_range_value(self, value):
        """Проверяем диапазоны"""
        value_str = str(value)
        range_indicators = ['≥', '≤', '>', '<', 'более', 'менее', 'от', 'до']
        return any(indicator in value_str for indicator in range_indicators)

    def _classify_terms(self, raw_terms):
        """Классификация"""
        classified = {
            'primary': raw_terms['name_terms'][:3],
            'secondary': [],
            'tertiary': []
        }

        for value in raw_terms['char_values']:
            if value not in ['нет', 'да', 'без', 'наличие', 'отсутствие']:
                classified['secondary'].append(value)

        for char_name in raw_terms['char_names']:
            if self.is_important_characteristic(char_name):
                classified['tertiary'].append(char_name)

        self.logger.debug(f"Классификация: Primary={len(classified['primary'])}, "
                          f"Secondary={len(classified['secondary'])}, Tertiary={len(classified['tertiary'])}")

        return classified

    def _expand_classified_terms(self, classified):
        """Расширяем синонимами"""
        expanded = {}

        for category, terms in classified.items():
            original_count = len(terms)
            expanded[category] = self.expand_with_synonyms(terms)
            new_count = len(expanded[category])

            if new_count > original_count:
                self.logger.debug(f"Расширение {category}: {original_count} → {new_count} "
                                  f"(+{new_count - original_count} синонимов)")

        return expanded

    def _build_optimal_tender_weights(self, expanded, tender_item, raw_terms):
        """Логика весов"""

        result = {
            'search_query': '',
            'boost_terms': {},
            'must_match_terms': [],
            'all_terms': [],
            'debug_info': {}
        }

        # 1. Основной запрос
        if expanded['primary']:
            # Берем оригинальное название без синонимов для основного запроса
            original_primary = raw_terms['name_terms'][:2]  # Оригинальные термины без синонимов
            result['search_query'] = ' '.join(original_primary)
            # В must_match_terms включаем ВСЕ термины с синонимами
            result['must_match_terms'] = expanded['primary']
            self.logger.debug(f"Основной запрос: '{result['search_query']}'")
            self.logger.debug(f"Обязательные термины (с синонимами): {result['must_match_terms']}")

        # 2. Анализ характеристик
        characteristics = tender_item.get('characteristics', [])
        required_chars = [c for c in characteristics if c.get('required', False)]
        optional_chars = [c for c in characteristics if not c.get('required', False)]

        self.logger.debug(f"Характеристики: обязательных={len(required_chars)}, "
                          f"опциональных={len(optional_chars)}")

        # 3. Обязательные характеристики - макс вес
        required_values = []
        for char in required_chars:
            char_value = char.get('value', '')
            if char_value and not self._is_range_value(char_value):
                clean_values = self._clean_and_filter_words(str(char_value))
                required_values.extend(clean_values)

        # Значения ОБЯЗАТЕЛЬНЫХ характеристик получают максимальные веса
        weights_config = settings.WEIGHTS['required_values']
        for i, term in enumerate(required_values[:weights_config['count']]):
            if term in expanded['secondary']:
                weight = weights_config['start'] - (i * weights_config['step'])
                result['boost_terms'][term] = weight
                self.logger.debug(f"Обязательная хар-ка: '{term}' = {weight}")

        # 4. Опциональным характеристикам - высокий вес
        optional_values = []
        for char in optional_chars:
            char_value = char.get('value', '')
            if char_value and not self._is_range_value(char_value):
                clean_values = self._clean_and_filter_words(str(char_value))
                optional_values.extend(clean_values)

        weights_config = settings.WEIGHTS['optional_values']
        for i, term in enumerate(optional_values[:weights_config['count']]):
            if term in expanded['secondary'] and term not in result['boost_terms']:
                weight = weights_config['start'] - (i * weights_config['step'])
                result['boost_terms'][term] = weight
                self.logger.debug(f"Опциональная хар-ка: '{term}' = {weight}")

        # 5. Для важных характеристик - средний вес
        important_char_names = []
        for char in required_chars[:3]:
            char_name = char.get('name', '')
            if char_name:
                clean_names = self._clean_and_filter_words(char_name)
                important_char_names.extend(clean_names)

        weights_config = settings.WEIGHTS['char_names']
        for i, term in enumerate(important_char_names[:weights_config['count']]):
            if term in expanded['tertiary'] and term not in result['boost_terms']:
                weight = weights_config['start'] - (i * weights_config['step'])
                result['boost_terms'][term] = weight
                self.logger.debug(f"Название хар-ки: '{term}' = {weight}")

        # 6. Синонимам понижаем вес
        original_terms = set()
        original_terms.update(self._clean_and_filter_words(tender_item.get('name', '')))
        for char in characteristics:
            original_terms.update(self._clean_and_filter_words(char.get('name', '')))
            original_terms.update(self._clean_and_filter_words(str(char.get('value', ''))))

        # Снижаем вес синонимов
        synonym_count = 0
        synonym_penalty = settings.WEIGHTS['synonym_penalty']
        for term, weight in list(result['boost_terms'].items()):
            if term not in original_terms:  # Это синоним
                old_weight = weight
                result['boost_terms'][term] = round(weight * synonym_penalty, 2)
                synonym_count += 1
                self.logger.debug(f"Синоним '{term}': {old_weight} → {result['boost_terms'][term]} (-30%)")

        # 7. Анти-шумовые механизмы
        original_count = len(result['boost_terms'])
        result['boost_terms'] = {
            term: weight for term, weight in result['boost_terms'].items()
            if weight >= settings.MIN_WEIGHT_THRESHOLD
        }
        removed_count = original_count - len(result['boost_terms'])

        if removed_count > 0:
            self.logger.debug(f"Удалено терминов с низким весом: {removed_count}")

        # 8. Все термины
        for terms in expanded.values():
            result['all_terms'].extend(terms)
        result['all_terms'] = list(set(result['all_terms']))

        # 9. Отладочная инфа
        result['debug_info'] = {
            'tender_name': tender_item.get('name', ''),
            'required_characteristics': len(required_chars),
            'optional_characteristics': len(optional_chars),
            'must_match_terms': result['must_match_terms'],
            'boost_terms_count': len(result['boost_terms']),
            'synonyms_penalized': synonym_count,
            'low_weight_removed': removed_count,
            'weight_ranges': {
                'required_values': '4.0 → 3.2',
                'optional_values': '2.5 → 1.9',
                'char_names': '1.8 → 1.2',
                'synonym_penalty': '-30%'
            }
        }

        self.logger.info(f"Результат извлечения: основной запрос='{result['search_query']}', "
                         f"boost терминов={len(result['boost_terms'])}, "
                         f"синонимов с пониженным весом={synonym_count}, "
                         f"убрано шумовых терминов={removed_count}")

        return result