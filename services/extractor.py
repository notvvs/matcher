import os
import re
from collections import defaultdict

from config.settings import settings


class ConfigurableTermExtractor:
    """Экстрактор с оптимальной логикой для тендеров"""

    def __init__(self, config_dir=None):
        self.config_dir = config_dir or settings.CONFIG_DIR

        # Загружаем конфигурации
        self.stop_words = self._load_config_file(settings.STOPWORDS_FILE)
        self.important_chars = self._load_config_file(settings.IMPORTANT_CHARS_FILE)
        self.synonyms_dict = self._load_synonyms()

        print(f"✅ Конфигурация загружена:")
        print(f"   - Стоп-слов: {len(self.stop_words)}")
        print(f"   - Важных характеристик: {len(self.important_chars)}")
        print(f"   - Синонимов: {len(self.synonyms_dict)}")

    def _load_config_file(self, filename):
        """Загружаем конфигурационный файл"""
        filepath = os.path.join(self.config_dir, filename)
        config_set = set()

        if not os.path.exists(filepath):
            print(f"⚠️ Файл {filepath} не найден")
            return config_set

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        config_set.add(line.lower())

            print(f"📚 Загружен {filename}: {len(config_set)} записей")
            return config_set

        except Exception as e:
            print(f"❌ Ошибка загрузки {filename}: {e}")
            return config_set

    def _load_synonyms(self):
        """Загружаем синонимы"""
        filepath = os.path.join(self.config_dir, settings.SYNONYMS_FILE)
        synonyms_dict = {}

        if not os.path.exists(filepath):
            print(f"⚠️ Файл синонимов не найден")
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

            return synonyms_dict

        except Exception as e:
            print(f"❌ Ошибка загрузки синонимов: {e}")
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
        print(f"🎯 ОПТИМАЛЬНЫЙ анализ тендера: {tender_name}")
        print("-" * 60)

        # 1. Извлекаем сырые термины
        raw_terms = self._extract_raw_terms(tender_item)

        # 2. Классифицируем
        classified = self._classify_terms(raw_terms)

        # 3. Расширяем синонимами
        expanded = self._expand_classified_terms(classified)

        # 4. ПРИМЕНЯЕМ ОПТИМАЛЬНУЮ ЛОГИКУ ВЕСОВ
        result = self._build_optimal_tender_weights(expanded, tender_item)

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
            print(f"📝 Из названия: {terms['name_terms']}")

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

        print(f"🔍 Сырые термины: название={len(terms['name_terms'])}, "
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

        print(f"📊 Классификация: Primary={len(classified['primary'])}, "
              f"Secondary={len(classified['secondary'])}, Tertiary={len(classified['tertiary'])}")

        return classified

    def _expand_classified_terms(self, classified):
        """Расширяем синонимами"""
        expanded = {}

        for category, terms in classified.items():
            original_count = len(terms)
            expanded[category] = self.expand_with_synonyms(terms)
            new_count = len(expanded[category])
            print(f"📈 {category}: {original_count} → {new_count} (+{new_count - original_count} синонимов)")

        return expanded

    def _build_optimal_tender_weights(self, expanded, tender_item):
        """🎯 ОПТИМАЛЬНАЯ ЛОГИКА ВЕСОВ ДЛЯ ТЕНДЕРОВ"""

        result = {
            'search_query': '',
            'boost_terms': {},
            'must_match_terms': [],
            'all_terms': [],
            'debug_info': {}
        }

        # 1. ОСНОВНОЙ ПОИСКОВЫЙ ЗАПРОС - ОБЯЗАТЕЛЬНЫЙ
        if expanded['primary']:
            result['search_query'] = ' '.join(expanded['primary'][:2])  # Только 2 главных слова
            result['must_match_terms'] = expanded['primary'][:3]

        # 2. АНАЛИЗИРУЕМ ХАРАКТЕРИСТИКИ ТЕНДЕРА
        characteristics = tender_item.get('characteristics', [])
        required_chars = [c for c in characteristics if c.get('required', False)]
        optional_chars = [c for c in characteristics if not c.get('required', False)]

        print(f"📋 Характеристики: обязательных={len(required_chars)}, опциональных={len(optional_chars)}")

        # 3. ОБЯЗАТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ - МАКСИМАЛЬНЫЙ ВЕС
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

        # 4. ОПЦИОНАЛЬНЫЕ ХАРАКТЕРИСТИКИ - ВЫСОКИЙ ВЕС
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

        # 5. НАЗВАНИЯ ВАЖНЫХ ХАРАКТЕРИСТИК - СРЕДНИЙ ВЕС
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

        # 6. СИНОНИМЫ ПОЛУЧАЮТ ПОНИЖЕННЫЙ ВЕС
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
                result['boost_terms'][term] = round(weight * synonym_penalty, 2)
                synonym_count += 1

        # 7. АНТИ-ШУМОВЫЕ МЕХАНИЗМЫ
        original_count = len(result['boost_terms'])
        result['boost_terms'] = {
            term: weight for term, weight in result['boost_terms'].items()
            if weight >= settings.MIN_WEIGHT_THRESHOLD
        }
        removed_count = original_count - len(result['boost_terms'])

        # 8. ВСЕ ТЕРМИНЫ
        for terms in expanded.values():
            result['all_terms'].extend(terms)
        result['all_terms'] = list(set(result['all_terms']))

        # 9. ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
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

        print(f"🎯 ОПТИМАЛЬНЫЙ РЕЗУЛЬТАТ:")
        print(f"   - Основной запрос: '{result['search_query']}'")
        print(f"   - Обязательные термины: {result['must_match_terms']}")
        print(f"   - Boost терминов: {len(result['boost_terms'])}")
        print(f"   - Синонимов с пониженным весом: {synonym_count}")
        print(f"   - Убрано шумовых терминов: {removed_count}")

        return result