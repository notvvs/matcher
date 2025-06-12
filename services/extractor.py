import os
import re
from collections import defaultdict


class ConfigurableTermExtractor:
    """Экстрактор с конфигурационными файлами"""

    def __init__(self, config_dir="config"):
        self.config_dir = config_dir

        # Загружаем конфигурации
        self.stop_words = self._load_config_file("C:/Users/ruzik/PycharmProjects/work/matcher/config/stopwords.txt")
        self.important_chars = self._load_config_file("C:/Users/ruzik/PycharmProjects/work/matcher/config/important_chars.txt")
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
                    # Пропускаем комментарии и пустые строки
                    if line and not line.startswith('#'):
                        config_set.add(line.lower())

            print(f"📚 Загружен {filename}: {len(config_set)} записей")
            return config_set

        except Exception as e:
            print(f"❌ Ошибка загрузки {filename}: {e}")
            return config_set

    def _load_synonyms(self):
        """Загружаем синонимы"""
        filepath = "C:/Users/ruzik/PycharmProjects/work/matcher/config/synonyms.txt"
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
                        # Каждое слово связываем со всеми синонимами
                        for word in synonyms:
                            if word not in synonyms_dict:
                                synonyms_dict[word] = set()
                            synonyms_dict[word].update(synonyms)
                            synonyms_dict[word].discard(word)  # Убираем само слово

            return synonyms_dict

        except Exception as e:
            print(f"❌ Ошибка загрузки синонимов: {e}")
            return {}

    def is_stop_word(self, word):
        """Проверяем, является ли слово стоп-словом"""
        return word.lower() in self.stop_words

    def is_important_characteristic(self, word):
        """Проверяем, важная ли это характеристика"""
        word_lower = word.lower()
        return any(important in word_lower for important in self.important_chars)

    def get_synonyms(self, word):
        """Получаем синонимы для слова"""
        return list(self.synonyms_dict.get(word.lower(), []))

    def expand_with_synonyms(self, words):
        """Расширяем список слов синонимами"""
        expanded = set(words)

        for word in words:
            synonyms = self.get_synonyms(word)
            expanded.update(synonyms)

        return list(expanded)

    def extract_from_tender(self, tender_item):
        """ГЛАВНАЯ ФУНКЦИЯ: извлечение терминов"""

        tender_name = tender_item.get('name', 'Без названия')
        print(f"🔧 Анализ тендера: {tender_name}")
        print("-" * 50)

        # 1. Извлекаем сырые термины
        raw_terms = self._extract_raw_terms(tender_item)

        # 2. Классифицируем
        classified = self._classify_terms(raw_terms)

        # 3. Расширяем синонимами
        expanded = self._expand_classified_terms(classified)

        # 4. Формируем финальный результат
        result = self._build_final_result(expanded, tender_item)

        return result

    def _extract_raw_terms(self, tender_item):
        """Извлекаем сырые термины"""
        terms = {
            'name_terms': [],
            'char_names': [],
            'char_values': [],
            'all_text': ''
        }

        # Из названия тендера
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

        # Статистика
        print(f"🔍 Сырые термины:")
        print(f"   - Название: {len(terms['name_terms'])}")
        print(f"   - Названия характеристик: {len(terms['char_names'])}")
        print(f"   - Значения характеристик: {len(terms['char_values'])}")

        return terms

    def _clean_and_filter_words(self, text):
        """Очистка и фильтрация слов"""
        if not text:
            return []

        # Убираем пунктуацию и лишние символы
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()

        # Фильтруем
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
        """Классификация терминов по важности"""
        classified = {
            'primary': raw_terms['name_terms'][:3],  # Из названия - самые важные
            'secondary': [],  # Значения характеристик - важные
            'tertiary': []  # Названия характеристик - менее важные
        }

        # Значения характеристик
        for value in raw_terms['char_values']:
            if value not in ['нет', 'да', 'без', 'наличие', 'отсутствие']:
                classified['secondary'].append(value)

        # Важные названия характеристик
        for char_name in raw_terms['char_names']:
            if self.is_important_characteristic(char_name):
                classified['tertiary'].append(char_name)

        print(f"📊 Классификация:")
        print(f"   - Primary (название): {len(classified['primary'])}")
        print(f"   - Secondary (значения): {len(classified['secondary'])}")
        print(f"   - Tertiary (характеристики): {len(classified['tertiary'])}")

        return classified

    def _expand_classified_terms(self, classified):
        """Расширяем каждую группу синонимами"""
        expanded = {}

        for category, terms in classified.items():
            original_count = len(terms)
            expanded[category] = self.expand_with_synonyms(terms)
            new_count = len(expanded[category])
            print(f"📈 {category}: {original_count} → {new_count} (добавлено {new_count - original_count} синонимов)")

        return expanded

    def _build_final_result(self, expanded, tender_item):
        """Формируем финальный результат"""
        result = {
            'search_query': '',
            'boost_terms': {},
            'all_terms': [],
            'debug_info': {}
        }

        # Основной поисковый запрос
        if expanded['primary']:
            result['search_query'] = ' '.join(expanded['primary'][:3])

            # Термины с весами
        weight_mapping = {
            'primary': 3.0,  # Самый высокий вес
            'secondary': 2.0,  # Высокий вес
            'tertiary': 1.0  # Средний вес
        }

        for category, terms in expanded.items():
            base_weight = weight_mapping.get(category, 1.0)
            for i, term in enumerate(terms[:5]):  # Максимум 5 терминов на категорию
                weight = base_weight - (i * 0.1)  # Постепенно снижаем вес
                if weight > 0.5:  # Минимальный вес
                    result['boost_terms'][term] = round(weight, 2)

        # Все термины для дальнейшего использования
        for terms in expanded.values():
            result['all_terms'].extend(terms)
        result['all_terms'] = list(set(result['all_terms']))  # Убираем дубли

        # Отладочная информация
        result['debug_info'] = {
            'tender_name': tender_item.get('name', ''),
            'characteristics_count': len(tender_item.get('characteristics', [])),
            'primary_terms': expanded['primary'][:3],
            'secondary_terms': expanded['secondary'][:5],
            'tertiary_terms': expanded['tertiary'][:3],
            'total_boost_terms': len(result['boost_terms']),
            'total_all_terms': len(result['all_terms'])
        }

        print(f"✅ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
        print(f"   - Поисковый запрос: '{result['search_query']}'")
        print(f"   - Термины с весами: {len(result['boost_terms'])}")
        print(f"   - Всего уникальных терминов: {len(result['all_terms'])}")

        return result


# Тестирование
def test_extractor():
    """Тест экстрактора"""
    extractor = ConfigurableTermExtractor()

    test_tender = {
        "name": "Блоки для записей",
        "characteristics": [
            {"name": "Цвет бумаги", "value": "Пастельный"},
            {"name": "Тип", "value": "С клейким краем"},
            {"name": "Количество листов в блоке", "value": "≥ 100"}
        ]
    }

    result = extractor.extract_from_tender(test_tender)

    print(f"\n🎯 ТЕСТОВЫЙ РЕЗУЛЬТАТ:")
    print(f"Запрос: {result['search_query']}")
    print(f"Boost terms: {result['boost_terms']}")
    print(f"Debug: {result['debug_info']}")


if __name__ == "__main__":
    test_extractor()