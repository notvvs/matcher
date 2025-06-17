# Веса для извлечения терминов
TERM_WEIGHTS = {
    'required_values': {
        'start': 4.0,
        'step': 0.2,
        'count': 5
    },
    'optional_values': {
        'start': 2.5,
        'step': 0.3,
        'count': 3
    },
    'char_names': {
        'start': 1.8,
        'step': 0.2,
        'count': 4
    },
    'synonym_penalty': 0.7  # Множитель для синонимов (30% штраф)
}

# Множители для полей Elasticsearch
ES_FIELD_MULTIPLIERS = {
    'title': 2.5,
    'category': 1.8,
    'brand': 1.5,
    'attr_value': 2.0,
    'attr_name': 1.0
}

# Индикаторы диапазонов значений
RANGE_INDICATORS = ['≥', '≤', '>', '<', 'более', 'менее', 'от', 'до']

# Слова для игнорирования в значениях характеристик
IGNORE_VALUES = ['нет', 'да', 'без', 'наличие', 'отсутствие']

# Минимальная длина слова для обработки
MIN_WORD_LENGTH = 2