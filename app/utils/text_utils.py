"""
Вспомогательные функции для работы с текстом
"""

import re
from typing import List, Tuple, Optional
import unicodedata


def normalize_text(text: str) -> str:
    """Нормализация текста (удаление диакритики, приведение к нижнему регистру)"""

    if not text:
        return ""

    # Нормализуем Unicode
    text = unicodedata.normalize('NFKD', text)

    # Удаляем диакритические знаки
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # Приводим к нижнему регистру
    text = text.lower()

    # Заменяем множественные пробелы на одинарные
    text = ' '.join(text.split())

    return text


def extract_numbers_with_units(text: str) -> List[Tuple[float, Optional[str]]]:
    """
    Извлекает числа с единицами измерения из текста

    Returns:
        Список кортежей (число, единица_измерения)
    """

    # Паттерн для чисел с опциональными единицами измерения
    pattern = re.compile(
        r'(\d+(?:[.,]\d+)?)\s*([а-яА-Яa-zA-Z]+)?'
    )

    results = []

    for match in pattern.finditer(text):
        number_str = match.group(1).replace(',', '.')
        unit = match.group(2)

        try:
            number = float(number_str)
            results.append((number, unit))
        except ValueError:
            continue

    return results


def remove_extra_whitespace(text: str) -> str:
    """Удаляет лишние пробелы и переносы строк"""

    if not text:
        return ""

    # Заменяем переносы строк на пробелы
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text)

    # Убираем пробелы в начале и конце
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Обрезает текст до максимальной длины с добавлением суффикса"""

    if not text or len(text) <= max_length:
        return text

    # Обрезаем с учетом длины суффикса
    truncated_length = max_length - len(suffix)

    if truncated_length <= 0:
        return suffix

    # Пытаемся обрезать по границе слова
    truncated = text[:truncated_length]
    last_space = truncated.rfind(' ')

    if last_space > truncated_length * 0.8:  # Если пробел не слишком далеко
        truncated = truncated[:last_space]

    return truncated + suffix


def is_cyrillic(text: str) -> bool:
    """Проверяет, содержит ли текст кириллицу"""

    if not text:
        return False

    cyrillic_pattern = re.compile('[а-яА-ЯёЁ]')
    return bool(cyrillic_pattern.search(text))


def is_latin(text: str) -> bool:
    """Проверяет, содержит ли текст латиницу"""

    if not text:
        return False

    latin_pattern = re.compile('[a-zA-Z]')
    return bool(latin_pattern.search(text))


def split_camel_case(text: str) -> List[str]:
    """Разбивает CamelCase текст на слова"""

    # Добавляем пробелы перед заглавными буквами
    result = re.sub(r'([A-Z])', r' \1', text)

    # Разбиваем на слова и фильтруем пустые
    words = [w.strip() for w in result.split() if w.strip()]

    return words


def levenshtein_distance(s1: str, s2: str) -> int:
    """Вычисляет расстояние Левенштейна между двумя строками"""

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            # Вычисляем стоимость операций
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """
    Вычисляет коэффициент схожести двух строк (0.0 - 1.0)
    """

    if not s1 or not s2:
        return 0.0

    # Нормализуем строки
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)

    # Если строки идентичны
    if s1 == s2:
        return 1.0

    # Вычисляем расстояние Левенштейна
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))

    # Преобразуем в коэффициент схожести
    return 1.0 - (distance / max_len)