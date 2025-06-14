import re
from typing import List

from app.core.stopwords import StopwordsManager
from app.core.constants import MIN_WORD_LENGTH, IGNORE_VALUES


class TextCleaner:
    """Очистка и подготовка текста"""

    def __init__(self, stopwords_manager: StopwordsManager):
        self.stopwords_manager = stopwords_manager

        # Паттерн для очистки текста
        self.clean_pattern = re.compile(r'[^\w\s]')

    def clean_text(self, text: str) -> str:
        """Базовая очистка текста"""
        if not text:
            return ""

        # Удаляем пунктуацию
        text = self.clean_pattern.sub(' ', text)

        # Нормализуем пробелы
        text = ' '.join(text.split())

        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        """Разбивает текст на токены"""
        clean_text = self.clean_text(text)
        return clean_text.split() if clean_text else []

    def filter_words(self, words: List[str]) -> List[str]:
        """Фильтрует слова по критериям"""
        filtered = []

        for word in words:
            # Пропускаем короткие слова
            if len(word) <= MIN_WORD_LENGTH:
                continue

            # Пропускаем числа
            if word.isdigit():
                continue

            # Пропускаем стоп-слова
            if self.stopwords_manager.is_stopword(word):
                continue

            # Пропускаем игнорируемые значения
            if word in IGNORE_VALUES:
                continue

            # Проверяем, что слово содержит буквы
            if not word.isalpha():
                continue

            filtered.append(word)

        return filtered

    def clean_and_filter(self, text: str) -> List[str]:
        """Полная обработка: очистка, токенизация и фильтрация"""
        tokens = self.tokenize(text)
        return self.filter_words(tokens)

    def extract_numbers(self, text: str) -> List[float]:
        """Извлекает числа из текста"""
        # Паттерн для чисел (включая десятичные)
        number_pattern = re.compile(r'\d+(?:[.,]\d+)?')

        numbers = []
        for match in number_pattern.findall(text):
            # Заменяем запятую на точку для float
            number_str = match.replace(',', '.')
            try:
                numbers.append(float(number_str))
            except ValueError:
                continue

        return numbers