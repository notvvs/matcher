from typing import Set, List
from pathlib import Path

from app.core.config_loader import ConfigLoader


class StopwordsManager:
    """Управление стоп-словами"""

    def __init__(self, stopwords: Set[str]):
        self.stopwords = stopwords

    @classmethod
    def from_file(cls, data_dir: Path, filename: str = 'stopwords.txt') -> 'StopwordsManager':
        """Создает менеджер из файла"""
        loader = ConfigLoader(data_dir)
        stopwords = loader.load_set(filename, lowercase=True)
        return cls(stopwords)

    def is_stopword(self, word: str) -> bool:
        """Проверяет, является ли слово стоп-словом"""
        return word.lower() in self.stopwords

    def filter_stopwords(self, words: List[str]) -> List[str]:
        """Фильтрует стоп-слова из списка"""
        return [w for w in words if not self.is_stopword(w)]

    def add_stopword(self, word: str):
        """Добавляет стоп-слово"""
        self.stopwords.add(word.lower())

    def remove_stopword(self, word: str):
        """Удаляет стоп-слово"""
        self.stopwords.discard(word.lower())

    def __len__(self) -> int:
        return len(self.stopwords)

    def __contains__(self, word: str) -> bool:
        return self.is_stopword(word)