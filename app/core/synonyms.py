from typing import Set, List, Dict
from pathlib import Path

from app.core.config_loader import ConfigLoader


class SynonymsManager:
    """Управление синонимами"""

    def __init__(self, synonyms_dict: Dict[str, Set[str]]):
        self.synonyms_dict = synonyms_dict

    @classmethod
    def from_file(cls, data_dir: Path, filename: str = 'synonyms.txt') -> 'SynonymsManager':
        """Создает менеджер из файла"""
        loader = ConfigLoader(data_dir)
        synonyms_dict = loader.load_synonyms(filename)
        return cls(synonyms_dict)

    def get_synonyms(self, word: str) -> List[str]:
        """Получает список синонимов для слова"""
        return list(self.synonyms_dict.get(word.lower(), []))

    def expand_with_synonyms(self, words: List[str]) -> List[str]:
        """Расширяет список слов их синонимами"""
        expanded = set(words)

        for word in words:
            synonyms = self.get_synonyms(word)
            expanded.update(synonyms)

        return list(expanded)

    def are_synonyms(self, word1: str, word2: str) -> bool:
        """Проверяет, являются ли слова синонимами"""
        word1_lower = word1.lower()
        word2_lower = word2.lower()

        # Проверяем в обе стороны
        return (word2_lower in self.synonyms_dict.get(word1_lower, set()) or
                word1_lower in self.synonyms_dict.get(word2_lower, set()))

    def add_synonym_group(self, words: List[str]):
        """Добавляет группу синонимов"""
        words_lower = [w.lower() for w in words]

        for word in words_lower:
            if word not in self.synonyms_dict:
                self.synonyms_dict[word] = set()
            # Добавляем все слова кроме самого слова
            self.synonyms_dict[word].update(w for w in words_lower if w != word)

    def __len__(self) -> int:
        return len(self.synonyms_dict)