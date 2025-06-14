from pathlib import Path
from typing import Set, List, Dict
import logging


class ConfigLoader:
    """Загрузчик конфигурационных файлов"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)

    def load_lines(self, filename: str) -> List[str]:
        """Загружает строки из файла"""
        filepath = self.data_dir / filename

        if not filepath.exists():
            self.logger.warning(f"Файл не найден: {filepath}")
            return []

        lines = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        lines.append(line)

            self.logger.info(f"Загружено {len(lines)} строк из {filename}")
            return lines

        except Exception as e:
            self.logger.error(f"Ошибка загрузки {filename}: {e}")
            return []

    def load_set(self, filename: str, lowercase: bool = True) -> Set[str]:
        """Загружает набор значений из файла"""
        lines = self.load_lines(filename)

        if lowercase:
            return {line.lower() for line in lines}
        else:
            return set(lines)

    def load_synonyms(self, filename: str) -> Dict[str, Set[str]]:
        """Загружает синонимы из файла"""
        lines = self.load_lines(filename)
        synonyms_dict = {}

        for line in lines:
            if ',' not in line:
                continue

            # Разбираем строку синонимов
            synonyms = [s.strip().lower() for s in line.split(',')]

            # Для каждого слова создаем набор его синонимов
            for word in synonyms:
                if word not in synonyms_dict:
                    synonyms_dict[word] = set()
                # Добавляем все синонимы кроме самого слова
                synonyms_dict[word].update(s for s in synonyms if s != word)

        self.logger.info(f"Загружено синонимов для {len(synonyms_dict)} слов")
        return synonyms_dict