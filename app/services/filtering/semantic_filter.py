import re
import time
import logging
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.settings import settings


class SemanticFilter:
    """Фильтрация товаров по семантической близости"""

    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name or settings.EMBEDDINGS_MODEL

        # Загружаем модель
        self.logger.info(f"Загрузка модели {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = 512

        # Паттерн для очистки значений
        self.clean_pattern = re.compile(r'[≥≤<>]=?')

    def filter_by_similarity(
            self,
            tender: Dict,
            products: List[Dict],
            threshold: Optional[float] = None,
            top_k: int = -1
    ) -> List[Dict]:
        """Фильтрует товары по семантической близости к тендеру"""

        if not products:
            self.logger.warning("Нет товаров для фильтрации")
            return []

        # Используем порог из настроек если не указан
        if threshold is None:
            threshold = settings.SEMANTIC_THRESHOLD

        start_time = time.time()

        # Создаем текстовые представления
        tender_text = self._create_tender_text(tender)
        product_texts = []
        valid_products = []

        for product in products:
            try:
                text = self._create_product_text(product)
                if text:
                    product_texts.append(text)
                    valid_products.append(product)
            except Exception as e:
                self.logger.error(f"Ошибка обработки товара: {e}")
                continue

        if not product_texts:
            self.logger.warning("Не удалось создать тексты для товаров")
            return []

        # Вычисляем эмбеддинги
        self.logger.info(f"Вычисление эмбеддингов для {len(product_texts)} товаров...")

        with torch.no_grad():
            tender_embedding = self.model.encode([tender_text], convert_to_numpy=True)

            # Батчевая обработка для эффективности
            batch_size = settings.SEMANTIC_BATCH_SIZE
            product_embeddings = []
            total_batches = (len(product_texts) + batch_size - 1) // batch_size

            for batch_num, i in enumerate(range(0, len(product_texts), batch_size)):
                batch = product_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                product_embeddings.append(batch_embeddings)

                # Логируем прогресс
                if batch_num % 5 == 0 or batch_num == total_batches - 1:
                    progress = min(i + batch_size, len(product_texts))
                    percent = (progress / len(product_texts)) * 100
                    self.logger.info(f"Обработано: {progress}/{len(product_texts)} ({percent:.0f}%)")

            product_embeddings = np.vstack(product_embeddings)

        # Вычисляем схожесть
        similarities = cosine_similarity(tender_embedding, product_embeddings)[0]

        # Фильтруем по порогу
        filtered_products = []

        for product, similarity in zip(valid_products, similarities):
            if similarity >= threshold:
                product['semantic_score'] = float(similarity)
                filtered_products.append(product)

        # Сортируем по убыванию схожести
        filtered_products.sort(key=lambda x: x['semantic_score'], reverse=True)

        # Ограничиваем количество если указано
        if top_k > 0:
            filtered_products = filtered_products[:top_k]

        elapsed_time = time.time() - start_time

        self.logger.info(
            f"Семантическая фильтрация: {len(filtered_products)} из {len(products)} "
            f"за {elapsed_time:.2f}с (порог: {threshold})"
        )

        return filtered_products

    def _create_tender_text(self, tender: Dict) -> str:
        """Создает текстовое представление тендера"""

        parts = []

        # Название тендера
        name = tender.get('name', '').strip()
        if name:
            parts.append(f"Товар {name}")

        # Обязательные характеристики
        for char in tender.get('characteristics', []):
            if char.get('required', False):
                char_name = char.get('name', '').strip()
                char_value = self._clean_value(char.get('value', ''))

                if char_name and char_value:
                    parts.append(f"{char_name} {char_value}")

        return ". ".join(parts)

    def _create_product_text(self, product: Dict) -> str:
        """Создает текстовое представление товара"""

        parts = []

        # Название с категорией
        title = product.get('title', '').strip()
        category = product.get('category', '').strip()

        if title:
            if category:
                parts.append(f"{category}: {title}")
            else:
                parts.append(title)

        # Бренд
        brand = product.get('brand', '').strip()
        if brand:
            parts.append(f"Производитель {brand}")

        # Ключевые атрибуты (максимум 5)
        attributes = product.get('attributes', [])
        key_attrs = []

        for attr in attributes[:5]:
            attr_name = attr.get('attr_name', '').strip()
            attr_value = self._clean_value(attr.get('attr_value', ''))

            if attr_name and attr_value and len(attr_value) < 50:
                key_attrs.append(f"{attr_name} {attr_value}")

        if key_attrs:
            parts.append(". ".join(key_attrs))

        return ". ".join(parts)

    def _clean_value(self, value: str) -> str:
        """Очищает значение от операторов сравнения"""

        if not value:
            return ""

        # Убираем операторы
        value = self.clean_pattern.sub('', str(value)).strip()

        # Нормализуем пробелы
        value = ' '.join(value.split())

        return value

    def compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Вычисляет эмбеддинги для списка текстов"""

        if not texts:
            return np.array([])

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=settings.SEMANTIC_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False
            )

        return embeddings