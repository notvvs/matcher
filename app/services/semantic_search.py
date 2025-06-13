import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import re

from app.config.settings import settings


class SemanticSearchService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDINGS_MODEL)
        self.model.max_seq_length = 512

    def create_tender_text(self, tender: Dict[str, Any]) -> str:
        """Создаем текст тендера для эмбеддинга."""
        parts = []

        # Название тендера
        name = tender.get('name', '').strip()
        if name:
            parts.append(f"Товар {name}")

        # Добавляем обязательные характеристики
        for char in tender.get('characteristics', []):
            if char.get('required', False):
                char_name = char.get('name', '').strip()
                char_value = self._clean_value(char.get('value', ''))

                if char_name and char_value:
                    parts.append(f"{char_name} {char_value}")

        return ". ".join(parts)

    def create_product_text(self, product: Dict[str, Any]) -> str:
        """Создаем текст товара с учетом категории."""
        parts = []

        # Название с категорией для контекста
        title = product.get('title', '').strip()
        category = product.get('category', '').strip()

        if title:
            if category:
                parts.append(f"{category}: {title}")
            else:
                parts.append(title)

        # Добавляем бренд
        brand = product.get('brand', '').strip()
        if brand:
            parts.append(f"Производитель {brand}")

        # Только ключевые атрибуты
        if product.get('attributes'):
            key_attrs = []
            for attr in product['attributes'][:5]:  # Максимум 5 атрибутов
                attr_name = attr.get('attr_name', '').strip()
                attr_value = self._clean_value(attr.get('attr_value', ''))

                if attr_name and attr_value and len(attr_value) < 50:
                    key_attrs.append(f"{attr_name} {attr_value}")

            if key_attrs:
                parts.append(". ".join(key_attrs))

        return ". ".join(parts)

    def _clean_value(self, value: str) -> str:
        """Очистка значения от лишних символов."""
        if not value:
            return ""
        # Убираем операторы сравнения
        value = re.sub(r'[≥≤<>]=?', '', value).strip()
        # Убираем лишние пробелы
        value = ' '.join(value.split())
        return value

    def filter_by_similarity(self, tender: Dict[str, Any],
                           products: List[Dict[str, Any]],
                           threshold: float = 0.35) -> Dict[str, Any]:
        """Фильтрация по семантической близости."""
        if not products:
            return {'products': [], 'stats': {}}

        # Создаем текст тендера
        tender_text = self.create_tender_text(tender)

        # Создаем тексты товаров
        product_texts = []
        valid_products = []

        for product in products:
            try:
                text = self.create_product_text(product)
                if text:
                    product_texts.append(text)
                    valid_products.append(product)
            except Exception as e:
                print(f"Ошибка обработки товара: {e}")
                continue

        if not product_texts:
            return {'products': [], 'stats': {}}

        # Вычисляем эмбеддинги
        print(f"🧠 Вычисление эмбеддингов для {len(product_texts)} товаров...")

        with torch.no_grad():
            tender_embedding = self.model.encode([tender_text], convert_to_numpy=True)

            # Батчевая обработка
            batch_size = 64
            product_embeddings = []

            for i in range(0, len(product_texts), batch_size):
                batch_texts = product_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                product_embeddings.append(batch_embeddings)

            product_embeddings = np.vstack(product_embeddings)

        # Вычисляем схожесть
        similarities = cosine_similarity(tender_embedding, product_embeddings)[0]

        # Фильтруем результаты
        filtered_products = []
        stats = {
            'total_processed': len(valid_products),
            'above_threshold': 0,
            'final_count': 0
        }

        for product, similarity in zip(valid_products, similarities):
            if similarity >= threshold:
                stats['above_threshold'] += 1
                product['semantic_score'] = float(similarity)
                filtered_products.append(product)

        stats['final_count'] = len(filtered_products)

        # Сортируем по семантической близости
        filtered_products.sort(key=lambda x: x['semantic_score'], reverse=True)

        return {
            'products': filtered_products,
            'stats': stats
        }



    def combine_with_es_scores(self, products: List[Dict[str, Any]],
                             es_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Комбинирование ES и семантических скоров с защитой от ложных срабатываний."""
        for product in products:
            product_id = product['_id']
            es_score = es_scores.get(product_id, 0.0)
            semantic_score = product.get('semantic_score', 0.0)

            # Нормализуем ES скор (обычно он в диапазоне 0-100)
            normalized_es_score = min(es_score / 10.0, 1.0)

            # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: если ES скор очень низкий, снижаем влияние семантики
            if normalized_es_score < 0.05:  # ES скор < 0.5
                # ES почти не нашел совпадений - семантика может ошибаться
                # Даем 80% веса ES скору, только 20% семантике
                final_score = (0.8 * normalized_es_score + 0.2 * semantic_score)
            elif normalized_es_score < 0.2:  # ES скор < 2.0
                # Низкий ES скор - осторожнее с семантикой
                final_score = (0.65 * normalized_es_score + 0.35 * semantic_score)
            elif semantic_score > 0.75 and normalized_es_score < 0.3:
                # Высокая семантика + низкий ES = подозрительно
                final_score = (0.6 * normalized_es_score + 0.4 * semantic_score)
            else:
                # Нормальная ситуация - равные веса
                final_score = (0.5 * normalized_es_score + 0.5 * semantic_score)

            product['es_score'] = normalized_es_score
            product['final_score'] = final_score

        return products