from typing import List, Dict
import logging

from app.core.settings import settings


class LLMScoreCombiner:
    """Комбинирование ES и LLM скоров"""

    def __init__(
            self,
            es_weight: float = None,
            llm_weight: float = None
    ):
        self.logger = logging.getLogger(__name__)
        self.es_weight = es_weight or settings.ES_SCORE_WEIGHT_LLM
        self.llm_weight = llm_weight or settings.LLM_SCORE_WEIGHT

        # Нормализуем веса
        total = self.es_weight + self.llm_weight
        if total > 0:
            self.es_weight = self.es_weight / total
            self.llm_weight = self.llm_weight / total

        self.logger.info(
            f"Инициализация LLM Score Combiner: "
            f"ES вес={self.es_weight:.2f}, LLM вес={self.llm_weight:.2f}"
        )

    def combine_scores(self, products: List[Dict]) -> List[Dict]:
        """Комбинирует скоры с приоритетом LLM анализа"""

        if not products:
            return products

        for product in products:
            # Получаем скоры
            es_score = product.get('elasticsearch_score', 0.0)
            llm_score = product.get('llm_score', 0.0)

            # Нормализуем ES скор (обычно в диапазоне 0-100)
            if es_score > 0:
                normalized_es_score = min(es_score / 20.0, 1.0)
            else:
                normalized_es_score = 0.0

            # Комбинируем скоры
            combined_score = (
                    self.es_weight * normalized_es_score +
                    self.llm_weight * llm_score
            )

            # Дополнительная логика:
            # Если LLM дал очень низкую оценку, снижаем итоговый скор
            if llm_score < 0.3:
                combined_score *= 0.7  # Штраф 30%

            # Если LLM дал высокую оценку, но ES низкий - это подозрительно
            if llm_score > 0.8 and normalized_es_score < 0.1:
                combined_score *= 0.85  # Небольшой штраф

            # Сохраняем скоры
            product['normalized_es_score'] = normalized_es_score
            product['combined_score'] = round(combined_score, 4)

        # Сортируем по комбинированному скору
        products.sort(key=lambda x: x['combined_score'], reverse=True)

        # Логируем топ результаты
        if products:
            self.logger.info("Топ-3 после комбинирования скоров:")
            for i, product in enumerate(products[:3], 1):
                self.logger.info(
                    f"  {i}. {product.get('title', '')[:50]}... "
                    f"(combined={product['combined_score']:.3f}, "
                    f"llm={product.get('llm_score', 0):.3f}, "
                    f"es={product.get('normalized_es_score', 0):.3f})"
                )

        return products