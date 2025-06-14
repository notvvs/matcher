from typing import List, Dict, Any
import logging

from app.core.settings import settings


class ScoreCombiner:
    """Комбинирование ES и семантических скоров"""

    def __init__(
            self,
            es_weight: float = None,
            semantic_weight: float = None
    ):
        self.logger = logging.getLogger(__name__)
        self.es_weight = es_weight or settings.ES_SCORE_WEIGHT
        self.semantic_weight = semantic_weight or settings.SEMANTIC_SCORE_WEIGHT

    def combine_scores(self, products: List[Dict]) -> List[Dict]:
        """Комбинирует скоры с адаптивной логикой"""

        for product in products:
            # Получаем скоры
            es_score = product.get('elasticsearch_score', 0.0)
            semantic_score = product.get('semantic_score', 0.0)

            # Нормализуем ES скор (обычно в диапазоне 0-100)
            normalized_es_score = min(es_score / 10.0, 1.0)

            # Адаптивное комбинирование
            combined_score = self._adaptive_combine(
                normalized_es_score,
                semantic_score
            )

            # Сохраняем скоры
            product['normalized_es_score'] = normalized_es_score
            product['combined_score'] = combined_score

            # Логируем подозрительные случаи
            if self._is_suspicious_score(normalized_es_score, semantic_score):
                self.logger.debug(
                    f"Подозрительное соотношение скоров для '{product.get('title', '')[:50]}...': "
                    f"ES={normalized_es_score:.3f}, semantic={semantic_score:.3f}"
                )

        # Сортируем по комбинированному скору
        products.sort(key=lambda x: x['combined_score'], reverse=True)

        return products

    def _adaptive_combine(
            self,
            es_score: float,
            semantic_score: float
    ) -> float:
        """Адаптивное комбинирование скоров"""

        # Если ES почти ничего не нашел
        if es_score < 0.05:
            # Не доверяем семантике - даем 80% веса ES
            return 0.8 * es_score + 0.2 * semantic_score

        # Если ES скор низкий
        elif es_score < 0.2:
            # Осторожнее с семантикой - 65/35
            return 0.65 * es_score + 0.35 * semantic_score

        # Если высокая семантика при низком ES
        elif semantic_score > 0.75 and es_score < 0.3:
            # Это подозрительно - 60/40
            return 0.6 * es_score + 0.4 * semantic_score

        # Нормальная ситуация - используем стандартные веса
        else:
            return (
                    self.es_weight * es_score +
                    self.semantic_weight * semantic_score
            )

    def _is_suspicious_score(
            self,
            es_score: float,
            semantic_score: float
    ) -> bool:
        """Проверяет подозрительное соотношение скоров"""

        # Высокая семантика при очень низком ES
        return semantic_score > 0.7 and es_score < 0.1

    def adjust_weights(
            self,
            es_weight: float,
            semantic_weight: float
    ):
        """Изменяет веса для комбинирования"""

        # Нормализуем веса
        total = es_weight + semantic_weight

        if total > 0:
            self.es_weight = es_weight / total
            self.semantic_weight = semantic_weight / total
        else:
            self.es_weight = 0.5
            self.semantic_weight = 0.5

        self.logger.info(
            f"Веса обновлены: ES={self.es_weight:.2f}, "
            f"Semantic={self.semantic_weight:.2f}"
        )