from typing import List, Dict, Any
import logging

from app.core.settings import settings

# Используем стандартный логгер
logger = logging.getLogger(__name__)


class ScoreCombiner:
    """Комбинирование ES и семантических скоров"""

    def __init__(
            self,
            es_weight: float = None,
            semantic_weight: float = None
    ):
        self.es_weight = es_weight or settings.ES_SCORE_WEIGHT
        self.semantic_weight = semantic_weight or settings.SEMANTIC_SCORE_WEIGHT

        logger.info(f"Инициализация ScoreCombiner с весами: ES={self.es_weight}, Semantic={self.semantic_weight}")

    def combine_scores(self, products: List[Dict]) -> List[Dict]:
        """Комбинирует скоры с адаптивной логикой"""

        logger.info(f"Начало комбинирования скоров для {len(products)} товаров")

        suspicious_count = 0

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

            # Считаем подозрительные случаи
            if self._is_suspicious_score(normalized_es_score, semantic_score):
                suspicious_count += 1
                logger.info(
                    f"Подозрительное соотношение скоров для '{product.get('title', '')[:50]}...': "
                    f"ES={normalized_es_score:.3f}, semantic={semantic_score:.3f}, "
                    f"combined={combined_score:.3f}"
                )

        # Сортируем по комбинированному скору
        products.sort(key=lambda x: x['combined_score'], reverse=True)

        logger.info(f"Комбинирование завершено. Подозрительных случаев: {suspicious_count}")

        # Логируем топ результаты после комбинирования
        if products:
            logger.info("Топ-3 товара после комбинирования скоров:")
            for i, product in enumerate(products[:3], 1):
                logger.info(
                    f"  {i}. {product.get('title', '')[:60]}... "
                    f"(ES: {product['normalized_es_score']:.3f}, "
                    f"Sem: {product.get('semantic_score', 0):.3f}, "
                    f"Combined: {product['combined_score']:.3f})"
                )

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
            combined = 0.8 * es_score + 0.2 * semantic_score
            logger.info(f"Низкий ES скор ({es_score:.3f}), используем веса 80/20")
            return combined

        # Если ES скор низкий
        elif es_score < 0.2:
            # Осторожнее с семантикой - 65/35
            combined = 0.65 * es_score + 0.35 * semantic_score
            return combined

        # Если высокая семантика при низком ES
        elif semantic_score > 0.75 and es_score < 0.3:
            # Это подозрительно - 60/40
            combined = 0.6 * es_score + 0.4 * semantic_score
            return combined

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

        logger.info(
            f"Веса обновлены: ES={self.es_weight:.2f}, "
            f"Semantic={self.semantic_weight:.2f}"
        )