import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.settings import settings


@dataclass
class LLMConfig:
    """Конфигурация для LLM"""
    model_name: str = "mistral:7b"
    api_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.1
    max_tokens: int = 300  # Увеличено для более полных ответов
    timeout: int = 30
    max_workers: int = 4


class LLMFilter:
    """Фильтрация товаров с помощью локальной LLM"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or LLMConfig()

        # Счетчики для отладки
        self.debug_stats = {
            'total_analyzed': 0,
            'passed_threshold': 0,
            'failed_threshold': 0,
            'errors': 0,
            'scores_distribution': []
        }

        # Проверяем доступность Ollama при инициализации
        self._check_ollama_availability()

    def _check_ollama_availability(self):
        """Проверка доступности Ollama API"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.config.model_name not in model_names:
                    self.logger.warning(
                        f"Модель {self.config.model_name} не найдена. "
                        f"Доступные модели: {model_names}"
                    )
                else:
                    self.logger.info(f"Ollama доступна, модель {self.config.model_name} загружена")
        except Exception as e:
            self.logger.error(f"Ollama не доступна: {e}")
            raise RuntimeError(
                "Требуется запущенный Ollama сервис. "
                "Установите Ollama и запустите: ollama serve"
            )

    def filter_by_llm_analysis(
            self,
            tender: Dict,
            products: List[Dict],
            threshold: float = 0.7,
            top_k: int = -1
    ) -> List[Dict]:
        """Фильтрует товары с помощью LLM анализа"""

        if not products:
            return []

        start_time = time.time()
        self.logger.info(f"Начало LLM анализа для {len(products)} товаров с порогом {threshold}")

        # Сбрасываем статистику
        self.debug_stats = {
            'total_analyzed': 0,
            'passed_threshold': 0,
            'failed_threshold': 0,
            'errors': 0,
            'scores_distribution': []
        }

        # Подготавливаем данные тендера
        tender_requirements = self._prepare_tender_requirements(tender)
        self.logger.debug(f"Требования тендера:\n{tender_requirements}")

        # Ограничиваем количество товаров для анализа
        if len(products) > 100:
            self.logger.warning(
                f"Слишком много товаров ({len(products)}), "
                f"анализируем только первые 100"
            )
            products = products[:100]

        # Анализируем первые 3 товара последовательно для отладки
        self.logger.info("Анализируем первые 3 товара для отладки...")
        for i, product in enumerate(products[:3]):
            score, reasoning = self._analyze_product_match(
                tender_requirements,
                product,
                tender.get('name', ''),
                debug=True
            )
            self.logger.info(
                f"Товар {i + 1}: {product.get('title', '')[:50]}... "
                f"Скор: {score:.2f}, Причина: {reasoning[:100]}..."
            )

        # Анализируем все товары
        analyzed_products = []

        # Сначала попробуем с пониженным порогом для отладки
        debug_threshold = min(threshold, 0.5)
        self.logger.warning(f"Используем пониженный порог для отладки: {debug_threshold}")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Создаем задачи
            future_to_product = {
                executor.submit(
                    self._analyze_product_match,
                    tender_requirements,
                    product,
                    tender.get('name', ''),
                    debug=False
                ): product
                for product in products
            }

            # Обрабатываем результаты по мере готовности
            completed = 0
            for future in as_completed(future_to_product):
                product = future_to_product[future]
                completed += 1

                try:
                    score, reasoning = future.result()
                    self.debug_stats['total_analyzed'] += 1
                    self.debug_stats['scores_distribution'].append(score)

                    if score >= debug_threshold:
                        product['llm_score'] = score
                        product['llm_reasoning'] = reasoning
                        analyzed_products.append(product)
                        self.debug_stats['passed_threshold'] += 1
                    else:
                        self.debug_stats['failed_threshold'] += 1

                    # Логируем прогресс
                    if completed % 10 == 0 or completed == len(products):
                        self.logger.info(
                            f"Обработано: {completed}/{len(products)} "
                            f"(прошли фильтр: {len(analyzed_products)})"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Ошибка анализа товара {product.get('id')}: {e}"
                    )
                    self.debug_stats['errors'] += 1

        # Выводим статистику
        self._log_debug_stats()

        # Сортируем по LLM скору
        analyzed_products.sort(key=lambda x: x['llm_score'], reverse=True)

        # Ограничиваем количество если указано
        if top_k > 0:
            analyzed_products = analyzed_products[:top_k]

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"LLM анализ завершен: {len(analyzed_products)} из {len(products)} "
            f"прошли порог {debug_threshold} за {elapsed_time:.2f}с"
        )

        # Если ничего не прошло с пониженным порогом, выводим топ-10 по скорам
        if not analyzed_products and self.debug_stats['scores_distribution']:
            scores = sorted(self.debug_stats['scores_distribution'], reverse=True)[:10]
            self.logger.warning(f"Топ-10 скоров: {scores}")
            self.logger.warning("Ни один товар не прошел даже пониженный порог!")

        return analyzed_products

    def _prepare_tender_requirements(self, tender: Dict) -> str:
        """Подготавливает требования тендера для промпта"""

        requirements = []
        requirements.append(f"Товар: {tender.get('name', '')}")

        # Группируем характеристики по обязательности
        required_chars = []
        optional_chars = []

        for char in tender.get('characteristics', []):
            char_str = f"- {char['name']}: {char['value']}"
            if char.get('required', False):
                required_chars.append(char_str)
            else:
                optional_chars.append(char_str)

        if required_chars:
            requirements.append("\nОБЯЗАТЕЛЬНЫЕ характеристики (все должны совпадать):")
            requirements.extend(required_chars)

        if optional_chars:
            requirements.append("\nЖЕЛАТЕЛЬНЫЕ характеристики:")
            requirements.extend(optional_chars)

        return "\n".join(requirements)

    def _prepare_product_description(self, product: Dict) -> str:
        """Подготавливает описание товара для анализа"""

        description = []
        description.append(f"Название: {product.get('title', '')}")
        description.append(f"Категория: {product.get('category', '')}")

        if product.get('brand'):
            description.append(f"Производитель: {product['brand']}")

        # Добавляем атрибуты
        attributes = product.get('attributes', [])
        if attributes:
            description.append("\nХарактеристики товара:")
            # Ограничиваем количество атрибутов
            for attr in attributes[:20]:  # Увеличено до 20
                attr_name = attr.get('attr_name', '')
                attr_value = attr.get('attr_value', '')
                if attr_name and attr_value:
                    # Очищаем от лишних символов
                    attr_value = str(attr_value).strip()
                    if attr_value and attr_value not in ['', '-', 'нет данных']:
                        description.append(f"- {attr_name}: {attr_value}")

        return "\n".join(description)

    def _analyze_product_match(
            self,
            tender_requirements: str,
            product: Dict,
            tender_name: str,
            debug: bool = False
    ) -> Tuple[float, str]:
        """Анализирует соответствие товара требованиям с помощью LLM"""

        product_desc = self._prepare_product_description(product)

        # Упрощенный промпт для лучшего понимания
        prompt = f"""Проверь соответствие товара требованиям тендера.

ТРЕБОВАНИЯ ТЕНДЕРА:
{tender_requirements}

АНАЛИЗИРУЕМЫЙ ТОВАР:
{product_desc}

ИНСТРУКЦИЯ:
1. Если товар НЕ того типа что требуется (например, нужен блокнот, а это ручка) - скор 0.0
2. Проверь каждую ОБЯЗАТЕЛЬНУЮ характеристику
3. Если хотя бы одна обязательная НЕ выполнена - максимум скор 0.3
4. Если все обязательные выполнены - минимум скор 0.7
5. Добавь 0.1 за каждую желательную характеристику

Учитывай синонимы: блокнот = тетрадь, клетка = клетчатая линовка

Ответь JSON:
{{
    "score": число от 0.0 до 1.0,
    "reasoning": "краткое объяснение"
}}"""

        if debug:
            self.logger.debug(f"Промпт для отладки:\n{prompt[:500]}...")

        try:
            # Вызываем LLM
            response = requests.post(
                self.config.api_url,
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "stream": False,
                    "format": "json"
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '{}')

                if debug:
                    self.logger.debug(f"Ответ LLM: {llm_response}")

                # Парсим JSON ответ
                try:
                    # Очищаем ответ от возможных артефактов
                    llm_response = llm_response.strip()
                    if llm_response.startswith('```json'):
                        llm_response = llm_response[7:]
                    if llm_response.endswith('```'):
                        llm_response = llm_response[:-3]

                    analysis = json.loads(llm_response.strip())
                    score = float(analysis.get('score', 0.0))
                    reasoning = analysis.get('reasoning', 'Нет обоснования')

                    # Ограничиваем скор диапазоном [0, 1]
                    score = max(0.0, min(1.0, score))

                    return score, reasoning

                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.error(
                        f"Ошибка парсинга ответа LLM: {e}\n"
                        f"Ответ: {llm_response[:200]}..."
                    )
                    return 0.0, "Ошибка парсинга ответа"
            else:
                self.logger.error(f"Ошибка API: {response.status_code}")
                return 0.0, "Ошибка API"

        except requests.exceptions.Timeout:
            self.logger.error("Таймаут при вызове LLM")
            return 0.0, "Таймаут"
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            return 0.0, f"Ошибка: {str(e)}"

    def _log_debug_stats(self):
        """Выводит отладочную статистику"""

        self.logger.info("=" * 50)
        self.logger.info("СТАТИСТИКА LLM АНАЛИЗА:")
        self.logger.info(f"Всего проанализировано: {self.debug_stats['total_analyzed']}")
        self.logger.info(f"Прошли порог: {self.debug_stats['passed_threshold']}")
        self.logger.info(f"Не прошли порог: {self.debug_stats['failed_threshold']}")
        self.logger.info(f"Ошибок: {self.debug_stats['errors']}")

        if self.debug_stats['scores_distribution']:
            scores = self.debug_stats['scores_distribution']
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            self.logger.info(f"Средний скор: {avg_score:.3f}")
            self.logger.info(f"Мин/Макс скор: {min_score:.3f} / {max_score:.3f}")

            # Распределение по диапазонам
            ranges = {
                '0.0-0.3': sum(1 for s in scores if s <= 0.3),
                '0.3-0.5': sum(1 for s in scores if 0.3 < s <= 0.5),
                '0.5-0.7': sum(1 for s in scores if 0.5 < s <= 0.7),
                '0.7-1.0': sum(1 for s in scores if s > 0.7)
            }

            self.logger.info("Распределение скоров:")
            for range_name, count in ranges.items():
                percent = (count / len(scores)) * 100
                self.logger.info(f"  {range_name}: {count} ({percent:.1f}%)")

        self.logger.info("=" * 50)