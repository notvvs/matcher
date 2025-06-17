import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from app.core.settings import settings


@dataclass
class LLMConfig:
    """Конфигурация для LLM"""
    model_name: str = "mistral:7b"
    api_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.1
    max_tokens: int = 1000  # Увеличено для батчей
    timeout: int = 60  # Увеличено для батчей
    max_workers: int = 4
    batch_size: int = 5  # Количество товаров в одном батче


class LLMFilter:
    """Фильтрация товаров с помощью локальной LLM с батчевой обработкой"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or LLMConfig()

        # Счетчики для отладки
        self.debug_stats = {
            'total_analyzed': 0,
            'passed_threshold': 0,
            'failed_threshold': 0,
            'errors': 0,
            'scores_distribution': [],
            'batches_processed': 0,
            'batch_times': []
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
        """Фильтрует товары с помощью LLM анализа используя батчи"""

        if not products:
            return []

        start_time = time.time()
        self.logger.info(
            f"Начало батчевого LLM анализа для {len(products)} товаров "
            f"с порогом {threshold} (размер батча: {self.config.batch_size})"
        )

        # Сбрасываем статистику
        self.debug_stats = {
            'total_analyzed': 0,
            'passed_threshold': 0,
            'failed_threshold': 0,
            'errors': 0,
            'scores_distribution': [],
            'batches_processed': 0,
            'batch_times': []
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

        # Разбиваем товары на батчи
        batches = self._create_batches(products, self.config.batch_size)
        self.logger.info(f"Создано {len(batches)} батчей для обработки")

        # Анализируем первый батч для отладки
        if batches:
            self.logger.info("Анализируем первый батч для отладки...")
            first_batch_results = self._analyze_batch(
                tender_requirements,
                batches[0],
                tender.get('name', ''),
                debug=True
            )
            for product_id, (score, reasoning) in first_batch_results.items():
                self.logger.info(
                    f"Товар {product_id}: Скор: {score:.2f}, "
                    f"Причина: {reasoning[:100]}..."
                )

        # Анализируем все батчи
        analyzed_products = []
        debug_threshold = min(threshold, 0.5)
        self.logger.warning(f"Используем пониженный порог для отладки: {debug_threshold}")

        # Обрабатываем батчи параллельно
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Создаем задачи для батчей
            future_to_batch = {
                executor.submit(
                    self._analyze_batch,
                    tender_requirements,
                    batch,
                    tender.get('name', ''),
                    debug=False
                ): batch
                for batch in batches
            }

            # Обрабатываем результаты по мере готовности
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                batch_start = time.time()

                try:
                    batch_results = future.result()
                    batch_time = time.time() - batch_start
                    self.debug_stats['batch_times'].append(batch_time)
                    self.debug_stats['batches_processed'] += 1

                    # Обрабатываем результаты батча
                    for product in batch:
                        product_id = product.get('id', '')
                        if product_id in batch_results:
                            score, reasoning = batch_results[product_id]
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
                    self.logger.info(
                        f"Обработано батчей: {self.debug_stats['batches_processed']}/{len(batches)} "
                        f"(товаров прошли фильтр: {len(analyzed_products)})"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Ошибка анализа батча: {e}"
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
            f"Батчевый LLM анализ завершен: {len(analyzed_products)} из {len(products)} "
            f"прошли порог {debug_threshold} за {elapsed_time:.2f}с"
        )

        return analyzed_products

    def _create_batches(self, products: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Разбивает список товаров на батчи"""
        batches = []
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            batches.append(batch)
        return batches

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

    def _prepare_batch_descriptions(self, batch: List[Dict]) -> str:
        """Подготавливает описания нескольких товаров для батчевого анализа"""

        descriptions = []
        for i, product in enumerate(batch):
            desc = [f"ТОВАР {i + 1} (ID: {product.get('id', 'unknown')}):"]
            desc.append(f"Название: {product.get('title', '')}")
            desc.append(f"Категория: {product.get('category', '')}")

            if product.get('brand'):
                desc.append(f"Производитель: {product['brand']}")

            # Добавляем ключевые атрибуты (ограничиваем для экономии токенов)
            attributes = product.get('attributes', [])
            if attributes:
                desc.append("Основные характеристики:")
                for attr in attributes[:10]:  # Берем только первые 10
                    attr_name = attr.get('attr_name', '')
                    attr_value = attr.get('attr_value', '')
                    if attr_name and attr_value:
                        attr_value = str(attr_value).strip()
                        if attr_value and attr_value not in ['', '-', 'нет данных']:
                            desc.append(f"- {attr_name}: {attr_value}")

            descriptions.append("\n".join(desc))

        return "\n\n".join(descriptions)

    def _analyze_batch(
            self,
            tender_requirements: str,
            batch: List[Dict],
            tender_name: str,
            debug: bool = False
    ) -> Dict[str, Tuple[float, str]]:
        """Анализирует батч товаров за один вызов LLM"""

        batch_descriptions = self._prepare_batch_descriptions(batch)

        # Промпт для батчевого анализа
        prompt = f"""Проверь соответствие товаров требованиям тендера.

ТРЕБОВАНИЯ ТЕНДЕРА:
{tender_requirements}

АНАЛИЗИРУЕМЫЕ ТОВАРЫ:
{batch_descriptions}

ИНСТРУКЦИЯ:
Для каждого товара:
1. Если товар НЕ того типа что требуется - скор 0.0
2. Проверь каждую ОБЯЗАТЕЛЬНУЮ характеристику
3. Если хотя бы одна обязательная НЕ выполнена - максимум скор 0.3
4. Если все обязательные выполнены - минимум скор 0.7
5. Добавь 0.1 за каждую желательную характеристику

Учитывай синонимы и вариации написания.

Ответь JSON массивом для всех товаров:
{{
    "results": [
        {{
            "product_id": "ID товара",
            "score": число от 0.0 до 1.0,
            "reasoning": "краткое объяснение"
        }},
        ...
    ]
}}"""

        if debug:
            self.logger.debug(f"Промпт для батча ({len(batch)} товаров):\n{prompt[:1000]}...")

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
                    self.logger.debug(f"Ответ LLM для батча: {llm_response[:500]}...")

                # Парсим JSON ответ
                try:
                    # Очищаем ответ от возможных артефактов
                    llm_response = llm_response.strip()
                    if llm_response.startswith('```json'):
                        llm_response = llm_response[7:]
                    if llm_response.endswith('```'):
                        llm_response = llm_response[:-3]

                    analysis = json.loads(llm_response.strip())
                    results = {}

                    # Обрабатываем результаты для каждого товара
                    for item in analysis.get('results', []):
                        product_id = item.get('product_id', '')
                        score = float(item.get('score', 0.0))
                        reasoning = item.get('reasoning', 'Нет обоснования')

                        # Ограничиваем скор диапазоном [0, 1]
                        score = max(0.0, min(1.0, score))

                        results[product_id] = (score, reasoning)

                    # Добавляем результаты для товаров, которых нет в ответе
                    for product in batch:
                        product_id = product.get('id', '')
                        if product_id not in results:
                            results[product_id] = (0.0, "Не проанализирован")

                    return results

                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.error(
                        f"Ошибка парсинга ответа LLM для батча: {e}\n"
                        f"Ответ: {llm_response[:200]}..."
                    )
                    # Возвращаем нулевые результаты для всего батча
                    return {
                        product.get('id', ''): (0.0, "Ошибка парсинга")
                        for product in batch
                    }
            else:
                self.logger.error(f"Ошибка API: {response.status_code}")
                return {
                    product.get('id', ''): (0.0, "Ошибка API")
                    for product in batch
                }

        except requests.exceptions.Timeout:
            self.logger.error("Таймаут при вызове LLM для батча")
            return {
                product.get('id', ''): (0.0, "Таймаут")
                for product in batch
            }
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM для батча: {e}")
            return {
                product.get('id', ''): (0.0, f"Ошибка: {str(e)}")
                for product in batch
            }

    def _log_debug_stats(self):
        """Выводит отладочную статистику"""

        self.logger.info("=" * 50)
        self.logger.info("СТАТИСТИКА БАТЧЕВОГО LLM АНАЛИЗА:")
        self.logger.info(f"Всего проанализировано: {self.debug_stats['total_analyzed']}")
        self.logger.info(f"Прошли порог: {self.debug_stats['passed_threshold']}")
        self.logger.info(f"Не прошли порог: {self.debug_stats['failed_threshold']}")
        self.logger.info(f"Ошибок: {self.debug_stats['errors']}")
        self.logger.info(f"Батчей обработано: {self.debug_stats['batches_processed']}")

        if self.debug_stats['batch_times']:
            avg_batch_time = sum(self.debug_stats['batch_times']) / len(self.debug_stats['batch_times'])
            self.logger.info(f"Среднее время на батч: {avg_batch_time:.2f}с")

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
                if scores:
                    percent = (count / len(scores)) * 100
                    self.logger.info(f"  {range_name}: {count} ({percent:.1f}%)")

        self.logger.info("=" * 50)