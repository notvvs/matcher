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
    model_name: str = "llama3.2:1b"
    api_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 60
    max_workers: int = 4
    batch_size: int = 5


class LLMFilter:
    """Фильтрация товаров с помощью локальной LLM с батчевой обработкой"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or LLMConfig()
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
            f"LLM анализ для {len(products)} товаров "
            f"(порог: {threshold}, размер батча: {self.config.batch_size})"
        )

        # Подготавливаем данные тендера
        tender_requirements = self._prepare_tender_requirements(tender)

        # Ограничиваем количество товаров для анализа
        if len(products) > 100:
            self.logger.warning(f"Анализируем только первые 100 товаров из {len(products)}")
            products = products[:100]

        # Разбиваем товары на батчи
        batches = self._create_batches(products, self.config.batch_size)

        # Анализируем все батчи
        analyzed_products = []

        # Обрабатываем батчи параллельно
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self._analyze_batch,
                    tender_requirements,
                    batch,
                    tender.get('name', '')
                ): batch
                for batch in batches
            }

            # Обрабатываем результаты по мере готовности
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]

                try:
                    batch_results = future.result()

                    # Обрабатываем результаты батча
                    for product in batch:
                        product_id = product.get('id', '')
                        if product_id in batch_results:
                            score, reasoning = batch_results[product_id]

                            if score >= threshold:
                                product['llm_score'] = score
                                product['llm_reasoning'] = reasoning
                                analyzed_products.append(product)

                except Exception as e:
                    self.logger.error(f"Ошибка анализа батча: {e}")

        # Сортируем по LLM скору
        analyzed_products.sort(key=lambda x: x['llm_score'], reverse=True)

        # Ограничиваем количество если указано
        if top_k > 0:
            analyzed_products = analyzed_products[:top_k]

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"LLM анализ завершен: {len(analyzed_products)} из {len(products)} "
            f"прошли порог {threshold} за {elapsed_time:.2f}с"
        )

        return analyzed_products

    def _create_batches(self, products: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Разбивает список товаров на батчи"""
        return [products[i:i + batch_size] for i in range(0, len(products), batch_size)]

    def _prepare_tender_requirements(self, tender: Dict) -> str:
        """Подготавливает требования тендера для промпта"""
        requirements = [f"Товар: {tender.get('name', '')}"]

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

            # Добавляем ключевые атрибуты
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
            tender_name: str
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

                # Парсим JSON ответ
                try:
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
                    self.logger.error(f"Ошибка парсинга ответа LLM: {e}")
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
            self.logger.error("Таймаут при вызове LLM")
            return {
                product.get('id', ''): (0.0, "Таймаут")
                for product in batch
            }
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            return {
                product.get('id', ''): (0.0, f"Ошибка: {str(e)}")
                for product in batch
            }