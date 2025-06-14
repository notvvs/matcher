"""
Гибридный AttributeMatcher v2 - оптимизированный для производительности
"""

import re
import json
import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import redis
from pymongo import MongoClient

from app.config.settings import settings
from app.utils.logger import setup_logger


@dataclass
class AttributeMatch:
    """Результат сопоставления атрибута"""
    matched: bool
    confidence: float
    tender_char: Dict[str, Any]
    product_attr: Optional[Dict[str, str]]
    reason: str
    match_type: str  # 'exact', 'semantic', 'partial', 'numeric'


@dataclass
class ProductMatch:
    """Результат сопоставления товара"""
    is_suitable: bool
    score: float
    matched_required: int
    total_required: int
    matched_optional: int
    total_optional: int
    matches: List[AttributeMatch]
    processing_time: float

    @property
    def match_percentage(self) -> float:
        total = self.total_required + self.total_optional
        if total == 0:
            return 0.0
        matched = self.matched_required + self.matched_optional
        return (matched / total) * 100


class HybridAttributeMatcher:
    """Гибридный матчер с ML и оптимизациями"""

    def __init__(self,
                 use_cache: bool = True,
                 cache_ttl: int = 86400,  # 24 часа
                 batch_size: int = 32,
                 num_workers: int = 4):

        self.logger = setup_logger(__name__)
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Инициализация компонентов
        self._init_ml_model()
        self._init_cache()
        self._init_patterns()
        self._init_mappings()

        # Статистика
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_matches': 0,
            'avg_processing_time': 0
        }

        self.logger.info("HybridAttributeMatcher инициализирован")

    def _init_ml_model(self):
        """Инициализация ML модели"""
        self.logger.info("Загрузка ML модели...")
        self.model = SentenceTransformer(settings.EMBEDDINGS_MODEL)

        # Оптимизация для CPU
        self.model.max_seq_length = 128  # Уменьшаем для скорости

        # Предзагрузка модели
        _ = self.model.encode(["тест"], show_progress_bar=False)
        self.logger.info("ML модель загружена")

    def _init_cache(self):
        """Инициализация кэша"""
        if self.use_cache:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=False  # Для бинарных данных
                )
                self.redis_client.ping()
                self.logger.info("Redis кэш подключен")
            except:
                self.logger.warning("Redis недоступен, работаем без кэша")
                self.use_cache = False
                self.redis_client = None
        else:
            self.redis_client = None

    def _init_patterns(self):
        """Паттерны для парсинга"""
        self.patterns = {
            # Числа с единицами измерения
            'number_unit': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*([а-яА-Яa-zA-Z]+(?:/[а-яА-Яa-zA-Z]+)?)?'
            ),
            # Диапазоны
            'range': re.compile(
                r'(?:от\s*)?(\d+(?:[.,]\d+)?)\s*(?:до|-)\s*(\d+(?:[.,]\d+)?)'
            ),
            # Операторы сравнения
            'comparison': re.compile(
                r'([<>≤≥]|(?:более|менее|от|до|не\s*(?:более|менее)))\s*(\d+(?:[.,]\d+)?)'
            ),
            # Размеры (ШxВxГ)
            'dimensions': re.compile(
                r'(\d+(?:[.,]\d+)?)\s*[xх×]\s*(\d+(?:[.,]\d+)?)\s*(?:[xх×]\s*(\d+(?:[.,]\d+)?))?'
            )
        }

    def _init_mappings(self):
        """Базовые маппинги единиц измерения"""
        self.unit_mappings = {
            # Объем
            'мл': ['ml', 'миллилитр', 'миллилитров'],
            'л': ['l', 'литр', 'литров'],
            # Вес
            'г': ['g', 'гр', 'грамм', 'граммов'],
            'кг': ['kg', 'килограмм', 'килограммов'],
            # Время
            'сек': ['с', 'секунд', 'секунды', 'sec', 's'],
            'мин': ['м', 'минут', 'минуты', 'min'],
            # Размер
            'мм': ['mm', 'миллиметр', 'миллиметров'],
            'см': ['cm', 'сантиметр', 'сантиметров'],
            'м': ['m', 'метр', 'метров']
        }

        # Обратный индекс
        self.unit_reverse = {}
        for canonical, variants in self.unit_mappings.items():
            self.unit_reverse[canonical] = canonical
            for var in variants:
                self.unit_reverse[var.lower()] = canonical

    def match_product(self, tender: Dict[str, Any], product: Dict[str, Any]) -> ProductMatch:
        """Основной метод сопоставления"""

        start_time = time.time()
        self.stats['total_matches'] += 1

        tender_name = tender.get('name', 'Без названия')
        product_title = product.get('title', 'Без названия')

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Сопоставление: {tender_name[:60]}...")
        self.logger.info(f"Товар: {product_title[:60]}...")

        # Подготовка данных
        characteristics = tender.get('characteristics', [])
        product_attrs = product.get('attributes', [])

        # Убираем дубликаты атрибутов товара
        product_attrs = self._deduplicate_attributes(product_attrs)

        # Предвычисляем эмбеддинги для всех атрибутов товара
        product_embeddings = self._get_product_embeddings(product_attrs)

        # Параллельная обработка характеристик
        matches = self._match_characteristics_parallel(
            characteristics,
            product_attrs,
            product_embeddings
        )

        # Подсчет результатов
        matched_required = sum(1 for m in matches if m.matched and m.tender_char.get('required', False))
        total_required = sum(1 for m in matches if m.tender_char.get('required', False))
        matched_optional = sum(1 for m in matches if m.matched and not m.tender_char.get('required', False))
        total_optional = sum(1 for m in matches if not m.tender_char.get('required', False))

        is_suitable = matched_required == total_required

        # Расчет скора
        score = self._calculate_score(matches)

        processing_time = time.time() - start_time

        # Обновляем статистику
        self.stats['avg_processing_time'] = (
            self.stats['avg_processing_time'] * (self.stats['total_matches'] - 1) + processing_time
        ) / self.stats['total_matches']

        result = ProductMatch(
            is_suitable=is_suitable,
            score=score,
            matched_required=matched_required,
            total_required=total_required,
            matched_optional=matched_optional,
            total_optional=total_optional,
            matches=matches,
            processing_time=processing_time
        )

        # Логирование результата
        self.logger.info(f"Результат: {'✅ ПОДХОДИТ' if is_suitable else '❌ НЕ ПОДХОДИТ'}")
        self.logger.info(f"Обязательных: {matched_required}/{total_required}")
        self.logger.info(f"Опциональных: {matched_optional}/{total_optional}")
        self.logger.info(f"Процент совпадения: {result.match_percentage:.1f}%")
        self.logger.info(f"Время обработки: {processing_time:.3f} сек")
        self.logger.info(f"{'='*80}\n")

        return result

    def _deduplicate_attributes(self, attributes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Удаляет дубликаты атрибутов"""
        seen = set()
        unique = []

        for attr in attributes:
            key = (attr.get('attr_name', '').strip().lower(),
                   attr.get('attr_value', '').strip().lower())
            if key not in seen and key[0]:  # Пропускаем пустые
                seen.add(key)
                unique.append(attr)

        return unique

    def _get_product_embeddings(self, attributes: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """Получает эмбеддинги для атрибутов товара с кэшированием"""

        embeddings = {}
        to_encode = []
        cache_keys = []

        for attr in attributes:
            attr_text = f"{attr['attr_name']}: {attr['attr_value']}"
            cache_key = f"emb:{hashlib.md5(attr_text.encode()).hexdigest()}"

            # Проверяем кэш
            if self.use_cache:
                cached = self.redis_client.get(cache_key)
                if cached:
                    embeddings[attr_text] = pickle.loads(cached)
                    self.stats['cache_hits'] += 1
                    continue

            to_encode.append(attr_text)
            cache_keys.append(cache_key)
            self.stats['cache_misses'] += 1

        # Кодируем некэшированные
        if to_encode:
            self.logger.debug(f"Кодирование {len(to_encode)} атрибутов...")
            new_embeddings = self.model.encode(
                to_encode,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Сохраняем в кэш
            for text, embedding, cache_key in zip(to_encode, new_embeddings, cache_keys):
                embeddings[text] = embedding
                if self.use_cache:
                    self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        pickle.dumps(embedding)
                    )

        return embeddings

    def _match_characteristics_parallel(self,
                                      characteristics: List[Dict[str, Any]],
                                      product_attrs: List[Dict[str, str]],
                                      product_embeddings: Dict[str, np.ndarray]) -> List[AttributeMatch]:
        """Параллельное сопоставление характеристик"""

        matches = []

        # Для небольшого количества характеристик последовательная обработка быстрее
        if len(characteristics) <= 5:
            for char in characteristics:
                match = self._match_single_characteristic(char, product_attrs, product_embeddings)
                matches.append(match)
        else:
            # Параллельная обработка для большого количества
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for char in characteristics:
                    future = executor.submit(
                        self._match_single_characteristic,
                        char, product_attrs, product_embeddings
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    matches.append(future.result())

        return matches

    def _match_single_characteristic(self,
                                   char: Dict[str, Any],
                                   product_attrs: List[Dict[str, str]],
                                   product_embeddings: Dict[str, np.ndarray]) -> AttributeMatch:
        """Сопоставление одной характеристики"""

        char_name = char.get('name', '').strip()
        char_value = char.get('value', '').strip()
        char_type = char.get('type', 'Качественная')

        if not char_name:
            return AttributeMatch(
                matched=False,
                confidence=0.0,
                tender_char=char,
                product_attr=None,
                reason="Пустое название характеристики",
                match_type='error'
            )

        # Кодируем характеристику тендера
        char_text = f"{char_name}: {char_value}"
        char_embedding = self._get_embedding(char_text)

        # Ищем лучшее соответствие
        best_match = None
        best_score = 0.0
        best_attr = None
        best_type = 'none'

        for attr in product_attrs:
            attr_text = f"{attr['attr_name']}: {attr['attr_value']}"
            attr_embedding = product_embeddings.get(attr_text)

            if attr_embedding is None:
                continue

            # Семантическая близость
            semantic_similarity = cosine_similarity([char_embedding], [attr_embedding])[0][0]

            # Если семантическая близость высокая, проверяем значения
            if semantic_similarity > 0.6:
                # Проверка значений в зависимости от типа
                if char_type == 'Количественная':
                    value_match = self._match_numeric_values(char_value, attr['attr_value'])
                else:
                    value_match = self._match_categorical_values(char_value, attr['attr_value'])

                # Комбинированный скор
                if value_match['matched']:
                    combined_score = semantic_similarity * 0.6 + value_match['confidence'] * 0.4
                    match_type = value_match.get('type', 'semantic')
                else:
                    combined_score = semantic_similarity * 0.3  # Штраф за несовпадение значения
                    match_type = 'partial'

                if combined_score > best_score:
                    best_score = combined_score
                    best_attr = attr
                    best_match = value_match
                    best_type = match_type

        # Порог для принятия решения
        threshold = 0.65 if char.get('required', False) else 0.6

        if best_score >= threshold and best_match and best_match['matched']:
            return AttributeMatch(
                matched=True,
                confidence=best_score,
                tender_char=char,
                product_attr=best_attr,
                reason=f"Найдено соответствие: {best_attr['attr_name']} = {best_attr['attr_value']}",
                match_type=best_type
            )
        else:
            reason = "Атрибут не найден"
            if best_attr:
                reason = f"Найден атрибут '{best_attr['attr_name']}', но значение не подходит"

            return AttributeMatch(
                matched=False,
                confidence=best_score,
                tender_char=char,
                product_attr=best_attr,
                reason=reason,
                match_type='not_found'
            )

    def _get_embedding(self, text: str) -> np.ndarray:
        """Получает эмбеддинг с кэшированием"""

        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"

        # Проверяем кэш
        if self.use_cache:
            cached = self.redis_client.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return pickle.loads(cached)

        self.stats['cache_misses'] += 1

        # Кодируем
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]

        # Сохраняем в кэш
        if self.use_cache:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                pickle.dumps(embedding)
            )

        return embedding

    def _match_numeric_values(self, tender_value: str, product_value: str) -> Dict[str, Any]:
        """Сопоставление числовых значений"""

        # Парсим числа и единицы измерения
        tender_parsed = self._parse_numeric(tender_value)
        product_parsed = self._parse_numeric(product_value)

        if not tender_parsed['number'] or not product_parsed['number']:
            # Не удалось распарсить как числа - пробуем как строки
            return self._match_categorical_values(tender_value, product_value)

        # Приводим к одинаковым единицам измерения
        if tender_parsed['unit'] and product_parsed['unit']:
            tender_unit = self.unit_reverse.get(tender_parsed['unit'].lower(), tender_parsed['unit'])
            product_unit = self.unit_reverse.get(product_parsed['unit'].lower(), product_parsed['unit'])

            if tender_unit != product_unit:
                # Пытаемся конвертировать
                product_parsed = self._try_convert_units(
                    product_parsed['number'],
                    product_unit,
                    tender_unit
                ) or product_parsed

        # Сравнение в зависимости от оператора
        tender_num = tender_parsed['number']
        product_num = product_parsed['number']
        operator = tender_parsed.get('operator', 'eq')

        matched = False
        tolerance = 0.1  # 10% допуск

        if operator == 'eq':
            matched = abs(product_num - tender_num) <= tender_num * tolerance
        elif operator == 'gte':
            matched = product_num >= tender_num * (1 - tolerance)
        elif operator == 'lte':
            matched = product_num <= tender_num * (1 + tolerance)
        elif operator == 'range':
            min_val = tender_parsed.get('range_min', tender_num) * (1 - tolerance)
            max_val = tender_parsed.get('range_max', tender_num) * (1 + tolerance)
            matched = min_val <= product_num <= max_val

        return {
            'matched': matched,
            'confidence': 0.9 if matched else 0.3,
            'type': 'numeric',
            'details': {
                'tender': tender_num,
                'product': product_num,
                'operator': operator
            }
        }

    def _match_categorical_values(self, tender_value: str, product_value: str) -> Dict[str, Any]:
        """Сопоставление категориальных значений"""

        # Нормализация
        tender_norm = tender_value.lower().strip()
        product_norm = product_value.lower().strip()

        # Точное совпадение
        if tender_norm == product_norm:
            return {
                'matched': True,
                'confidence': 1.0,
                'type': 'exact'
            }

        # Проверка вхождения
        if len(tender_norm) > 3:
            if tender_norm in product_norm or product_norm in tender_norm:
                return {
                    'matched': True,
                    'confidence': 0.85,
                    'type': 'substring'
                }

        # Семантическая близость для коротких значений
        if len(tender_norm) <= 20 and len(product_norm) <= 20:
            similarity = cosine_similarity(
                [self._get_embedding(tender_norm)],
                [self._get_embedding(product_norm)]
            )[0][0]

            if similarity > 0.8:
                return {
                    'matched': True,
                    'confidence': similarity,
                    'type': 'semantic'
                }

        # Специальные случаи
        # Да/Нет
        positive = {'да', 'есть', 'имеется', 'присутствует', '+', 'true', 'наличие'}
        negative = {'нет', 'отсутствует', 'без', '-', 'false', 'отсутствие'}

        if (tender_norm in positive and product_norm in positive) or \
           (tender_norm in negative and product_norm in negative):
            return {
                'matched': True,
                'confidence': 0.95,
                'type': 'boolean'
            }

        return {
            'matched': False,
            'confidence': 0.0,
            'type': 'no_match'
        }

    def _parse_numeric(self, value: str) -> Dict[str, Any]:
        """Парсинг числового значения"""

        if not value:
            return {'number': None, 'unit': None}

        value = value.lower().strip()

        # Проверяем операторы сравнения
        operator = 'eq'
        comp_match = self.patterns['comparison'].search(value)
        if comp_match:
            op_text = comp_match.group(1)
            if 'боле' in op_text or '>' in op_text or '≥' in op_text:
                operator = 'gte'
            elif 'мене' in op_text or '<' in op_text or '≤' in op_text:
                operator = 'lte'

        # Проверяем диапазоны
        range_match = self.patterns['range'].search(value)
        if range_match:
            return {
                'number': float(range_match.group(1).replace(',', '.')),
                'range_min': float(range_match.group(1).replace(',', '.')),
                'range_max': float(range_match.group(2).replace(',', '.')),
                'operator': 'range',
                'unit': None
            }

        # Обычные числа с единицами
        num_match = self.patterns['number_unit'].search(value)
        if num_match:
            number = float(num_match.group(1).replace(',', '.'))
            unit = num_match.group(2) if num_match.group(2) else None

            return {
                'number': number,
                'unit': unit,
                'operator': operator
            }

        return {'number': None, 'unit': None}

    def _try_convert_units(self, value: float, from_unit: str, to_unit: str) -> Optional[Dict[str, Any]]:
        """Попытка конвертации единиц измерения"""

        # Простые конверсии
        conversions = {
            ('мл', 'л'): 0.001,
            ('л', 'мл'): 1000,
            ('г', 'кг'): 0.001,
            ('кг', 'г'): 1000,
            ('мм', 'см'): 0.1,
            ('см', 'мм'): 10,
            ('см', 'м'): 0.01,
            ('м', 'см'): 100
        }

        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            return {
                'number': value * conversions[key],
                'unit': to_unit,
                'operator': 'eq'
            }

        return None

    def _calculate_score(self, matches: List[AttributeMatch]) -> float:
        """Расчет итогового скора"""

        if not matches:
            return 0.0

        # Веса для разных типов характеристик
        weights = {
            'required_match': 1.0,
            'required_miss': -2.0,
            'optional_match': 0.5,
            'optional_miss': 0.0
        }

        score = 0.0
        max_possible = 0.0

        for match in matches:
            is_required = match.tender_char.get('required', False)

            if match.matched:
                if is_required:
                    score += weights['required_match'] * match.confidence
                    max_possible += weights['required_match']
                else:
                    score += weights['optional_match'] * match.confidence
                    max_possible += weights['optional_match']
            else:
                if is_required:
                    score += weights['required_miss']
                    max_possible += weights['required_match']
                else:
                    score += weights['optional_miss']
                    max_possible += weights['optional_match']

        # Нормализация
        if max_possible > 0:
            normalized = (score + abs(weights['required_miss'] * len(matches))) / \
                        (max_possible + abs(weights['required_miss'] * len(matches)))
            return max(0.0, min(1.0, normalized))

        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики работы"""

        cache_total = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / cache_total if cache_total > 0 else 0

        return {
            'total_matches': self.stats['total_matches'],
            'avg_processing_time': f"{self.stats['avg_processing_time']:.3f} сек",
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses']
        }

    def precompute_product_embeddings(self,
                                    products: List[Dict[str, Any]],
                                    batch_size: int = 100) -> None:
        """Предвычисление эмбеддингов для списка товаров"""

        self.logger.info(f"Предвычисление эмбеддингов для {len(products)} товаров...")

        all_texts = []
        for product in products:
            attrs = self._deduplicate_attributes(product.get('attributes', []))
            for attr in attrs:
                all_texts.append(f"{attr['attr_name']}: {attr['attr_value']}")

        # Батчевая обработка
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            _ = self.model.encode(batch, show_progress_bar=True)

            # Сохраняем в кэш
            if self.use_cache:
                embeddings = self.model.encode(batch, show_progress_bar=False)
                for text, emb in zip(batch, embeddings):
                    cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
                    self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        pickle.dumps(emb)
                    )

        self.logger.info("Предвычисление завершено")


# Вспомогательные функции для интеграции

def create_matcher() -> HybridAttributeMatcher:
    """Создает экземпляр матчера с оптимальными настройками"""

    # Проверяем доступность Redis
    try:
        redis.Redis(host='localhost', port=6379).ping()
        use_cache = True
    except:
        use_cache = False

    return HybridAttributeMatcher(
        use_cache=use_cache,
        cache_ttl=86400,  # 24 часа
        batch_size=32,
        num_workers=4
    )


def match_products_batch(matcher: HybridAttributeMatcher,
                        tender: Dict[str, Any],
                        products: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], ProductMatch]]:
    """Сопоставление тендера с несколькими товарами"""

    results = []

    # Предвычисляем эмбеддинги для всех товаров
    matcher.precompute_product_embeddings(products, batch_size=50)

    # Сопоставляем
    for product in products:
        match_result = matcher.match_product(tender, product)
        if match_result.is_suitable:
            results.append((product, match_result))

    # Сортируем по скору
    results.sort(key=lambda x: x[1].score, reverse=True)

    return results


