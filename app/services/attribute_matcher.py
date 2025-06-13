"""
Сервис для точного сопоставления характеристик тендера и атрибутов товара
с умной обработкой категорий
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from dataclasses import dataclass
import logging
from pathlib import Path

from app.config.settings import settings


@dataclass
class MatchResult:
    """Результат сопоставления характеристики"""
    matched: bool
    score: float
    confidence: float  # Уверенность в совпадении 0-1
    matched_attribute: Optional[Dict[str, str]]
    reason: str
    details: Dict[str, Any] = None


class AttributeMatcher:
    """Rule-based матчер для сопоставления характеристик с умной логикой категорий"""

    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)

        # Загружаем конфигурацию
        self.config_path = config_path or settings.CONFIG_DIR / 'attribute_matching_config.json'
        self.config = self._load_config()

        # Веса для скоринга
        self.weights = self.config.get('weights', {
            'required_match': 1.0,
            'required_miss': -2.0,
            'optional_match': 0.5,
            'optional_miss': 0.0,
            'partial_match': 0.3
        })

        # Пороги
        self.thresholds = self.config.get('thresholds', {
            'name_match': 0.7,
            'value_match': 0.8,
            'numeric_tolerance': 0.1
        })

        # Словари соответствий
        self.attribute_mappings = self.config.get('attribute_mappings', {})
        self.value_mappings = self.config.get('value_mappings', {})
        self.unit_conversions = self.config.get('unit_conversions', {})

        # НОВОЕ: Правила для категорий
        self.category_rules = self.config.get('category_rules', {})
        self.smart_matching_enabled = self.config.get('smart_matching_enabled', True)

        # Статистика
        self.stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'partial_matches': 0,
            'smart_fixes': 0  # Сколько раз умная логика помогла
        }

    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}, using defaults")

        # Дефолтная конфигурация с умными правилами
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Дефолтная конфигурация с умными правилами для категорий"""
        config = {
            'smart_matching_enabled': True,

            # Правила для категорий
            'category_rules': {
                # USB флешки
                'usb_flash': {
                    'category_patterns': ['флеш', 'flash', 'usb-накопител', 'usb накопител'],
                    'attribute_overrides': {
                        'тип': {
                            'acceptable_values': ['классический', 'стандартный', 'обычный', 'usb flash', 'флешка'],
                            'confidence': 0.9
                        }
                    }
                },

                # Ноутбуки
                'notebooks': {
                    'category_patterns': ['ноутбук', 'notebook', 'laptop', 'лэптоп'],
                    'attribute_overrides': {
                        'тип': {
                            'acceptable_values': ['ноутбук', 'портативный компьютер', 'лэптоп', 'классический'],
                            'confidence': 0.9
                        },
                        'форм-фактор': {
                            'acceptable_values': ['ноутбук', 'portable', 'мобильный'],
                            'confidence': 0.85
                        }
                    }
                },

                # Мониторы
                'monitors': {
                    'category_patterns': ['монитор', 'дисплей', 'экран', 'display'],
                    'attribute_overrides': {
                        'разрешение': {
                            'value_mappings': {
                                'full hd': ['1920x1080', '1920х1080', '1080p', 'fhd'],
                                '4k': ['3840x2160', '2160p', 'uhd', 'ultra hd'],
                                'hd': ['1366x768', '1280x720', '720p']
                            },
                            'confidence': 0.95
                        }
                    }
                },

                # Принтеры
                'printers': {
                    'category_patterns': ['принтер', 'printer', 'мфу', 'печат'],
                    'attribute_overrides': {
                        'тип печати': {
                            'acceptable_values': ['лазерный', 'струйный', 'матричный', 'термо'],
                            'confidence': 0.9
                        },
                        'цветность': {
                            'value_mappings': {
                                'цветной': ['цветная печать', 'color', 'полноцветный'],
                                'монохромный': ['черно-белый', 'ч/б', 'mono', 'grayscale']
                            }
                        }
                    }
                },

                # Канцелярские товары
                'stationery': {
                    'category_patterns': ['канцел', 'бумаг', 'ручк', 'каранд', 'папк', 'блок', 'тетрад'],
                    'attribute_overrides': {
                        'цвет': {
                            'fuzzy_match': True,  # Разрешаем нечеткое совпадение цветов
                            'confidence': 0.8
                        }
                    }
                },

                # Мебель
                'furniture': {
                    'category_patterns': ['стол', 'стул', 'кресло', 'шкаф', 'тумб', 'полк', 'мебел'],
                    'attribute_overrides': {
                        'материал': {
                            'fuzzy_match': True,
                            'confidence': 0.85
                        }
                    }
                }
            },

            # Стандартные маппинги
            'attribute_mappings': {
                'объем памяти': ['память', 'объем', 'емкость памяти', 'storage', 'объём памяти', 'ram'],
                'тип': ['вид', 'тип устройства', 'категория', 'type', 'модель', 'тип товара'],
                'цвет': ['окраска', 'расцветка', 'оттенок', 'тон', 'цвет корпуса'],
                'размер': ['габариты', 'габарит', 'размеры', 'dimensions'],
                'материал': ['состав', 'материал изготовления', 'сырье', 'материал корпуса']
            },

            'value_mappings': {
                'черный': ['чёрный', 'black', 'темный', 'dark'],
                'usb flash': ['usb-flash', 'флешка', 'флеш-накопитель', 'usb накопитель', 'flash drive'],
                'full hd': ['1920x1080', '1080p', 'fhd', 'фулл хд']
            },

            'unit_conversions': {
                'гб': {'to': 'мб', 'factor': 1024},
                'gb': {'to': 'mb', 'factor': 1024}
            },

            'weights': {
                'required_match': 1.0,
                'required_miss': -2.0,
                'optional_match': 0.5,
                'optional_miss': 0.0,
                'partial_match': 0.3
            },

            'thresholds': {
                'name_match': 0.7,
                'value_match': 0.8,
                'numeric_tolerance': 0.1
            }
        }

        # Добавляем остальные маппинги из старой конфигурации
        return config

    def match_product(self, tender: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод для сопоставления товара с тендером
        Теперь с умной обработкой категорий
        """

        self.stats['total_matches'] += 1

        characteristics = tender.get('characteristics', [])
        product_attrs = product.get('attributes', [])

        if not characteristics:
            return {
                'score': 1.0,
                'confidence': 1.0,
                'matched_required': 0,
                'total_required': 0,
                'matched_optional': 0,
                'total_optional': 0,
                'match_percentage': 100.0,
                'is_suitable': True,
                'details': []
            }

        # Определяем категорию товара для умного матчинга
        product_category = product.get('category', '').lower()
        category_rule = self._get_category_rule(product_category) if self.smart_matching_enabled else None

        # Разделяем характеристики
        required_chars = [c for c in characteristics if c.get('required', False)]
        optional_chars = [c for c in characteristics if not c.get('required', False)]

        # Результаты
        results = {
            'score': 0.0,
            'confidence': 0.0,
            'matched_required': 0,
            'total_required': len(required_chars),
            'matched_optional': 0,
            'total_optional': len(optional_chars),
            'match_percentage': 0.0,
            'is_suitable': False,
            'details': [],
            'smart_fixes_applied': 0
        }

        total_confidence = 0.0
        confidence_count = 0

        # Проверяем обязательные характеристики
        for char in required_chars:
            match_result = self._match_characteristic(char, product_attrs, category_rule, product_category)

            if match_result.matched:
                results['matched_required'] += 1
                results['score'] += self.weights['required_match'] * match_result.score

                # Считаем умные исправления
                if match_result.details and match_result.details.get('smart_fix'):
                    results['smart_fixes_applied'] += 1
                    self.stats['smart_fixes'] += 1
            else:
                results['score'] += self.weights['required_miss']

            total_confidence += match_result.confidence
            confidence_count += 1

            results['details'].append({
                'characteristic': char,
                'matched': match_result.matched,
                'score': match_result.score,
                'confidence': match_result.confidence,
                'matched_with': match_result.matched_attribute,
                'reason': match_result.reason,
                'required': True,
                'smart_fix': match_result.details.get('smart_fix', False) if match_result.details else False
            })

        # Проверяем опциональные характеристики
        for char in optional_chars:
            match_result = self._match_characteristic(char, product_attrs, category_rule, product_category)

            if match_result.matched:
                results['matched_optional'] += 1
                results['score'] += self.weights['optional_match'] * match_result.score
            else:
                results['score'] += self.weights['optional_miss']

            total_confidence += match_result.confidence
            confidence_count += 1

            results['details'].append({
                'characteristic': char,
                'matched': match_result.matched,
                'score': match_result.score,
                'confidence': match_result.confidence,
                'matched_with': match_result.matched_attribute,
                'reason': match_result.reason,
                'required': False,
                'smart_fix': match_result.details.get('smart_fix', False) if match_result.details else False
            })

        # Считаем итоговые метрики
        results['is_suitable'] = (results['matched_required'] == results['total_required'])

        # Процент совпадения
        total_matched = results['matched_required'] + results['matched_optional']
        total_chars = results['total_required'] + results['total_optional']
        results['match_percentage'] = (total_matched / total_chars * 100) if total_chars > 0 else 0

        # Средняя уверенность
        results['confidence'] = (total_confidence / confidence_count) if confidence_count > 0 else 0

        # Нормализуем скор
        max_possible_score = (
            len(required_chars) * self.weights['required_match'] +
            len(optional_chars) * self.weights['optional_match']
        )

        if max_possible_score > 0:
            results['score'] = max(0, min(1, results['score'] / max_possible_score))

        # Обновляем статистику
        if results['is_suitable']:
            self.stats['successful_matches'] += 1
        else:
            self.stats['failed_matches'] += 1

        if results['match_percentage'] > 0 and results['match_percentage'] < 100:
            self.stats['partial_matches'] += 1

        return results

    def _get_category_rule(self, product_category: str) -> Optional[Dict[str, Any]]:
        """Определяет правило для категории товара"""

        category_lower = product_category.lower()

        for rule_name, rule in self.category_rules.items():
            patterns = rule.get('category_patterns', [])
            if any(pattern in category_lower for pattern in patterns):
                return rule

        return None

    def _match_characteristic(self, char: Dict[str, Any],
                            product_attrs: List[Dict[str, str]],
                            category_rule: Optional[Dict[str, Any]] = None,
                            product_category: str = '') -> MatchResult:
        """Сопоставляет одну характеристику с учетом правил категории"""

        char_name = char.get('name', '').strip()
        char_value = char.get('value', '').strip()
        char_type = char.get('type', 'Качественная')

        if not char_name:
            return MatchResult(
                matched=False,
                score=0.0,
                confidence=0.0,
                matched_attribute=None,
                reason="Пустое название характеристики"
            )

        # 1. Ищем подходящий атрибут
        attr_match = self._find_best_attribute_match(char_name, product_attrs)

        if not attr_match['attribute']:
            return MatchResult(
                matched=False,
                score=0.0,
                confidence=attr_match['confidence'],
                matched_attribute=None,
                reason=f"Атрибут '{char_name}' не найден в товаре"
            )

        # 2. Сравниваем значения
        best_attr = attr_match['attribute']
        attr_value = best_attr.get('attr_value', '').strip()

        # Проверяем умные правила для категории
        if category_rule and self.smart_matching_enabled:
            smart_result = self._apply_smart_rules(
                char_name, char_value, attr_value,
                category_rule, product_category
            )

            if smart_result:
                return smart_result

        # Стандартная проверка
        if char_type == 'Количественная':
            value_match = self._match_numeric_value(char_value, attr_value, char_name)
        else:
            value_match = self._match_categorical_value(char_value, attr_value)

        # 3. Комбинируем результаты
        combined_confidence = attr_match['confidence'] * value_match['confidence']

        if value_match['matched']:
            return MatchResult(
                matched=True,
                score=value_match['score'],
                confidence=combined_confidence,
                matched_attribute=best_attr,
                reason=value_match['reason'],
                details={
                    'name_match_confidence': attr_match['confidence'],
                    'value_match_confidence': value_match['confidence'],
                    'match_type': attr_match['match_type']
                }
            )
        else:
            return MatchResult(
                matched=False,
                score=0.0,
                confidence=combined_confidence,
                matched_attribute=best_attr,
                reason=value_match['reason']
            )

    def _apply_smart_rules(self, char_name: str, char_value: str,
                          attr_value: str, category_rule: Dict[str, Any],
                          product_category: str) -> Optional[MatchResult]:
        """Применяет умные правила для категории"""

        char_name_lower = char_name.lower()
        overrides = category_rule.get('attribute_overrides', {})

        # Проверяем, есть ли правило для этого атрибута
        for override_attr, rules in overrides.items():
            if override_attr in char_name_lower or char_name_lower in override_attr:

                # Проверяем допустимые значения
                if 'acceptable_values' in rules:
                    attr_value_lower = attr_value.lower()
                    for acceptable in rules['acceptable_values']:
                        if acceptable.lower() == attr_value_lower:
                            confidence = rules.get('confidence', 0.9)
                            return MatchResult(
                                matched=True,
                                score=0.9,  # Немного ниже точного совпадения
                                confidence=confidence,
                                matched_attribute={'attr_name': char_name, 'attr_value': attr_value},
                                reason=f"Принято для категории '{product_category}': '{char_value}' ≈ '{attr_value}'",
                                details={'smart_fix': True}
                            )

                # Проверяем маппинги значений
                if 'value_mappings' in rules:
                    char_value_lower = char_value.lower()
                    for expected, alternatives in rules['value_mappings'].items():
                        if char_value_lower == expected.lower():
                            attr_value_lower = attr_value.lower()
                            if any(alt.lower() == attr_value_lower for alt in alternatives):
                                confidence = rules.get('confidence', 0.95)
                                return MatchResult(
                                    matched=True,
                                    score=0.95,
                                    confidence=confidence,
                                    matched_attribute={'attr_name': char_name, 'attr_value': attr_value},
                                    reason=f"Умное соответствие: '{char_value}' = '{attr_value}'",
                                    details={'smart_fix': True}
                                )

                # Проверяем fuzzy matching
                if rules.get('fuzzy_match', False):
                    similarity = SequenceMatcher(None, char_value.lower(), attr_value.lower()).ratio()
                    if similarity >= 0.7:  # Пониженный порог для fuzzy
                        return MatchResult(
                            matched=True,
                            score=similarity,
                            confidence=similarity * rules.get('confidence', 0.8),
                            matched_attribute={'attr_name': char_name, 'attr_value': attr_value},
                            reason=f"Fuzzy match для категории: '{char_value}' ~ '{attr_value}'",
                            details={'smart_fix': True}
                        )

        return None

    # Остальные методы остаются без изменений
    def _find_best_attribute_match(self, char_name: str,
                                  product_attrs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Находит наиболее подходящий атрибут с учетом синонимов"""

        char_name_lower = char_name.lower().strip()
        best_match = {
            'attribute': None,
            'confidence': 0.0,
            'match_type': 'none'
        }

        # Проверяем каждый атрибут товара
        for attr in product_attrs:
            attr_name = attr.get('attr_name', '').lower().strip()

            if not attr_name:
                continue

            # 1. Точное совпадение
            if attr_name == char_name_lower:
                return {
                    'attribute': attr,
                    'confidence': 1.0,
                    'match_type': 'exact'
                }

            # 2. Проверяем через маппинги
            for key, synonyms in self.attribute_mappings.items():
                if char_name_lower == key or char_name_lower in synonyms:
                    if attr_name == key or attr_name in synonyms:
                        return {
                            'attribute': attr,
                            'confidence': 0.95,
                            'match_type': 'synonym'
                        }

            # 3. Проверяем вхождение подстроки
            if char_name_lower in attr_name or attr_name in char_name_lower:
                confidence = 0.85 if len(char_name_lower) > 3 else 0.7
                if confidence > best_match['confidence']:
                    best_match = {
                        'attribute': attr,
                        'confidence': confidence,
                        'match_type': 'substring'
                    }

            # 4. Fuzzy matching
            similarity = SequenceMatcher(None, char_name_lower, attr_name).ratio()
            if similarity >= self.thresholds['name_match'] and similarity > best_match['confidence']:
                best_match = {
                    'attribute': attr,
                    'confidence': similarity,
                    'match_type': 'fuzzy'
                }

        return best_match

    def _match_numeric_value(self, tender_value: str, product_value: str,
                           attr_name: str = '') -> Dict[str, Any]:
        """Сопоставляет числовые значения с учетом единиц измерения"""

        # Извлекаем числа и единицы измерения
        tender_parsed = self._parse_numeric_value(tender_value)
        product_parsed = self._parse_numeric_value(product_value)

        if not tender_parsed['numbers'] or not product_parsed['numbers']:
            return {
                'matched': False,
                'score': 0.0,
                'confidence': 0.0,
                'reason': f"Не удалось извлечь числа из '{tender_value}' или '{product_value}'"
            }

        # Приводим к одной единице измерения если нужно
        if tender_parsed['unit'] and product_parsed['unit']:
            product_parsed = self._convert_units(product_parsed, tender_parsed['unit'])

        # Определяем тип сравнения
        tender_num = tender_parsed['numbers'][0]
        product_num = product_parsed['numbers'][0]

        # Проверяем операторы
        if tender_parsed['operator'] == 'min':
            matched = product_num >= tender_num
            reason = f"{product_num} {'>=' if matched else '<'} {tender_num}"

        elif tender_parsed['operator'] == 'max':
            matched = product_num <= tender_num
            reason = f"{product_num} {'<=' if matched else '>'} {tender_num}"

        elif tender_parsed['operator'] == 'range' and len(tender_parsed['numbers']) == 2:
            min_val, max_val = tender_parsed['numbers']
            matched = min_val <= product_num <= max_val
            reason = f"{product_num} {'в' if matched else 'не в'} диапазоне [{min_val}, {max_val}]"

        else:
            # Точное совпадение с допуском
            tolerance = self.thresholds['numeric_tolerance']
            if attr_name.lower() in ['количество', 'количество листов']:
                # Для количества используем точное совпадение или больше
                matched = product_num >= tender_num
                reason = f"{product_num} {'>=' if matched else '<'} {tender_num}"
            else:
                # Для других - допуск ±10%
                diff = abs(product_num - tender_num) / tender_num if tender_num != 0 else 0
                matched = diff <= tolerance
                reason = f"{product_num} {'≈' if matched else '!='} {tender_num} (откл. {diff*100:.1f}%)"

        # Уверенность зависит от точности совпадения
        if matched:
            if tender_parsed['operator'] in ['min', 'max', 'range']:
                confidence = 0.95
            else:
                # Чем ближе значения, тем выше уверенность
                diff_ratio = abs(product_num - tender_num) / tender_num if tender_num != 0 else 0
                confidence = max(0.5, 1.0 - diff_ratio)
        else:
            confidence = 0.8  # Высокая уверенность в несовпадении

        return {
            'matched': matched,
            'score': 1.0 if matched else 0.0,
            'confidence': confidence,
            'reason': reason
        }

    def _match_categorical_value(self, tender_value: str, product_value: str) -> Dict[str, Any]:
        """Сопоставляет категориальные значения"""

        tender_val_lower = tender_value.lower().strip()
        product_val_lower = product_value.lower().strip()

        # 1. Точное совпадение
        if tender_val_lower == product_val_lower:
            return {
                'matched': True,
                'score': 1.0,
                'confidence': 1.0,
                'reason': f"Точное совпадение: '{product_value}'"
            }

        # 2. Проверяем маппинги значений
        for key, synonyms in self.value_mappings.items():
            if tender_val_lower == key or tender_val_lower in synonyms:
                if product_val_lower == key or product_val_lower in synonyms:
                    return {
                        'matched': True,
                        'score': 0.95,
                        'confidence': 0.95,
                        'reason': f"Синонимы: '{tender_value}' = '{product_value}'"
                    }

        # 3. Проверяем вхождение
        if len(tender_val_lower) > 3:  # Для коротких строк не проверяем
            if tender_val_lower in product_val_lower:
                return {
                    'matched': True,
                    'score': 0.85,
                    'confidence': 0.85,
                    'reason': f"'{tender_value}' входит в '{product_value}'"
                }
            if product_val_lower in tender_val_lower:
                return {
                    'matched': True,
                    'score': 0.85,
                    'confidence': 0.85,
                    'reason': f"'{product_value}' входит в '{tender_value}'"
                }

        # 4. Fuzzy matching
        similarity = SequenceMatcher(None, tender_val_lower, product_val_lower).ratio()
        if similarity >= self.thresholds['value_match']:
            return {
                'matched': True,
                'score': similarity,
                'confidence': similarity,
                'reason': f"Схожие значения: '{tender_value}' ~ '{product_value}' ({similarity:.0%})"
            }

        return {
            'matched': False,
            'score': 0.0,
            'confidence': 0.9,  # Высокая уверенность в несовпадении
            'reason': f"Не совпадает: '{tender_value}' != '{product_value}'"
        }

    def _parse_numeric_value(self, value: str) -> Dict[str, Any]:
        """Парсит числовое значение, извлекая числа, оператор и единицы измерения"""

        value_clean = value.lower().strip()
        result = {
            'numbers': [],
            'operator': 'exact',
            'unit': None
        }

        # Убираем пробелы между цифрами
        value_clean = re.sub(r'(\d)\s+(\d)', r'\1\2', value_clean)

        # Определяем оператор
        # Проверяем диапазон
        range_match = re.search(r'от\s*(\d+\.?\d*)\s*до\s*(\d+\.?\d*)', value_clean)
        if range_match:
            result['numbers'] = [float(range_match.group(1)), float(range_match.group(2))]
            result['operator'] = 'range'
        else:
            # Проверяем минимум
            min_match = re.search(r'(?:от|более|больше|свыше|минимум|не менее|≥|>=|>)\s*(\d+\.?\d*)', value_clean)
            if min_match:
                result['numbers'] = [float(min_match.group(1))]
                result['operator'] = 'min'
            else:
                # Проверяем максимум
                max_match = re.search(r'(?:до|менее|меньше|максимум|не более|≤|<=|<)\s*(\d+\.?\d*)', value_clean)
                if max_match:
                    result['numbers'] = [float(max_match.group(1))]
                    result['operator'] = 'max'
                else:
                    # Извлекаем все числа
                    numbers = re.findall(r'\d+\.?\d*', value_clean)
                    if numbers:
                        result['numbers'] = [float(numbers[0])]

        # Извлекаем единицу измерения
        units = ['гб', 'gb', 'мб', 'mb', 'гбайт', 'мбайт', 'тб', 'кг', 'г', 'см', 'мм', 'м']
        for unit in units:
            if unit in value_clean:
                result['unit'] = unit
                break

        return result

    def _convert_units(self, value_dict: Dict[str, Any], target_unit: str) -> Dict[str, Any]:
        """Конвертирует единицы измерения"""

        if not value_dict['unit'] or value_dict['unit'] == target_unit:
            return value_dict

        conversions = self.unit_conversions

        # Прямая конверсия
        if value_dict['unit'] in conversions:
            conv = conversions[value_dict['unit']]
            if conv['to'] == target_unit:
                value_dict['numbers'] = [n * conv['factor'] for n in value_dict['numbers']]
                value_dict['unit'] = target_unit

        # Обратная конверсия
        for unit, conv in conversions.items():
            if conv['to'] == value_dict['unit'] and unit == target_unit:
                value_dict['numbers'] = [n / conv['factor'] for n in value_dict['numbers']]
                value_dict['unit'] = target_unit
                break

        return value_dict

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику матчинга"""
        total = self.stats['total_matches']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'success_rate': self.stats['successful_matches'] / total,
            'failure_rate': self.stats['failed_matches'] / total,
            'partial_rate': self.stats['partial_matches'] / total,
            'smart_fix_rate': self.stats['smart_fixes'] / total if total > 0 else 0
        }