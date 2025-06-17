from typing import Dict, List, Any, Tuple


class TenderValidator:
    """Валидатор тендеров"""

    @staticmethod
    def validate(tender: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Проверяет корректность структуры тендера

        Returns:
            (is_valid, list_of_errors)
        """

        errors = []

        # Проверка обязательных полей
        if not isinstance(tender, dict):
            errors.append("Тендер должен быть словарем")
            return False, errors

        # Проверка названия
        if 'name' not in tender:
            errors.append("Отсутствует поле 'name'")
        elif not tender['name'] or not isinstance(tender['name'], str):
            errors.append("Поле 'name' должно быть непустой строкой")

        # Проверка характеристик
        if 'characteristics' not in tender:
            errors.append("Отсутствует поле 'characteristics'")
        else:
            char_errors = TenderValidator._validate_characteristics(
                tender['characteristics']
            )
            errors.extend(char_errors)

        return len(errors) == 0, errors

    @staticmethod
    def _validate_characteristics(characteristics: Any) -> List[str]:
        """Проверяет характеристики тендера"""

        errors = []

        if not isinstance(characteristics, list):
            errors.append("Поле 'characteristics' должно быть списком")
            return errors

        if len(characteristics) == 0:
            errors.append("Список характеристик пуст")
            return errors

        # Проверяем наличие хотя бы одной обязательной характеристики
        has_required = False

        for i, char in enumerate(characteristics):
            if not isinstance(char, dict):
                errors.append(f"Характеристика {i} должна быть словарем")
                continue

            # Проверка обязательных полей характеристики
            if 'name' not in char:
                errors.append(f"Характеристика {i}: отсутствует поле 'name'")
            elif not char['name']:
                errors.append(f"Характеристика {i}: пустое поле 'name'")

            if 'value' not in char:
                errors.append(f"Характеристика {i}: отсутствует поле 'value'")
            elif not str(char['value']).strip():
                errors.append(f"Характеристика {i}: пустое поле 'value'")

            # Проверка типа
            if 'type' in char:
                valid_types = ['Качественная', 'Количественная']
                if char['type'] not in valid_types:
                    errors.append(
                        f"Характеристика {i}: неверный тип '{char['type']}'. "
                        f"Допустимые: {valid_types}"
                    )

            # Проверка флага обязательности
            if char.get('required', False):
                has_required = True

        if not has_required:
            errors.append("Должна быть хотя бы одна обязательная характеристика")

        return errors


class ProductValidator:
    """Валидатор товаров"""

    @staticmethod
    def validate(product: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Проверяет корректность структуры товара

        Returns:
            (is_valid, list_of_errors)
        """

        errors = []

        if not isinstance(product, dict):
            errors.append("Товар должен быть словарем")
            return False, errors

        # Проверка обязательных полей
        if 'id' not in product:
            errors.append("Отсутствует поле 'id'")

        if 'title' not in product:
            errors.append("Отсутствует поле 'title'")
        elif not product['title']:
            errors.append("Поле 'title' не должно быть пустым")

        # Проверка атрибутов
        if 'attributes' in product:
            attr_errors = ProductValidator._validate_attributes(
                product['attributes']
            )
            errors.extend(attr_errors)

        return len(errors) == 0, errors

    @staticmethod
    def _validate_attributes(attributes: Any) -> List[str]:
        """Проверяет атрибуты товара"""

        errors = []

        if not isinstance(attributes, list):
            errors.append("Поле 'attributes' должно быть списком")
            return errors

        for i, attr in enumerate(attributes):
            if not isinstance(attr, dict):
                errors.append(f"Атрибут {i} должен быть словарем")
                continue

            # Проверка обязательных полей атрибута
            if 'attr_name' not in attr:
                errors.append(f"Атрибут {i}: отсутствует поле 'attr_name'")

            if 'attr_value' not in attr:
                errors.append(f"Атрибут {i}: отсутствует поле 'attr_value'")

        return errors


class SearchTermsValidator:
    """Валидатор поисковых терминов"""

    @staticmethod
    def validate(search_terms: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Проверяет корректность поисковых терминов"""

        errors = []

        if not isinstance(search_terms, dict):
            errors.append("search_terms должен быть словарем")
            return False, errors

        # Проверка обязательных полей
        required_fields = ['search_query', 'boost_terms', 'must_match_terms']

        for field in required_fields:
            if field not in search_terms:
                errors.append(f"Отсутствует обязательное поле '{field}'")

        # Проверка типов
        if 'search_query' in search_terms:
            if not isinstance(search_terms['search_query'], str):
                errors.append("Поле 'search_query' должно быть строкой")

        if 'boost_terms' in search_terms:
            if not isinstance(search_terms['boost_terms'], dict):
                errors.append("Поле 'boost_terms' должно быть словарем")
            else:
                # Проверяем, что все веса - числа
                for term, weight in search_terms['boost_terms'].items():
                    if not isinstance(weight, (int, float)):
                        errors.append(
                            f"Вес для термина '{term}' должен быть числом"
                        )

        if 'must_match_terms' in search_terms:
            if not isinstance(search_terms['must_match_terms'], list):
                errors.append("Поле 'must_match_terms' должно быть списком")

        return len(errors) == 0, errors