from typing import Dict, List, Any

from app.core.constants import ES_FIELD_MULTIPLIERS


class ElasticsearchQueryBuilder:
    """Построение оптимизированных запросов для поиска товаров"""

    def __init__(self, field_multipliers: Dict[str, float] = None):
        self.field_multipliers = field_multipliers or ES_FIELD_MULTIPLIERS

    def build_tender_query(self, search_terms: Dict) -> Dict:
        """Строит полный запрос для поиска по тендеру"""

        query = {
            "query": {
                "bool": {
                    "must": self._build_must_clauses(search_terms),
                    "should": self._build_should_clauses(search_terms),
                    "minimum_should_match": 0
                }
            },
            "sort": [
                {"_score": {"order": "desc"}}
            ]
        }

        return query

    def _build_must_clauses(self, search_terms: Dict) -> List[Dict]:
        """Строит обязательные условия поиска"""

        must_clauses = []

        if not search_terms.get('search_query'):
            return must_clauses

        # Получаем все термины для поиска
        all_search_terms = search_terms.get('must_match_terms', [])
        if not all_search_terms:
            all_search_terms = search_terms['search_query'].split()

        # Создаем составное условие
        must_clauses.append({
            "bool": {
                "should": [
                    # Точная фраза в названии (высший приоритет)
                    {
                        "match_phrase": {
                            "title": {
                                "query": search_terms['search_query'],
                                "boost": 2.0
                            }
                        }
                    },
                    # Любое слово в названии
                    {
                        "match": {
                            "title": {
                                "query": ' '.join(all_search_terms),
                                "operator": "or",
                                "boost": 1.0
                            }
                        }
                    },
                    # В категории
                    {
                        "match": {
                            "category": {
                                "query": ' '.join(all_search_terms),
                                "operator": "or",
                                "boost": 1.0
                            }
                        }
                    },
                    # В атрибутах (как запасной вариант)
                    {
                        "nested": {
                            "path": "attributes",
                            "query": {
                                "multi_match": {
                                    "query": ' '.join(all_search_terms),
                                    "fields": [
                                        "attributes.attr_name",
                                        "attributes.attr_value"
                                    ],
                                    "operator": "or"
                                }
                            },
                            "boost": 0.8
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        })

        return must_clauses

    def _build_should_clauses(self, search_terms: Dict) -> List[Dict]:
        """Строит желательные условия с весами"""

        should_clauses = []
        boost_terms = search_terms.get('boost_terms', {})

        for term, weight in boost_terms.items():
            # Добавляем условия для каждого поля
            should_clauses.extend(self._create_term_clauses(term, weight))

        return should_clauses

    def _create_term_clauses(self, term: str, weight: float) -> List[Dict]:
        """Создает условия поиска для одного термина"""

        clauses = []

        # В названии товара
        clauses.append({
            "match": {
                "title": {
                    "query": term,
                    "boost": weight * self.field_multipliers['title']
                }
            }
        })

        # В категории
        clauses.append({
            "match": {
                "category": {
                    "query": term,
                    "boost": weight * self.field_multipliers['category']
                }
            }
        })

        # В бренде
        clauses.append({
            "match": {
                "brand": {
                    "query": term,
                    "boost": weight * self.field_multipliers['brand']
                }
            }
        })

        # В значениях атрибутов
        clauses.append({
            "nested": {
                "path": "attributes",
                "query": {
                    "match": {
                        "attributes.attr_value": {
                            "query": term,
                            "boost": weight * self.field_multipliers['attr_value']
                        }
                    }
                }
            }
        })

        # В названиях атрибутов
        clauses.append({
            "nested": {
                "path": "attributes",
                "query": {
                    "match": {
                        "attributes.attr_name": {
                            "query": term,
                            "boost": weight * self.field_multipliers['attr_name']
                        }
                    }
                }
            }
        })

        return clauses

    def add_filters(self, query: Dict, filters: Dict) -> Dict:
        """Добавляет фильтры к запросу"""

        if not filters:
            return query

        # Добавляем фильтры в секцию filter
        if 'query' not in query:
            query['query'] = {'bool': {}}

        if 'bool' not in query['query']:
            query['query']['bool'] = {}

        if 'filter' not in query['query']['bool']:
            query['query']['bool']['filter'] = []

        # Добавляем каждый фильтр
        for field, value in filters.items():
            if isinstance(value, list):
                # Множественный выбор
                query['query']['bool']['filter'].append({
                    "terms": {field: value}
                })
            else:
                # Одиночное значение
                query['query']['bool']['filter'].append({
                    "term": {field: value}
                })

        return query