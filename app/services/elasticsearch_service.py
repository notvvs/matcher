from elasticsearch import Elasticsearch
import json

from app.config.settings import settings


class ElasticsearchService:
    """Сервис ES с оптимальными запросами для тендеров"""

    def __init__(self, host=None, port=None):
        self.host = host or settings.ELASTICSEARCH_HOST
        self.port = port or settings.ELASTICSEARCH_PORT
        self.index_name = settings.ELASTICSEARCH_INDEX
        self.es = None
        self.connect()

    def connect(self):
        """Подключение"""
        try:
            self.es = Elasticsearch(**settings.get_elasticsearch_config())

            info = self.es.info()
            print(f"✅ ES подключен: {info['version']['number']}")
            return True

        except Exception as e:
            print(f"❌ Ошибка подключения к ES: {e}")
            return False

    def search_products(self, search_terms, size=None):
        """ОПТИМАЛЬНЫЙ поиск товаров"""

        if not self.es:
            return {'error': 'ES не подключен'}

        if not self.es.indices.exists(index=self.index_name):
            return {'error': f'Индекс {self.index_name} не существует'}

        print(f"🎯 ОПТИМАЛЬНЫЙ ES поиск:")
        print(f"   - Основной запрос: '{search_terms['search_query']}'")
        print(f"   - Обязательные термины: {search_terms.get('must_match_terms', [])}")
        print(f"   - Boost терминов: {len(search_terms['boost_terms'])}")

        # Строим ОПТИМАЛЬНЫЙ ES запрос
        query = self._build_optimal_elasticsearch_query(search_terms)

        # Параметры запроса
        if size:
            query['size'] = size
        query['_source'] = ["title", "category", "brand", "attributes"]

        try:
            response = self.es.search(index=self.index_name, body=query)

            # Обрабатываем результаты
            candidates = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                candidates.append({
                    'id': hit['_id'],
                    'title': source.get('title', 'Без названия'),
                    'category': source.get('category', 'Без категории'),
                    'brand': source.get('brand', ''),
                    'elasticsearch_score': hit['_score'],
                    'attributes': source.get('attributes', [])
                })

            result = {
                'candidates': candidates,
                'total_found': response['hits']['total']['value'],
                'max_score': response['hits']['max_score'] or 0,  # Защита от None
                'search_terms_used': search_terms,
                'query_type': 'optimal_tender_search'
            }

            print(f"✅ ОПТИМАЛЬНЫЙ результат: {len(candidates)} из {result['total_found']}")
            if result['max_score'] > 0:
                print(f"   - Макс. релевантность: {result['max_score']:.2f}")
            else:
                print(f"   - Нет результатов с релевантностью")

            return result

        except Exception as e:
            print(f"❌ Ошибка ES поиска: {e}")
            return {'error': str(e)}

    def _build_optimal_elasticsearch_query(self, search_terms):
        """🎯 ОПТИМАЛЬНЫЙ ES запрос для тендеров"""

        # 1. ОБЯЗАТЕЛЬНЫЕ УСЛОВИЯ (MUST)
        must_clauses = []

        # Основной поисковый запрос ОБЯЗАТЕЛЕН
        if search_terms['search_query']:
            must_clauses.append({
                "bool": {
                    "should": [
                        # Точная фраза в названии
                        {
                            "match_phrase": {
                                "title": {
                                    "query": search_terms['search_query'],
                                    "boost": 1.0
                                }
                            }
                        },
                        # Все слова в названии
                        {
                            "match": {
                                "title": {
                                    "query": search_terms['search_query'],
                                    "operator": "and",  # ВСЕ слова
                                    "boost": 1.0
                                }
                            }
                        },
                        # В категории
                        {
                            "match": {
                                "category": {
                                    "query": search_terms['search_query'],
                                    "boost": 1.0
                                }
                            }
                        },
                        # В атрибутах (fallback)
                        {
                            "nested": {
                                "path": "attributes",
                                "query": {
                                    "multi_match": {
                                        "query": search_terms['search_query'],
                                        "fields": ["attributes.attr_name", "attributes.attr_value"]
                                    }
                                },
                                "boost": 0.8
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            })

        # 2. ЖЕЛАТЕЛЬНЫЕ УСЛОВИЯ С ВЕСАМИ (SHOULD)
        should_clauses = []

        # Термины с индивидуальными весами
        multipliers = settings.WEIGHTS['es_field_multipliers']

        for term, weight in search_terms['boost_terms'].items():
            should_clauses.extend([
                # В названии товара (ВЫСШИЙ приоритет для точных совпадений)
                {
                    "match": {
                        "title": {
                            "query": term,
                            "boost": weight * multipliers['title']
                        }
                    }
                },
                # В категории
                {
                    "match": {
                        "category": {
                            "query": term,
                            "boost": weight * multipliers['category']
                        }
                    }
                },
                # В бренде
                {
                    "match": {
                        "brand": {
                            "query": term,
                            "boost": weight * multipliers['brand']
                        }
                    }
                },
                # В значениях атрибутов (ОЧЕНЬ ВАЖНО для характеристик)
                {
                    "nested": {
                        "path": "attributes",
                        "query": {
                            "match": {
                                "attributes.attr_value": {
                                    "query": term,
                                    "boost": weight * multipliers['attr_value']
                                }
                            }
                        }
                    }
                },
                # В названиях атрибутов
                {
                    "nested": {
                        "path": "attributes",
                        "query": {
                            "match": {
                                "attributes.attr_name": {
                                    "query": term,
                                    "boost": weight * multipliers['attr_name']
                                }
                            }
                        }
                    }
                }
            ])

        # 3. ФИНАЛЬНЫЙ ОПТИМАЛЬНЫЙ ЗАПРОС
        query = {
            "query": {
                "bool": {
                    "must": must_clauses,  # ОБЯЗАТЕЛЬНЫЕ (без этого товар не попадает)
                    "should": should_clauses,  # Желательные с весами (ранжирование)
                    "minimum_should_match": 0  # Для should клауз
                }
            },
            # Точная сортировка по релевантности
            "sort": [
                {"_score": {"order": "desc"}}
            ]
        }

        return query