from elasticsearch import Elasticsearch
import json


class ElasticsearchService:
    """Сервис для работы с Elasticsearch"""

    def __init__(self, host="localhost", port=9200):
        self.host = host
        self.port = port
        self.es = None
        self.connect()

    def connect(self):
        """Подключение к ES"""
        try:
            self.es = Elasticsearch(
                [{"host": self.host, "port": self.port, "scheme": "http"}],
                verify_certs=False,
                ssl_show_warn=False,
                request_timeout=30
            )

            info = self.es.info()
            print(f"✅ ES подключен: {info['version']['number']}")
            return True

        except Exception as e:
            print(f"❌ Ошибка подключения к ES: {e}")
            return False

    def search_products(self, search_terms, size=None):
        """Поиск товаров по извлеченным терминам"""

        if not self.es:
            return {'error': 'ES не подключен'}

        if not self.es.indices.exists(index="ipointer_index"):
            return {'error': 'Индекс products не существует'}

        print(f"🔍 ES поиск:")
        print(f"   - Основной запрос: '{search_terms['search_query']}'")
        print(f"   - Boost терминов: {len(search_terms['boost_terms'])}")

        # Строим ES запрос
        query = self._build_elasticsearch_query(search_terms)

        # Добавляем size если указан
        if size:
            query['size'] = size

        # Указываем какие поля возвращать
        query['_source'] = ["title", "category", "brand", "attributes"]

        try:
            # Выполняем поиск
            response = self.es.search(index="ipointer_index", body=query)

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
                'max_score': response['hits']['max_score'],
                'search_terms_used': search_terms,
                'elasticsearch_query': query
            }

            print(f"✅ ES результат: найдено {len(candidates)} из {result['total_found']}")

            return result

        except Exception as e:
            print(f"❌ Ошибка ES поиска: {e}")
            return {'error': str(e)}

    def _build_elasticsearch_query(self, search_terms):
        """Строим ES запрос"""

        should_clauses = []

        # 1. Основной поисковый запрос (обязательный)
        if search_terms['search_query']:
            should_clauses.extend([
                # Точная фраза в названии - максимальный приоритет
                {
                    "match_phrase": {
                        "title": {
                            "query": search_terms['search_query'],
                            "boost": 10.0
                        }
                    }
                },
                # Отдельные слова в названии и категории
                {
                    "multi_match": {
                        "query": search_terms['search_query'],
                        "fields": ["title^5", "category^3"],
                        "type": "best_fields",
                        "boost": 7.0
                    }
                },
                # В атрибутах
                {
                    "nested": {
                        "path": "attributes",
                        "query": {
                            "multi_match": {
                                "query": search_terms['search_query'],
                                "fields": ["attributes.attr_name^2", "attributes.attr_value^3"]
                            }
                        },
                        "boost": 3.0
                    }
                }
            ])

        # 2. Термины с индивидуальными весами
        for term, weight in search_terms['boost_terms'].items():
            should_clauses.extend([
                # В основных полях
                {
                    "multi_match": {
                        "query": term,
                        "fields": ["title^3", "category^2", "brand"],
                        "boost": weight
                    }
                },
                # В атрибутах товара
                {
                    "nested": {
                        "path": "attributes",
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "match": {
                                            "attributes.attr_value": {
                                                "query": term,
                                                "boost": 2.0
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "attributes.attr_name": {
                                                "query": term,
                                                "boost": 1.0
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        "boost": weight * 0.8
                    }
                }
            ])

        # Финальный запрос
        query = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        }

        return query


# Тест сервиса
def test_elasticsearch_service():
    """Тест ES сервиса"""
    es_service = ElasticsearchService()

    if es_service.es:
        # Тестовые термины
        test_terms = {
            'search_query': 'блоки записей',
            'boost_terms': {
                'пастельный': 3.0,
                'клейким': 2.6,
                'краем': 2.2
            }
        }

        result = es_service.search_products(test_terms, size=5)

        if 'candidates' in result:
            print(f"\n📋 НАЙДЕНО {len(result['candidates'])} ТОВАРОВ:")
            for i, candidate in enumerate(result['candidates'], 1):
                print(f"{i}. {candidate['title']}")
                print(f"   Категория: {candidate['category']}")
                print(f"   Релевантность: {candidate['elasticsearch_score']:.2f}")
        else:
            print(f"❌ Ошибка: {result.get('error')}")


if __name__ == "__main__":
    test_elasticsearch_service()