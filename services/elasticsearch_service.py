from elasticsearch import Elasticsearch
import json


class ElasticsearchService:
    """Сервис ES с оптимальными запросами для тендеров"""

    def __init__(self, host="localhost", port=9200):
        self.host = host
        self.port = port
        self.es = None
        self.connect()

    def connect(self):
        """Подключение"""
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
        """ОПТИМАЛЬНЫЙ поиск товаров"""

        if not self.es:
            return {'error': 'ES не подключен'}

        if not self.es.indices.exists(index="ipointer_index"):
            return {'error': 'Индекс products не существует'}

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
                'query_type': 'optimal_tender_search'
            }

            print(f"✅ ОПТИМАЛЬНЫЙ результат: {len(candidates)} из {result['total_found']}")
            print(f"   - Макс. релевантность: {result['max_score']:.2f}")

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
        for term, weight in search_terms['boost_terms'].items():
            should_clauses.extend([
                # В названии товара (ВЫСШИЙ приоритет для точных совпадений)
                {
                    "match": {
                        "title": {
                            "query": term,
                            "boost": weight * 2.5  # Максимальный множитель для названия
                        }
                    }
                },
                # В категории
                {
                    "match": {
                        "category": {
                            "query": term,
                            "boost": weight * 1.8
                        }
                    }
                },
                # В бренде
                {
                    "match": {
                        "brand": {
                            "query": term,
                            "boost": weight * 1.5
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
                                    "boost": weight * 2.0  # Высокий приоритет для значений
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
                                    "boost": weight * 1.0  # Стандартный вес для названий
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

    def create_test_index_with_data(self):
        """Создание тестового индекса с данными"""

        if not self.es:
            print("❌ ES не подключен")
            return False

        # Удаляем старый индекс
        if self.es.indices.exists(index="products"):
            self.es.indices.delete(index="products")
            print("🗑️ Старый индекс удален")

        # Создаем индекс
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "russian_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "category": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "brand": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "attributes": {
                        "type": "nested",
                        "properties": {
                            "attr_name": {
                                "type": "text",
                                "analyzer": "russian_analyzer",
                                "fields": {"keyword": {"type": "keyword"}}
                            },
                            "attr_value": {
                                "type": "text",
                                "analyzer": "russian_analyzer",
                                "fields": {"keyword": {"type": "keyword"}}
                            }
                        }
                    }
                }
            }
        }

        self.es.indices.create(index="products", body=index_settings)
        print("✅ Индекс создан")

        # Добавляем тестовые товары
        test_products = [
            {
                "title": "Липкие блоки Post-it 76x76мм пастельных цветов 400 листов",
                "category": "Бумага для заметок",
                "brand": "3M",
                "attributes": [
                    {"attr_name": "Цвет бумаги", "attr_value": "Пастельный"},
                    {"attr_name": "Размер", "attr_value": "76x76 мм"},
                    {"attr_name": "Количество листов", "attr_value": "400"},
                    {"attr_name": "Тип", "attr_value": "С клейким краем"}
                ]
            },
            {
                "title": "Блоки для записей Attache пастельные 75x75мм 100 листов",
                "category": "Канцелярские товары",
                "brand": "Attache",
                "attributes": [
                    {"attr_name": "Цвет бумаги", "attr_value": "Пастельный"},
                    {"attr_name": "Размер", "attr_value": "75x75 мм"},
                    {"attr_name": "Количество листов", "attr_value": "100"},
                    {"attr_name": "Тип", "attr_value": "С клейким краем"}
                ]
            },
            {
                "title": "Стикеры разноцветные для записей 50x50мм",
                "category": "Бумага для заметок",
                "brand": "Office",
                "attributes": [
                    {"attr_name": "Цвет", "attr_value": "Разноцветный"},
                    {"attr_name": "Размер", "attr_value": "50x50 мм"},
                    {"attr_name": "Тип", "attr_value": "Клейкий край"}
                ]
            },
            {
                "title": "Штемпель автоматический Colop пластиковый корпус",
                "category": "Штемпели и печати",
                "brand": "Colop",
                "attributes": [
                    {"attr_name": "Материал корпуса", "attr_value": "Пластик"},
                    {"attr_name": "Тип механизма", "attr_value": "Автоматический"}
                ]
            },
            {
                "title": "Печать штемпель самонаборный пластик автомат",
                "category": "Офисные печати",
                "brand": "Shiny",
                "attributes": [
                    {"attr_name": "Материал", "attr_value": "Пластик"},
                    {"attr_name": "Тип", "attr_value": "Автоматический"}
                ]
            }
        ]

        # Индексируем товары
        for i, product in enumerate(test_products):
            self.es.index(index="products", id=f"test_{i + 1}", body=product)

        self.es.indices.refresh(index="products")
        print(f"✅ Добавлено {len(test_products)} тестовых товаров")

        return True


# Тест
def test_optimal_service():
    """Тест оптимального сервиса"""
    es_service = ElasticsearchService()

    if es_service.es:
        # Создаем тестовые данные
        es_service.create_test_index_with_data()

        # Тестовые термины (как будут приходить от экстрактора)
        test_terms = {
            'search_query': 'блоки записей',
            'must_match_terms': ['блоки', 'записей', 'стикеры'],
            'boost_terms': {
                'пастельный': 4.0,  # Обязательная характеристика
                'клейким': 3.8,  # Обязательная характеристика
                'краем': 3.6,  # Обязательная характеристика
                'цвет': 1.8,  # Название характеристики
                'тип': 1.6  # Название характеристики
            }
        }

        result = es_service.search_products(test_terms, size=10)

        if 'candidates' in result:
            print(f"\n📋 НАЙДЕНО {len(result['candidates'])} ТОВАРОВ:")
            for i, candidate in enumerate(result['candidates'], 1):
                print(f"{i}. {candidate['title']}")
                print(f"   Категория: {candidate['category']}")
                print(f"   Релевантность: {candidate['elasticsearch_score']:.2f}")
                print()


if __name__ == "__main__":
    test_optimal_service()