import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import re
from collections import defaultdict
import logging
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "all_products"  # Измените на вашу БД
COLLECTION_NAME = "products"  # Измените на вашу коллекцию

ELASTICSEARCH_HOST = "http://localhost:9200"
ES_INDEX_NAME = "all_products"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "all_products"

# Модели
EMBEDDING_MODEL = 'ai-forever/sbert_large_nlu_ru'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # Или русская модель если есть


@dataclass
class SearchConfig:
    """Конфигурация поиска"""
    # Веса для RRF
    elastic_weight: float = 0.5
    vector_weight: float = 0.5

    # Параметры RRF - УМЕНЬШАЕМ K для более высоких скоров
    rrf_k: int = 10  # Было 60

    # Количество кандидатов
    candidates_multiplier: int = 5  # Увеличиваем для лучшего покрытия

    # Параметры ElasticSearch
    fuzziness: str = "AUTO"
    prefix_length: int = 2

    # Boosting для полей в ES
    title_boost: float = 3.0
    description_boost: float = 1.0
    category_boost: float = 1.5
    brand_boost: float = 2.0
    attributes_boost: float = 1.5  # Увеличиваем важность атрибутов

    # Использовать cross-encoder для переранжирования
    use_cross_encoder: bool = True
    cross_encoder_top_k: int = 30  # Увеличиваем


@dataclass
class SearchResult:
    """Результат поиска"""
    id: str
    title: str
    score: float
    elastic_score: float = 0.0
    vector_score: float = 0.0
    cross_encoder_score: float = 0.0
    debug_info: Dict = field(default_factory=dict)


class UniversalHybridSearch:
    """Универсальная гибридная поисковая система"""

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()

        # Инициализация клиентов
        try:
            self.es = Elasticsearch([ELASTICSEARCH_HOST])
            # Проверяем подключение
            if not self.es.ping():
                raise Exception("ElasticSearch не отвечает")
            logger.info("✅ Подключение к ElasticSearch успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к ElasticSearch: {e}")
            logger.error(f"Убедитесь, что ElasticSearch запущен на {ELASTICSEARCH_HOST}")
            raise

        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.mongo = MongoClient(MONGODB_URI)[DATABASE_NAME][COLLECTION_NAME]

        # Инициализация моделей
        logger.info("Загрузка моделей...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        if self.config.use_cross_encoder:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        # Проверка/создание индексов
        self._setup_elasticsearch()

    def _parse_query_attributes(self, query: str) -> Dict:
        """Извлечение атрибутов из запроса"""
        attributes = {}

        # Паттерны для извлечения характеристик
        patterns = {
            'diameter': r'(\d+(?:\.\d+)?)\s*мм',
            'color': r'(синий|черный|красный|зеленый|желтый|белый|бесцветный|прозрачный)',
            'sheets': r'(\d+)\s*(?:листов|лист|л\.)',
            'format': r'(A\d|А\d)',
            'quantity': r'(?:не менее|от|≥)\s*(\d+)',
            'type': r'тип\s+(\d+|[а-яА-Я]+)',
        }

        query_lower = query.lower()

        for attr_name, pattern in patterns.items():
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                attributes[attr_name] = match.group(1)

        return attributes

    def _setup_elasticsearch(self):
        """Настройка ElasticSearch с оптимальными настройками для русского языка"""

        # Проверяем версию ElasticSearch
        try:
            info = self.es.info()
            es_version = info['version']['number']
            logger.info(f"ElasticSearch версия: {es_version}")
        except Exception as e:
            logger.error(f"Ошибка получения информации о ElasticSearch: {e}")

        # Mapping для индекса
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "filter": {
                        "russian_stop": {
                            "type": "stop",
                            "stopwords": "_russian_"
                        },
                        "russian_stemmer": {
                            "type": "stemmer",
                            "language": "russian"
                        },
                        "edge_ngram_filter": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 20
                        }
                    },
                    "analyzer": {
                        "russian_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "russian_stop",
                                "russian_stemmer"
                            ]
                        },
                        "edge_ngram_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "edge_ngram_filter"
                            ]
                        },
                        "keyword_analyzer": {
                            "tokenizer": "keyword",
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
                        "fields": {
                            "exact": {
                                "type": "keyword"
                            },
                            "ngram": {
                                "type": "text",
                                "analyzer": "edge_ngram_analyzer",
                                "search_analyzer": "standard"
                            }
                        }
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "category": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "brand": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "article": {
                        "type": "keyword"
                    },
                    "attributes": {
                        "type": "nested",
                        "properties": {
                            "attr_name": {"type": "keyword"},
                            "attr_value": {"type": "text", "analyzer": "russian_analyzer"}
                        }
                    },
                    "all_text": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "min_price": {
                        "type": "float"
                    }
                }
            }
        }

        # Проверяем и создаем индекс
        try:
            # Пытаемся проверить существование индекса
            exists = False
            try:
                exists = self.es.indices.exists(index=ES_INDEX_NAME)
            except Exception as e:
                logger.warning(f"Ошибка при проверке индекса: {e}")
                # Пытаемся удалить проблемный индекс
                try:
                    self.es.indices.delete(index=ES_INDEX_NAME)
                    logger.info(f"Удален проблемный индекс {ES_INDEX_NAME}")
                except:
                    pass

            if not exists:
                logger.info(f"Создание индекса {ES_INDEX_NAME}...")
                self.es.indices.create(index=ES_INDEX_NAME, body=index_settings)
                logger.info(f"✅ Индекс {ES_INDEX_NAME} создан успешно")
            else:
                logger.info(f"Индекс {ES_INDEX_NAME} уже существует")

        except Exception as e:
            logger.error(f"Ошибка при создании индекса: {e}")
            # Попробуем создать упрощенный индекс
            try:
                simple_settings = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "description": {"type": "text"},
                            "category": {"type": "keyword"},
                            "brand": {"type": "keyword"},
                            "article": {"type": "keyword"},
                            "all_text": {"type": "text"},
                            "min_price": {"type": "float"}
                        }
                    }
                }
                self.es.indices.create(index=ES_INDEX_NAME, body=simple_settings)
                logger.warning("Создан упрощенный индекс без анализаторов")
            except Exception as e2:
                logger.error(f"Не удалось создать даже упрощенный индекс: {e2}")
                raise

    def index_products(self, batch_size: int = 100):
        """Индексация продуктов в ElasticSearch и Qdrant"""

        # Проверка Qdrant
        try:
            collection_info = self.qdrant.get_collection(QDRANT_COLLECTION)
            qdrant_exists = True
            logger.info(f"Коллекция Qdrant существует: {collection_info.points_count} точек")
        except:
            qdrant_exists = False
            logger.info("Создание коллекции Qdrant...")
            vector_size = self.embedder.get_sentence_embedding_dimension()
            self.qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        # Подсчет документов
        total_docs = self.mongo.count_documents({})
        logger.info(f"Всего документов для индексации: {total_docs}")

        # Индексация батчами
        es_actions = []
        qdrant_points = []
        texts_for_embedding = []

        with tqdm(total=total_docs, desc="Индексация") as pbar:
            for doc in self.mongo.find().batch_size(batch_size):
                # Подготовка данных для ElasticSearch
                es_doc = self._prepare_es_document(doc)
                es_actions.append({
                    "_index": ES_INDEX_NAME,
                    "_id": str(doc["_id"]),
                    "_source": es_doc
                })

                # Подготовка текста для эмбеддинга
                embedding_text = self._prepare_embedding_text(doc)
                texts_for_embedding.append(embedding_text)

                # Обработка батча
                if len(es_actions) >= batch_size:
                    # Индексация в ElasticSearch
                    helpers.bulk(self.es, es_actions)

                    # Создание эмбеддингов и индексация в Qdrant
                    if not qdrant_exists or True:  # Always update
                        embeddings = self.embedder.encode(
                            texts_for_embedding,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )

                        for i, (action, embedding) in enumerate(zip(es_actions, embeddings)):
                            qdrant_points.append(PointStruct(
                                id=abs(hash(action["_id"])) % (10 ** 8),
                                vector=embedding.tolist(),
                                payload={
                                    "mongodb_id": action["_id"],
                                    "title": action["_source"]["title"]
                                }
                            ))

                        self.qdrant.upsert(
                            collection_name=QDRANT_COLLECTION,
                            points=qdrant_points
                        )

                    pbar.update(len(es_actions))
                    es_actions = []
                    qdrant_points = []
                    texts_for_embedding = []

            # Последний батч
            if es_actions:
                helpers.bulk(self.es, es_actions)

                if not qdrant_exists or True:
                    embeddings = self.embedder.encode(
                        texts_for_embedding,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                    for i, (action, embedding) in enumerate(zip(es_actions, embeddings)):
                        qdrant_points.append(PointStruct(
                            id=abs(hash(action["_id"])) % (10 ** 8),
                            vector=embedding.tolist(),
                            payload={
                                "mongodb_id": action["_id"],
                                "title": action["_source"]["title"]
                            }
                        ))

                    self.qdrant.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=qdrant_points
                    )

                pbar.update(len(es_actions))

        # Обновление индекса
        self.es.indices.refresh(index=ES_INDEX_NAME)
        logger.info("Индексация завершена!")

    def _prepare_es_document(self, mongo_doc: Dict) -> Dict:
        """Подготовка документа для ElasticSearch"""

        # Базовые поля
        es_doc = {
            "title": mongo_doc.get("title", ""),
            "description": mongo_doc.get("description", ""),
            "category": mongo_doc.get("category", ""),
            "brand": mongo_doc.get("brand", ""),
            "article": mongo_doc.get("article", "")
        }

        # Атрибуты
        if "attributes" in mongo_doc:
            es_doc["attributes"] = mongo_doc["attributes"]

        # Минимальная цена
        min_price = self._extract_min_price(mongo_doc)
        if min_price:
            es_doc["min_price"] = min_price

        # Объединенное текстовое поле для полнотекстового поиска
        all_text_parts = [
            es_doc["title"],
            es_doc["description"],
            es_doc["category"],
            es_doc["brand"]
        ]

        # Добавляем атрибуты в общий текст
        if "attributes" in mongo_doc:
            for attr in mongo_doc["attributes"]:
                if attr.get("attr_value") and attr["attr_value"] != "Нет данных":
                    all_text_parts.append(f"{attr.get('attr_name', '')}: {attr['attr_value']}")

        es_doc["all_text"] = " ".join(filter(None, all_text_parts))

        return es_doc

    def _prepare_embedding_text(self, mongo_doc: Dict) -> str:
        """Подготовка текста для создания эмбеддинга с акцентом на важные атрибуты"""
        parts = []

        # Основные поля с весами (повторяем важные)
        if mongo_doc.get("title"):
            # Название - самое важное, повторяем 3 раза
            parts.extend([mongo_doc["title"]] * 3)

        if mongo_doc.get("description"):
            parts.append(mongo_doc["description"])

        if mongo_doc.get("category") and mongo_doc["category"] != "Нет данных":
            parts.append(f"Категория: {mongo_doc['category']}")
            parts.append(mongo_doc['category'])  # Дублируем для важности

        if mongo_doc.get("brand") and mongo_doc["brand"] != "Нет данных":
            parts.append(f"Бренд: {mongo_doc['brand']}")
            parts.append(mongo_doc['brand'])  # Дублируем

        # Важные атрибуты с усилением
        if "attributes" in mongo_doc:
            important_attrs = []
            for attr in mongo_doc["attributes"]:
                if attr.get("attr_value") and attr["attr_value"] != "Нет данных":
                    attr_text = f"{attr.get('attr_name', '')}: {attr['attr_value']}"
                    parts.append(attr_text)

                    # Особо важные атрибуты дублируем
                    attr_name_lower = attr.get('attr_name', '').lower()
                    if any(keyword in attr_name_lower for keyword in
                           ['цвет', 'размер', 'диаметр', 'формат', 'тип', 'количество']):
                        important_attrs.append(attr_text)

            # Добавляем важные атрибуты еще раз
            parts.extend(important_attrs)

        # Добавляем артикул если есть
        if mongo_doc.get("article") and mongo_doc["article"] != "Нет данных":
            parts.append(f"Артикул: {mongo_doc['article']}")

        return " ".join(parts)

    def _extract_min_price(self, mongo_doc: Dict) -> Optional[float]:
        """Извлечение минимальной цены"""
        min_price = None

        for supplier in mongo_doc.get("suppliers", []):
            for offer in supplier.get("supplier_offers", []):
                for price_item in offer.get("price", []):
                    if "price" in price_item and isinstance(price_item["price"], (int, float)):
                        if min_price is None or price_item["price"] < min_price:
                            min_price = float(price_item["price"])

        return min_price

    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Основной метод поиска"""

        # Параллельный поиск
        es_results = self._elasticsearch_search(query, top_k * self.config.candidates_multiplier, filters)
        vector_results = self._vector_search(query, top_k * self.config.candidates_multiplier)

        # RRF объединение
        combined_results = self._reciprocal_rank_fusion(es_results, vector_results)

        # Cross-encoder переранжирование (если включено)
        if self.config.use_cross_encoder and len(combined_results) > 0:
            combined_results = self._cross_encoder_rerank(query, combined_results[:self.config.cross_encoder_top_k])

        return combined_results[:top_k]

    def _elasticsearch_search(self, query: str, size: int, filters: Dict = None) -> List[Tuple[str, float]]:
        """Поиск через ElasticSearch с улучшенной обработкой атрибутов"""

        # Извлекаем атрибуты из запроса
        query_attributes = self._parse_query_attributes(query)

        # Автоматическое определение типа запроса
        query_lower = query.lower()

        # Если запрос - артикул
        if re.match(r'^[A-Za-z0-9\-]+$', query) and len(query) > 4:
            es_query = {
                "bool": {
                    "should": [
                        {"term": {"article": query}},
                        {"match": {"article": {"query": query, "fuzziness": "AUTO"}}}
                    ]
                }
            }
        else:
            # Создаем основной запрос
            should_clauses = [
                # Точное совпадение фразы
                {
                    "match_phrase": {
                        "title": {
                            "query": query,
                            "boost": self.config.title_boost * 2
                        }
                    }
                },
                # Поиск по всем словам
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            f"title^{self.config.title_boost}",
                            f"description^{self.config.description_boost}",
                            f"category^{self.config.category_boost}",
                            f"brand^{self.config.brand_boost}",
                            "all_text"
                        ],
                        "type": "best_fields",
                        "fuzziness": self.config.fuzziness,
                        "prefix_length": self.config.prefix_length
                    }
                }
            ]

            # Добавляем поиск по атрибутам если найдены
            if query_attributes:
                # Для цвета
                if 'color' in query_attributes:
                    color = query_attributes['color']
                    should_clauses.append({
                        "nested": {
                            "path": "attributes",
                            "query": {
                                "bool": {
                                    "should": [
                                        {"match": {"attributes.attr_value": color}},
                                        {"match": {"description": color}},
                                        {"match": {"title": color}}
                                    ]
                                }
                            },
                            "boost": 2.0
                        }
                    })

                # Для диаметра/размера
                if 'diameter' in query_attributes:
                    diameter = query_attributes['diameter']
                    should_clauses.extend([
                        {"match": {"title": f"{diameter}мм"}},
                        {"match": {"description": f"{diameter}мм"}},
                        {
                            "nested": {
                                "path": "attributes",
                                "query": {
                                    "match": {"attributes.attr_value": f"{diameter}"}
                                },
                                "boost": 1.5
                            }
                        }
                    ])

                # Для количества листов
                if 'sheets' in query_attributes:
                    sheets = query_attributes['sheets']
                    should_clauses.extend([
                        {"match": {"title": f"{sheets} лист"}},
                        {"match": {"description": f"{sheets} лист"}},
                        {
                            "nested": {
                                "path": "attributes",
                                "query": {
                                    "match": {"attributes.attr_value": sheets}
                                },
                                "boost": 1.5
                            }
                        }
                    ])

                # Для формата
                if 'format' in query_attributes:
                    format_value = query_attributes['format'].upper()
                    should_clauses.extend([
                        {"match": {"title": format_value}},
                        {"match": {"description": format_value}},
                        {
                            "nested": {
                                "path": "attributes",
                                "query": {
                                    "match": {"attributes.attr_value": format_value}
                                },
                                "boost": 2.0
                            }
                        }
                    ])

            es_query = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }

            # Автоматические негативные фильтры
            negative_patterns = self._detect_negative_patterns(query)
            if negative_patterns:
                es_query["bool"]["must_not"] = [
                    {"match": {"title": pattern}} for pattern in negative_patterns
                ]

        # Добавляем пользовательские фильтры
        if filters:
            if "bool" not in es_query:
                es_query = {"bool": {"must": [es_query]}}

            filter_clauses = []

            if "category" in filters:
                filter_clauses.append({"term": {"category.keyword": filters["category"]}})

            if "price_range" in filters:
                filter_clauses.append({
                    "range": {
                        "min_price": {
                            "gte": filters["price_range"][0],
                            "lte": filters["price_range"][1]
                        }
                    }
                })

            if filter_clauses:
                es_query["bool"]["filter"] = filter_clauses

        # Выполнение запроса
        response = self.es.search(
            index=ES_INDEX_NAME,
            body={
                "query": es_query,
                "size": size,
                "_source": ["title", "category", "brand"],
                "highlight": {
                    "fields": {
                        "title": {},
                        "description": {},
                        "attributes.attr_value": {}
                    }
                }
            }
        )

        # Логирование для отладки
        logger.debug(f"ES запрос для '{query}': найдено {response['hits']['total']['value']} документов")
        if query_attributes:
            logger.debug(f"Извлеченные атрибуты: {query_attributes}")

        # Возвращаем список (id, score) с нормализацией скоров
        results = []
        max_score = response['hits']['max_score'] or 1.0

        for hit in response["hits"]["hits"]:
            # Нормализуем скор от 0 до 1
            normalized_score = hit["_score"] / max_score if max_score > 0 else 0
            results.append((hit["_id"], normalized_score))

        return results

    def _vector_search(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """Векторный поиск через Qdrant"""

        # Создаем эмбеддинг запроса
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)

        # Поиск
        try:
            results = self.qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_embedding.tolist(),
                limit=limit
            ).points
        except Exception as e:
            logger.error(f"Ошибка векторного поиска: {e}")
            return []

        # Возвращаем список (id, score)
        return [(point.payload["mongodb_id"], point.score) for point in results]

    def _reciprocal_rank_fusion(self,
                                es_results: List[Tuple[str, float]],
                                vector_results: List[Tuple[str, float]]) -> List[SearchResult]:
        """Объединение результатов через RRF с улучшенной формулой"""

        from bson import ObjectId

        # Словарь для хранения скоров
        all_scores = defaultdict(lambda: {
            "es_rank": None,
            "vector_rank": None,
            "es_score": 0,
            "vector_score": 0,
            "es_normalized": 0,
            "vector_normalized": 0
        })

        # Нормализация скоров ElasticSearch
        if es_results:
            max_es_score = max(score for _, score in es_results[:10])  # Топ-10 для нормализации
            for rank, (doc_id, score) in enumerate(es_results):
                all_scores[doc_id]["es_rank"] = rank + 1
                all_scores[doc_id]["es_score"] = score
                all_scores[doc_id]["es_normalized"] = score / max_es_score if max_es_score > 0 else 0

        # Обработка векторных результатов (скоры уже от 0 до 1)
        for rank, (doc_id, score) in enumerate(vector_results):
            all_scores[doc_id]["vector_rank"] = rank + 1
            all_scores[doc_id]["vector_score"] = score
            all_scores[doc_id]["vector_normalized"] = score

        # Вычисление финальных скоров
        final_results = []

        for doc_id, scores in all_scores.items():
            # Комбинированная формула
            # 1. RRF компонент
            rrf_score = 0
            if scores["es_rank"] is not None:
                rrf_score += self.config.elastic_weight / (self.config.rrf_k + scores["es_rank"])
            if scores["vector_rank"] is not None:
                rrf_score += self.config.vector_weight / (self.config.rrf_k + scores["vector_rank"])

            # 2. Нормализованные скоры компонент
            norm_score = (
                    self.config.elastic_weight * scores["es_normalized"] +
                    self.config.vector_weight * scores["vector_normalized"]
            )

            # 3. Комбинированный финальный скор
            # Даем больше веса нормализованным скорам для более высоких значений
            final_score = 0.3 * rrf_score + 0.7 * norm_score

            # Бонус за присутствие в обоих результатах
            if scores["es_rank"] is not None and scores["vector_rank"] is not None:
                final_score *= 1.1

            # Получаем информацию о документе
            doc = None

            # Сначала пробуем найти в MongoDB по ObjectId
            try:
                doc = self.mongo.find_one({"_id": ObjectId(doc_id)})
            except Exception as e:
                logger.debug(f"Не удалось преобразовать {doc_id} в ObjectId: {e}")

            # Если не нашли в MongoDB, берем из ElasticSearch
            if not doc:
                try:
                    es_doc = self.es.get(index=ES_INDEX_NAME, id=doc_id)
                    if es_doc and '_source' in es_doc:
                        doc = es_doc['_source']
                        doc['_id'] = doc_id
                except Exception as e:
                    logger.debug(f"Не удалось получить документ {doc_id} из ES: {e}")
                    continue

            if doc:
                final_results.append(SearchResult(
                    id=doc_id,
                    title=doc.get("title", ""),
                    score=final_score,
                    elastic_score=scores["es_normalized"],
                    vector_score=scores["vector_normalized"],
                    debug_info={
                        "es_rank": scores["es_rank"],
                        "vector_rank": scores["vector_rank"],
                        "rrf_score": rrf_score,
                        "norm_score": norm_score
                    }
                ))

        # Сортировка по финальному скору
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results

    def _cross_encoder_rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Переранжирование с помощью cross-encoder"""
        from bson import ObjectId

        # Подготовка пар для cross-encoder
        pairs = []
        valid_results = []

        for result in results:
            # Получаем полный документ
            doc = None
            try:
                doc = self.mongo.find_one({"_id": ObjectId(result.id)})
            except:
                pass

            # Если не нашли в MongoDB, пробуем получить из ElasticSearch
            if not doc:
                try:
                    es_doc = self.es.get(index=ES_INDEX_NAME, id=result.id)
                    if es_doc and '_source' in es_doc:
                        doc = es_doc['_source']
                except:
                    pass

            if doc:
                # Создаем расширенный текст документа для лучшего сравнения
                doc_parts = [doc.get('title', '')]

                if doc.get('description'):
                    doc_parts.append(doc['description'])

                if doc.get('category'):
                    doc_parts.append(f"Категория: {doc['category']}")

                if doc.get('brand') and doc['brand'] != 'Нет данных':
                    doc_parts.append(f"Бренд: {doc['brand']}")

                # Добавляем важные атрибуты
                if 'attributes' in doc:
                    for attr in doc['attributes']:
                        if attr.get('attr_value') and attr['attr_value'] != 'Нет данных':
                            doc_parts.append(f"{attr.get('attr_name', '')}: {attr['attr_value']}")

                doc_text = " ".join(doc_parts)
                pairs.append([query, doc_text])
                valid_results.append(result)
            else:
                # Используем только название если документ не найден
                pairs.append([query, result.title])
                valid_results.append(result)

        # Получаем скоры от cross-encoder
        if pairs:
            ce_scores = self.cross_encoder.predict(pairs)

            # Нормализуем CE скоры
            ce_scores_normalized = []
            ce_min = min(ce_scores)
            ce_max = max(ce_scores)
            ce_range = ce_max - ce_min

            if ce_range > 0:
                # Нормализация от 0 до 1
                for score in ce_scores:
                    normalized = (score - ce_min) / ce_range
                    ce_scores_normalized.append(normalized)
            else:
                ce_scores_normalized = [0.5] * len(ce_scores)

            # Обновляем результаты
            for result, ce_score, ce_norm in zip(valid_results, ce_scores, ce_scores_normalized):
                result.cross_encoder_score = float(ce_score)

                # Улучшенная формула комбинирования
                # Даем больше веса cross-encoder для точности
                result.score = (
                        0.4 * result.score +  # Базовый скор (RRF + normalized)
                        0.6 * ce_norm  # Cross-encoder скор
                )

                # Бонус за высокий CE скор
                if ce_norm > 0.8:
                    result.score *= 1.1
                elif ce_norm < 0.2:
                    result.score *= 0.9

        # Пересортировка
        valid_results.sort(key=lambda x: x.score, reverse=True)

        return valid_results

    def _detect_negative_patterns(self, query: str) -> List[str]:
        """Автоматическое определение негативных паттернов"""
        negative_patterns = []

        # Паттерны противоположностей
        opposite_prefixes = ["анти", "де", "раз", "без"]

        # Проверяем, содержит ли запрос базовое слово без префикса
        words = query.lower().split()
        for word in words:
            # Если слово НЕ начинается с негативного префикса
            has_negative_prefix = any(word.startswith(prefix) for prefix in opposite_prefixes)

            if not has_negative_prefix and len(word) > 4:
                # Добавляем возможные негативные варианты в фильтр
                for prefix in opposite_prefixes:
                    negative_patterns.append(f"{prefix}{word}")

        return negative_patterns


# Функция для тестирования
def test_search_system():
    """Тестирование системы"""

    # Создаем систему поиска
    search_system = UniversalHybridSearch()

    # Проверяем, нужна ли индексация
    doc_count = search_system.es.count(index=ES_INDEX_NAME)["count"]
    if doc_count == 0:
        logger.info("Индекс пуст, запускаем индексацию...")
        search_system.index_products()
    else:
        logger.info(f"Индекс содержит {doc_count} документов")

    # Тестовые запросы
    test_queries = [
        "степлер",
        "антистеплер",
        "ноутбук для работы",
        "сода пищевая",
        "046727",  # артикул
        "ручка шариковая 0.5мм синяя",
        "фотобумага А4 50 листов",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Запрос: {query}")
        print('=' * 60)

        results = search_system.search(query, top_k=5)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   Финальный скор: {result.score:.4f}")
            print(f"   ES скор: {result.elastic_score:.4f} (ранг: {result.debug_info.get('es_rank', 'N/A')})")
            print(f"   Векторный скор: {result.vector_score:.4f} (ранг: {result.debug_info.get('vector_rank', 'N/A')})")
            if result.cross_encoder_score > 0:
                print(f"   Cross-encoder скор: {result.cross_encoder_score:.4f}")


# Интерактивный режим
def interactive_search():
    """Интерактивный поиск"""
    from bson import ObjectId

    search_system = UniversalHybridSearch()

    # Проверка индексации
    doc_count = search_system.es.count(index=ES_INDEX_NAME)["count"]
    if doc_count == 0:
        print("❌ Индекс пуст! Сначала запустите индексацию.")
        response = input("Запустить индексацию сейчас? (y/n): ")
        if response.lower() == 'y':
            search_system.index_products()
        else:
            return

    print("\n" + "=" * 60)
    print("УНИВЕРСАЛЬНЫЙ ГИБРИДНЫЙ ПОИСК")
    print("=" * 60)
    print("Команды:")
    print("  'exit' - выход")
    print("  'debug on/off' - включить/выключить отладку")
    print("  'config' - показать конфигурацию")
    print("  'set k VALUE' - изменить RRF K (текущее: 10)")
    print("  'set weights ES_WEIGHT VECTOR_WEIGHT' - изменить веса")
    print("  'reindex' - переиндексировать данные")
    print("=" * 60)

    debug_mode = False

    while True:
        query = input("\n🔍 Введите запрос: ").strip()

        if query.lower() == 'exit':
            break

        if query.lower() == 'debug on':
            debug_mode = True
            print("✅ Режим отладки включен")
            continue

        if query.lower() == 'debug off':
            debug_mode = False
            print("✅ Режим отладки выключен")
            continue

        if query.lower() == 'config':
            print("\nТекущая конфигурация:")
            print(f"  ES вес: {search_system.config.elastic_weight}")
            print(f"  Векторный вес: {search_system.config.vector_weight}")
            print(f"  RRF K: {search_system.config.rrf_k}")
            print(f"  Cross-encoder: {'Включен' if search_system.config.use_cross_encoder else 'Выключен'}")
            continue

        if query.lower().startswith('set k '):
            try:
                new_k = int(query.split()[2])
                search_system.config.rrf_k = new_k
                print(f"✅ RRF K изменен на {new_k}")
            except:
                print("❌ Формат: set k ЧИСЛО")
            continue

        if query.lower().startswith('set weights '):
            try:
                parts = query.split()
                es_weight = float(parts[2])
                vector_weight = float(parts[3])
                search_system.config.elastic_weight = es_weight
                search_system.config.vector_weight = vector_weight
                print(f"✅ Веса изменены: ES={es_weight}, Vector={vector_weight}")
            except:
                print("❌ Формат: set weights ES_WEIGHT VECTOR_WEIGHT")
            continue

        if query.lower() == 'reindex':
            print("Запуск переиндексации...")
            search_system.index_products()
            continue

        if not query:
            continue

        # Выполняем поиск
        try:
            # Если режим отладки, показываем извлеченные атрибуты
            if debug_mode:
                attributes = search_system._parse_query_attributes(query)
                if attributes:
                    print(f"\n🔍 Извлеченные атрибуты:")
                    for key, value in attributes.items():
                        print(f"   - {key}: {value}")

            results = search_system.search(query, top_k=10)

            if results:
                print(f"\n📋 Найдено результатов: {len(results)}")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result.title}")
                    print(f"   Скор: {result.score:.4f}")

                    # Отладочная информация
                    if debug_mode:
                        print(f"   Детали скоринга:")
                        print(
                            f"     - ES скор: {result.elastic_score:.3f} (ранг: {result.debug_info.get('es_rank', 'N/A')})")
                        print(
                            f"     - Vector скор: {result.vector_score:.3f} (ранг: {result.debug_info.get('vector_rank', 'N/A')})")
                        if result.cross_encoder_score > 0:
                            print(f"     - Cross-encoder: {result.cross_encoder_score:.3f}")
                        print(f"     - RRF компонент: {result.debug_info.get('rrf_score', 0):.3f}")
                        print(f"     - Norm компонент: {result.debug_info.get('norm_score', 0):.3f}")

                    # Получаем дополнительную информацию
                    doc = None
                    try:
                        doc = search_system.mongo.find_one({"_id": ObjectId(result.id)})
                    except:
                        pass

                    if not doc:
                        try:
                            es_doc = search_system.es.get(index=ES_INDEX_NAME, id=result.id)
                            if es_doc and '_source' in es_doc:
                                doc = es_doc['_source']
                        except:
                            pass

                    if doc:
                        print(f"   Категория: {doc.get('category', 'N/A')}")
                        print(f"   Бренд: {doc.get('brand', 'N/A')}")
                        print(f"   Артикул: {doc.get('article', 'N/A')}")

                        # Цена
                        min_price = search_system._extract_min_price(doc)
                        if min_price:
                            print(f"   Цена от: {min_price} руб.")
            else:
                print("❌ Ничего не найдено")

        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            logger.error(f"Search error: {e}", exc_info=True)


if __name__ == "__main__":
    print("Выберите режим:")
    print("1. Тестирование")
    print("2. Интерактивный поиск")
    print("3. Индексация данных")

    choice = input("Ваш выбор (1/2/3): ").strip()

    if choice == "1":
        test_search_system()
    elif choice == "2":
        interactive_search()
    elif choice == "3":
        search_system = UniversalHybridSearch()
        search_system.index_products()
    else:
        print("Неверный выбор")