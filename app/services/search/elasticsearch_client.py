from typing import Dict, List, Any, Optional
import logging

from elasticsearch import Elasticsearch

from app.core.settings import settings
from app.services.search.query_builder import ElasticsearchQueryBuilder

# Используем стандартный логгер
logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Клиент для поиска товаров в Elasticsearch"""

    def __init__(self, query_builder: ElasticsearchQueryBuilder = None):
        self.es = None
        self.index_name = settings.ELASTICSEARCH_INDEX
        self.query_builder = query_builder or ElasticsearchQueryBuilder()

        # Подключаемся при инициализации
        self.connect()

    def connect(self) -> bool:
        """Подключение к Elasticsearch"""

        try:
            self.es = Elasticsearch(**settings.get_elasticsearch_config())

            # Проверяем подключение
            info = self.es.info()
            logger.info(f"Подключен к Elasticsearch {info['version']['number']}")

            return True

        except Exception as e:
            logger.error(f"Ошибка подключения к Elasticsearch: {e}")
            return False

    def search_products(
            self,
            search_terms: Dict,
            size: Optional[int] = None,
            filters: Optional[Dict] = None
    ) -> Dict:
        """Поиск товаров по терминам"""

        if not self.es:
            logger.error("Elasticsearch не подключен")
            return {'error': 'ES не подключен', 'candidates': []}

        # Проверяем существование индекса
        if not self.es.indices.exists(index=self.index_name):
            logger.error(f"Индекс {self.index_name} не существует")
            return {'error': f'Индекс {self.index_name} не существует', 'candidates': []}

        # Строим запрос
        query = self.query_builder.build_tender_query(search_terms)
        logger.info(f"Построен запрос для индекса {self.index_name}")

        # Добавляем фильтры если есть
        if filters:
            query = self.query_builder.add_filters(query, filters)
            logger.info(f"Добавлены фильтры: {list(filters.keys())}")

        # Устанавливаем размер выборки
        if size:
            query['size'] = size
        else:
            query['size'] = settings.MAX_SEARCH_RESULTS

        logger.info(f"Размер выборки: {query['size']}")

        # Указываем какие поля возвращать
        query['_source'] = ["title", "category", "brand", "attributes"]

        try:
            # Выполняем поиск
            logger.info("Выполнение поиска в Elasticsearch...")
            response = self.es.search(index=self.index_name, body=query)

            # Обрабатываем результаты
            candidates = self._process_search_results(response)

            result = {
                'candidates': candidates,
                'total_found': response['hits']['total']['value'],
                'max_score': response['hits']['max_score'] or 0,
                'search_terms_used': search_terms,
                'query_type': 'tender_search'
            }

            logger.info(
                f"Поиск завершен: найдено {len(candidates)} из {result['total_found']} товаров "
                f"(макс. скор: {result['max_score']:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска: {e}", exc_info=True)
            return {'error': str(e), 'candidates': []}

    def _process_search_results(self, response: Dict) -> List[Dict]:
        """Обработка результатов поиска"""

        candidates = []

        for hit in response['hits']['hits']:
            source = hit['_source']

            candidate = {
                'id': hit['_id'],
                'title': source.get('title', 'Без названия'),
                'category': source.get('category', 'Без категории'),
                'brand': source.get('brand', ''),
                'elasticsearch_score': hit['_score'],
                'attributes': source.get('attributes', [])
            }

            candidates.append(candidate)

        return candidates

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Получить товар по ID"""

        if not self.es:
            return None

        try:
            response = self.es.get(index=self.index_name, id=product_id)

            if response['found']:
                source = response['_source']
                return {
                    'id': product_id,
                    'title': source.get('title', ''),
                    'category': source.get('category', ''),
                    'brand': source.get('brand', ''),
                    'attributes': source.get('attributes', [])
                }

        except Exception as e:
            logger.error(f"Ошибка получения товара {product_id}: {e}")

        return None

    def ping(self) -> bool:
        """Проверка доступности Elasticsearch"""

        if not self.es:
            return False

        try:
            return self.es.ping()
        except:
            return False