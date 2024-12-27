# Настройки для подключения к Milvus
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_DB_NAME = "movies_database"
MILVUS_COLLECTION_NAME = "movies_collection"

# Настройки для подключения к Elasticsearch
ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "password123"
ES_INDEX_NAME = "movies"

# Модель для эмбеддингов
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Модель для реранкинга
RERANKER_MODEL_NAME = "amberoad/bert-multilingual-passage-reranking-msmarco"
