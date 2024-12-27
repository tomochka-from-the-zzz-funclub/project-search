import pandas as pd
from pymilvus import connections, db, Collection
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from elasticsearch import Elasticsearch
from src.settings import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_DB_NAME, MILVUS_COLLECTION_NAME,
    ES_HOST, ES_USER, ES_PASSWORD, ES_INDEX_NAME,
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME
)

# Подключение к Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
db.using_database(MILVUS_DB_NAME)

# Подключение к Elasticsearch
es = Elasticsearch(ES_HOST, http_auth=(ES_USER, ES_PASSWORD))

# Загрузка данных
df_movies = pd.read_csv('data/merged_movies.csv')
df_movies = df_movies.dropna(how='any')
df_movies_small = df_movies.iloc[:800]

# Загрузка моделей
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

# Функция для генерации эмбеддингов
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings[0]

# Milvus поиск
def search_movies_by_milvus(query, top_k=5):
    query_embedding = get_embedding(query)
    collection = Collection(MILVUS_COLLECTION_NAME)
    collection.load()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=top_k, output_fields=["movie_id"])
    movie_ids = [result.id for result in results[0]]
    return movie_ids

# Elasticsearch поиск
def search_movie_by_elastic(query, top_k=5):
    search_body = {
        "query": {
            "function_score": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "genres^2", "overview", "production_countries", "spoken_languages"],
                        "fuzziness": "AUTO"
                    }
                },
                "boost_mode": "multiply",
                "functions": [
                    {"field_value_factor": {"field": "vote_average", "factor": 1.5, "missing": 0}},
                    {"field_value_factor": {"field": "vote_count", "factor": 1.2, "missing": 1}}
                ]
            }
        },
        "size": top_k
    }
    response = es.search(index=ES_INDEX_NAME, body=search_body)
    return [{"movieId": hit["_source"]["movieId"], "title": hit["_source"]["title"], "overview": hit["_source"]["overview"]} for hit in response["hits"]["hits"]]

# Финальное ранжирование
def final_ranking(query, top_k=5):
    milvus_candidates = search_movies_by_milvus(query)
    milvus_results = [
        {
            "movieId": movie_id,
            "title": df_movies_small[df_movies_small['movieId'] == movie_id]['title'].values[0],
            "overview": df_movies_small[df_movies_small['movieId'] == movie_id]['overview'].values[0]
        }
        for movie_id in milvus_candidates
    ]
    elastic_results = search_movie_by_elastic(query)
    unique_candidates = {result["movieId"]: result for result in milvus_results + elastic_results}.values()
    inputs = [(query, f"{candidate['title']} | {candidate['overview']}") for candidate in unique_candidates]
    encoded_inputs = reranker_tokenizer([q for q, _ in inputs], [text for _, text in inputs], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = reranker_model(**encoded_inputs)
        scores = outputs.logits[:, 0]
    return sorted(
        [{**candidate, "rank_score": score.item()} for candidate, score in zip(unique_candidates, scores)],
        key=lambda x: x["rank_score"], reverse=True
    )[:top_k]
