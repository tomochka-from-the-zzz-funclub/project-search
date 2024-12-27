from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.search import final_ranking

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    movieId: int
    title: str
    rank_score: float

@app.post("/search_movies/", response_model=List[SearchResponse])
async def search_movies(request: QueryRequest):
    try:
        results = final_ranking(request.query, top_k=request.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
