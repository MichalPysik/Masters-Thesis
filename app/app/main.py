"""The entrypoint for the application."""

from fastapi import FastAPI
from pydantic import BaseModel
import os

from app.preprocess import get_image_text_similarity, extract_video_frames
from app.search_embeddings import search_similar_frames
from app.utils import format_timestamp


app = FastAPI(title="Video LLM-based search engine API")


@app.get("/")
def main_route():
    #return {"message": "Welcome to the Video LLM-based search engine API. Visit /docs for the API documentation."}
    # load from .env
    current_embedding_model = os.getenv("EMBEDDING_MODEL")
    current_mllm_model = os.getenv("MLLM_MODEL")
    return {"message": f"Welcome to the Video LLM-based search engine API. Current embedding model: {current_embedding_model}, Current multimodal language model: {current_mllm_model}. Visit /docs for the API documentation."}



class SimilarityRequest(BaseModel):
    image_path: str
    text: str

@app.post("/test-embedding-model-similarity")
def test_embedding_model_similarity(request: SimilarityRequest):
    similarity_score = get_image_text_similarity(request.image_path, request.text)
    return {"similarity_score": similarity_score}


@app.post("/upload")
def upload_video(video_path: str):
    extract_video_frames(video_path)
    return {"message": "Video frames extracted and embeddings stored in Milvus."}


@app.get("/search")
def search(text: str, top_k: int = 5):
    results = search_similar_frames(text, top_k)
    return {
        "results": [
            {
                "video_name": result.entity.get("video_name"),
                "timestamp": format_timestamp(result.entity.get("timestamp")),
                "similarity_score": result.distance
            }
            for result in results[0]
        ]
    }
