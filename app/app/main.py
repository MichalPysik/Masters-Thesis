"""The entrypoint for the application."""

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pymilvus import connections

from app.preprocess import get_image_text_similarity

load_dotenv()
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")
connections.connect(alias="default", host=host, port=port)

#debug - check if connection is successful
cns = connections.list_connections()
print(cns)

app = FastAPI(title="Video LLM-based search engine API")


@app.get("/")
def main_route():
    return {"message": "Welcome to the Video LLM-based search engine API!"}


class SimilarityRequest(BaseModel):
    image_path: str
    text: str

@app.post("/similarity")
def similarity(request: SimilarityRequest):
    similarity_score = get_image_text_similarity(request.image_path, request.text)
    return {"similarity_score": similarity_score}