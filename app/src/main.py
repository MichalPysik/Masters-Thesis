"""The entrypoint for the application."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, RedirectResponse
import os
from typing import List, Dict
from pymilvus.exceptions import MilvusException
import logging

from src.preprocess import upload_video_to_bucket, extract_and_store_embeddings, recreate_embeddings, synchronize_all_data
from src.search_embeddings import search_similar_frames, get_frame_from_video
from src.utils import format_timestamp
from src.analyze_video import perform_video_analysis

# Ensure that db_and_storage.py is executed at startup (but after __init__.py)
import src.db_and_storage as db_and_storage


tags_metadata = [
    {
        "name": "Search and analysis",
        "description": "Endpoints for searching similar frames and analyzing videos.",
    },
    {
        "name": "Data management",
        "description": "Endpoints for uploading, deleting, and managing data.",
    },
]

app = FastAPI(title="System for search and analysis of traffic footage.", openapi_tags=tags_metadata)



@app.get("/search-embeddings", tags=["Search and analysis"])
def search_embeddings(text: str, top_k: int = 5):
    if os.getenv("EMBEDDING_MODEL").lower() == "none":
        raise HTTPException(status_code=412, detail="Cannot use this function as no embedding model is currently configured.")
    try:
        results = search_similar_frames(text, top_k)
        return {
            "results": [
                {
                    "video_name": result.entity.get("video_name"),
                    "timestamp": result.entity.get("timestamp"),
                    "human_timestamp": format_timestamp(result.entity.get("timestamp")),
                    "similarity_score": result.distance,
                }
                for result in results
            ]
        }
    except MilvusException as e:
        if "vector dimension mismatch" in str(e):
            detail = "The dimensions of embedding vectors stored in the Milvus database do not match the dimensions of the currently configured embedding model. Either synchronize the data with 'force_bucket_mirror=True' to recreate the Milvus collection, or switch back to the corresponding embedding model."
            logging.error(detail)
            raise HTTPException(status_code=500, detail=detail)
        raise e
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image-from-video/{video_name}", tags=["Search and analysis"])
def get_image_from_video(video_name: str, timestamp: float):
    try:
        image_bytes = get_frame_from_video(video_name, timestamp)
        return Response(content=image_bytes, media_type="image/jpeg")
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-video-analysis/{video_name}", tags=["Search and analysis"])
def start_video_analysis(query: str, video_name: str, start_timestamp: float, end_timestamp: float, num_frames: int = 16):
    if os.getenv("MLLM").lower() == "none":
        raise HTTPException(status_code=412, detail="Cannot use this function as no MLLM model is currently configured.")
    try:
        answer, conversation = perform_video_analysis(query, video_name, start_timestamp=start_timestamp, end_timestamp=end_timestamp, num_frames=num_frames)
        return {"answer": answer, "conversation": conversation}
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/continue-video-analysis", tags=["Search and analysis"])
def continue_video_analysis(query: str, existing_conversation: List[Dict]):
    if os.getenv("MLLM").lower() == "none":
        raise HTTPException(status_code=412, detail="Cannot use this function as no MLLM model is currently configured.")
    try:
        answer, conversation = perform_video_analysis(query, existing_conversation=existing_conversation)
        return {"answer": answer, "conversation": conversation}
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/upload-video", tags=["Data management"])
def upload_video(video_path: str, bucket_only: bool = False, sampling_fps: float = 1.0):
    if os.getenv("EMBEDDING_MODEL").lower() == "none" and not bucket_only:
        raise HTTPException(status_code=412, detail="Cannot use this function with bucket_only=False as no embedding model is currently configured.")
    try:
        video_name = upload_video_to_bucket(video_path)
        if bucket_only:
            return {"message": f"Successfully uploaded video '{video_name}' to the Minio bucket only."}
        num_frames = extract_and_store_embeddings(video_name, sampling_fps)
        return {"message": f"Successfully uploaded video '{video_name}' and processed {num_frames} frames."}
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except MilvusException as e:
        if "should divide the dim" in str(e):
            detail="The dimensions of embedding vectors stored in the Milvus database do not match the dimensions of the currently configured embedding model. Either synchronize the data with 'force_bucket_mirror=True' to recreate the Milvus collection, or switch back to the corresponding embedding model."
            logging.error(detail)
            raise HTTPException(status_code=500, detail=detail)
        raise e
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/data/list-all", tags=["Data management"])
def list_all_data():
    try:
        minio_data, milvus_data = db_and_storage.list_all_data()
        video_names = set(minio_data.keys()).union(milvus_data.keys())
        all_data = [
            {
                "video_name": video_name,
                "in_bucket": video_name in minio_data,
                "embedding_entries": milvus_data.get(video_name) if video_name in milvus_data else 0,
                "duration": minio_data.get(video_name),
                "human_duration": format_timestamp(minio_data.get(video_name)) if minio_data.get(video_name) else None,
            } for video_name in video_names
        ]
        return all_data
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/synchronize-video/{video_name}", tags=["Data management"])
def synchronize_video(video_name: str, sampling_fps: float = 1.0):
    if os.getenv("EMBEDDING_MODEL").lower() == "none":
        raise HTTPException(status_code=412, detail="Cannot use this function as no embedding model is currently configured.")
    try:
        num_frames = recreate_embeddings(video_name, sampling_fps)
        return {"message": f"Successfully recreated embeddings for video '{video_name}' by processing {num_frames} frames."}
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except MilvusException as e:
        if "should divide the dim" in str(e):
            detail="The dimensions of embedding vectors stored in the Milvus database do not match the dimensions of the currently configured embedding model. Either synchronize the data with 'force_bucket_mirror=True' to recreate the Milvus collection, or switch back to the corresponding embedding model."
            logging.error(detail)
            raise HTTPException(status_code=500, detail=detail)
        raise e
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/synchronize-all", tags=["Data management"])
def synchronize_all_data(force_bucket_mirror: bool = False):
    if os.getenv("EMBEDDING_MODEL").lower() == "none":
        raise HTTPException(status_code=412, detail="Cannot use this function as no embedding model is currently configured.")
    try:
        synchronize_all_data(force_bucket_mirror)
        if force_bucket_mirror:
            return {"message": "Successfully recreated the Milvus collection based on the current state of the Minio bucket."}
        return {"message": "Successfully synchronized the Milvus collection with the Minio bucket."}
    except MilvusException as e:
        if "should divide the dim" in str(e):
            detail="The dimensions of embedding vectors stored in the Milvus database do not match the dimensions of the currently configured embedding model. Either synchronize the data with 'force_bucket_mirror=True' to recreate the Milvus collection, or switch back to the corresponding embedding model."
            logging.error(detail)
            raise HTTPException(status_code=500, detail=detail)
        raise e
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/data/delete-video/{video_name}", tags=["Data management"])
def delete_video(video_name: str):
    try:
        db_and_storage.delete_video(video_name)
        return {"message": f"Video '{video_name}' and all related data have been deleted."}
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/data/delete-all", tags=["Data management"])
def delete_all_data():
    try:
        db_and_storage.delete_all_data()
        return {"message": "All data has been deleted."}
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Mount frontend last to avoid conflicts with API routes (or redirect to docs is frontend is not used)
if os.getenv("USE_FRONTEND", "false").lower() == "true":
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
else:
    @app.get("/")
    def redirect_to_docs():
        return RedirectResponse(url="/docs")
