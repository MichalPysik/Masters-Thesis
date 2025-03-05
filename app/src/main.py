"""The entrypoint for the application."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, RedirectResponse
import os
from typing import List, Dict

from src.preprocess import upload_video_to_bucket, extract_and_store_embeddings
from src.search_embeddings import search_similar_frames, get_frame_from_video
from src.utils import format_timestamp
from src.analyze_video import perform_video_analysis

# Ensure that db_and_storage.py is executed at startup (but after __init__.py)
import src.db_and_storage as db_and_storage


app = FastAPI(title="Video LLM-based search engine API")


@app.post("/upload-video")
def upload_video(video_path: str):
    try:
        video_name = upload_video_to_bucket(video_path)
        num_frames = extract_and_store_embeddings(video_name)
        return {"message": f"Successfully uploaded video '{video_name}' and processed {num_frames} frames."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search-embeddings")
def search_embeddings(text: str, top_k: int = 5):
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
                for result in results[0]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image-from-video/{video_name}")
def get_image_from_video(video_name: str, timestamp: float):
    try:
        image_bytes = get_frame_from_video(video_name, timestamp)
        return Response(content=image_bytes, media_type="image/jpeg")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-video-analysis/{video_name}")
def start_video_analysis(query: str, video_name: str, start_timestamp: float, end_timestamp: float, num_frames: int = 16):
    try:
        answer, conversation = perform_video_analysis(query, video_name, start_timestamp=start_timestamp, end_timestamp=end_timestamp, num_frames=num_frames)
        return {"answer": answer, "conversation": conversation}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/continue-video-analysis")
def continue_video_analysis(query: str, existing_conversation: List[Dict]):
    try:
        answer, conversation = perform_video_analysis(query, existing_conversation=existing_conversation)
        return {"answer": answer, "conversation": conversation}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/data/delete-video/{video_name}")
def delete_video(video_name: str):
    try:
        db_and_storage.delete_video(video_name)
        return {"message": f"Video {video_name} and all related data have been deleted."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/data/delete-all")
def delete_all_data():
    db_and_storage.delete_all_data()
    return {"message": "All data has been deleted."}


# Mount frontend last to avoid conflicts with API routes (or redirect to docs is frontend is not used)
if os.getenv("USE_FRONTEND", "false").lower() == "true":
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
else:
    @app.get("/")
    def redirect_to_docs():
        return RedirectResponse(url="/docs")
