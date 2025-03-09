"""The entrypoint for the application."""

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, RedirectResponse, FileResponse
import os
from typing import List, Dict
from pymilvus.exceptions import MilvusException
import logging
from pydantic import BaseModel
import uvicorn

from src.preprocess import (
    upload_video_to_bucket,
    extract_and_store_embeddings,
    recreate_embeddings,
    synchronize_embeddings_with_bucket,
)
from src.search_embeddings import search_similar_frames, get_frame_from_video
from src.utils import format_timestamp
from src.analyze_video import perform_video_analysis

# Ensure that db_and_storage.py is executed at startup (but after __init__.py)
import src.db_and_storage as db_and_storage


tags_metadata = [
    {
        "name": "Search and analysis",
        "description": "Endpoints for searching frame embeddings and analyzing videos.",
    },
    {
        "name": "Data management",
        "description": "Endpoints for uploading, deleting, and managing data.",
    },
]

app = FastAPI(
    title="System for search and analysis of traffic footage.",
    openapi_tags=tags_metadata,
)

# All backend endpoints have /api prefix
api_router = APIRouter(prefix="/api")


@api_router.get(
    "/search-embeddings",
    tags=["Search and analysis"],
    summary="Search frame embeddings",
)
def search_embeddings(text: str, top_k: int = 5):
    """
    Search for video frames with the most similar embeddings to the input text.

    Returns a list of the top K most similar frames, with each entry including the video name, timestamp, and similarity score.
    """
    if os.getenv("EMBEDDING_MODEL").lower() == "none":
        raise HTTPException(
            status_code=412,
            detail="Cannot use this function as no embedding model is currently configured.",
        )
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


@api_router.get(
    "/image-from-video/{video_name}",
    tags=["Search and analysis"],
    summary="Get image from video",
)
def get_image_from_video(video_name: str, timestamp: float):
    """
    Get a JPEG image sampled from the specified video at the specified timestamp (in seconds).
    """
    try:
        image_bytes = get_frame_from_video(video_name, timestamp)
        return Response(content=image_bytes, media_type="image/jpeg")
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


class NewConversationModel(BaseModel):
    query: str
    start_timestamp: float
    end_timestamp: float
    num_frames: int = 16


@api_router.post(
    "/start-video-analysis/{video_name}",
    tags=["Search and analysis"],
    summary="Start video analysis",
)
def start_video_analysis(video_name: str, payload: NewConversationModel):
    """
    Start a new video analysis conversation about a specified video segment with the configured MLLM.

    The video segment is defined by the start and end timestamps (in seconds) and the number of frames to uniformly sample from it.

    Returns the MLLM's answer and the entire conversation context including the answer.
    """
    if os.getenv("MLLM").lower() == "none":
        raise HTTPException(
            status_code=412,
            detail="Cannot use this function as no MLLM model is currently configured.",
        )
    try:
        answer, conversation = perform_video_analysis(
            payload.query,
            video_name,
            start_timestamp=payload.start_timestamp,
            end_timestamp=payload.end_timestamp,
            num_frames=payload.num_frames,
        )
        return {"answer": answer, "conversation": conversation}
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


class ExistingConversationModel(BaseModel):
    query: str
    existing_conversation: List[Dict]


@api_router.post(
    "/continue-video-analysis",
    tags=["Search and analysis"],
    summary="Continue video analysis",
)
def continue_video_analysis(payload: ExistingConversationModel):
    """
    Continue an existing video analysis conversation with the configured MLLM.

    The existing conversation must include metadata generated by the system when starting a new video analysis conversation.

    Returns the MLLM's answer and the entire conversation context including the answer.
    """
    if os.getenv("MLLM").lower() == "none":
        raise HTTPException(
            status_code=412,
            detail="Cannot use this function as no MLLM model is currently configured.",
        )
    try:
        answer, conversation = perform_video_analysis(
            payload.query, existing_conversation=payload.existing_conversation
        )
        return {"answer": answer, "conversation": conversation}
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


class VideoUploadModel(BaseModel):
    video_path: str
    bucket_only: bool = False
    sampling_fps: float = 1.0


@api_router.post("/data/upload-video", tags=["Data management"], summary="Upload video")
def upload_video(payload: VideoUploadModel):
    """
    Upload a video to the Minio bucket, sample frames from it at the specified framerate (float), and extract and store corresponding embeddings in the Milvus database collection.

    If 'bucket_only' is set to True, the video is only uploaded to the Minio bucket without processing the frames.
    """
    if os.getenv("EMBEDDING_MODEL").lower() == "none" and not payload.bucket_only:
        raise HTTPException(
            status_code=412,
            detail="Cannot use this function with bucket_only=False as no embedding model is currently configured.",
        )
    try:
        video_name = upload_video_to_bucket(payload.video_path)
        if payload.bucket_only:
            return {
                "message": f"Successfully uploaded video '{video_name}' to the Minio bucket only."
            }
        num_frames = extract_and_store_embeddings(video_name, payload.sampling_fps)
        return {
            "message": f"Successfully uploaded video '{video_name}' and processed {num_frames} frames."
        }
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except MilvusException as e:
        if "should divide the dim" in str(e):
            detail = "The dimensions of embedding vectors stored in the Milvus database do not match the dimensions of the currently configured embedding model. Either synchronize the data with 'force_bucket_mirror=True' to recreate the Milvus collection, or switch back to the corresponding embedding model."
            logging.error(detail)
            raise HTTPException(status_code=500, detail=detail)
        raise e
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/list-all", tags=["Data management"], summary="List all data")
def list_all_data():
    """ "
    List all video data in the Minio bucket and the Milvus database collection.

    Videos only present in the bucket will have 'embedding_entries' set to 0.
    Videos only present in the Milvus database collection will have 'in_bucket' set to False and duration set to None.
    """
    try:
        minio_data, milvus_data = db_and_storage.list_all_data()
        video_names = set(minio_data.keys()).union(milvus_data.keys())
        all_data = [
            {
                "video_name": video_name,
                "in_bucket": video_name in minio_data,
                "embedding_entries": (
                    milvus_data.get(video_name) if video_name in milvus_data else 0
                ),
                "duration": minio_data.get(video_name),
                "human_duration": (
                    format_timestamp(minio_data.get(video_name))
                    if minio_data.get(video_name)
                    else None
                ),
            }
            for video_name in video_names
        ]
        return all_data
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


class SynchronizeVideoModel(BaseModel):
    sampling_fps: float = 1.0


@api_router.post(
    "/data/synchronize-video/{video_name}",
    tags=["Data management"],
    summary="Synchronize video",
)
def synchronize_video(video_name: str, payload: SynchronizeVideoModel):
    """
    Recreate embeddings for a specific video by re-processing its frames at specified framerate (float).

    This is useful when the video embedding processing was interrupted, or when the video was only uploaded to the Minio bucket.
    """
    if os.getenv("EMBEDDING_MODEL").lower() == "none":
        raise HTTPException(
            status_code=412,
            detail="Cannot use this function as no embedding model is currently configured.",
        )
    try:
        num_frames = recreate_embeddings(video_name, payload.sampling_fps)
        return {
            "message": f"Successfully recreated embeddings for video '{video_name}' by processing {num_frames} frames."
        }
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except MilvusException as e:
        if "should divide the dim" in str(e):
            detail = "The dimensions of embedding vectors stored in the Milvus database do not match the dimensions of the currently configured embedding model. Either synchronize the data with 'force_bucket_mirror=True' to recreate the Milvus collection, or switch back to the corresponding embedding model."
            logging.error(detail)
            raise HTTPException(status_code=500, detail=detail)
        raise e
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


class SynchronizeAllModel(BaseModel):
    force_bucket_mirror: bool = False


@api_router.post(
    "/data/synchronize-all",
    tags=["Data management"],
    summary="Synchronize all data",
)
def synchronize_all_data(payload: SynchronizeAllModel):
    """
    Synchronize the Milvus database collection with the Minio bucket.

    If 'force_bucket_mirror' is set to True, the Milvus collection is recreated based on the current state of the Minio bucket.

    Otherwise, embeddings without a corresponding video in the bucket are deleted from the Milvus collection,
    and all videos in the bucket that have 0 corresponding Milvus entries get re-processed to extract embeddings.
    """
    if os.getenv("EMBEDDING_MODEL").lower() == "none":
        raise HTTPException(
            status_code=412,
            detail="Cannot use this function as no embedding model is currently configured.",
        )
    try:
        synchronize_embeddings_with_bucket(payload.force_bucket_mirror)
        if payload.force_bucket_mirror:
            return {
                "message": "Successfully recreated the Milvus collection based on the current state of the Minio bucket."
            }
        return {
            "message": "Successfully synchronized the Milvus collection with the Minio bucket."
        }
    except MilvusException as e:
        if "should divide the dim" in str(e):
            detail = "The dimensions of embedding vectors stored in the Milvus database do not match the dimensions of the currently configured embedding model. Either synchronize the data with 'force_bucket_mirror=True' to recreate the Milvus collection, or switch back to the corresponding embedding model."
            logging.error(detail)
            raise HTTPException(status_code=500, detail=detail)
        raise e
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete(
    "/data/delete-video/{video_name}",
    tags=["Data management"],
    summary="Delete video",
)
def delete_video(video_name: str):
    """Delete a video from the Minio bucket along with all related data in the Milvus database collection."""
    try:
        db_and_storage.delete_video(video_name)
        return {
            "message": f"Video '{video_name}' and all related data have been deleted."
        }
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete(
    "/data/delete-all", tags=["Data management"], summary="Delete all data"
)
def delete_all_data():
    """Delete all data from both the Minio bucket and the Milvus database collection."""
    try:
        db_and_storage.delete_all_data()
        return {"message": "All data has been deleted."}
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(api_router)


# Mount frontend last to avoid conflicts with API routes (or redirect to docs is frontend is not used)
if os.getenv("USE_FRONTEND", "false").lower() == "true":
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

    # Catch-all route to support Vue Router's history mode
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        return FileResponse("frontend/dist/index.html")

else:

    @app.get("/")
    def redirect_to_docs():
        """Redirect to the API documentation."""
        return RedirectResponse(url="/docs")


def main():
    """Entrypoint for running the FastAPI application."""
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
