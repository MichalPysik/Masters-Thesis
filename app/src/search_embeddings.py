import torch
import os
import ffmpeg
import datetime
from typing import List, Dict

from src import device, emb_model, emb_processor
from src.db_and_storage import (
    collection,
    minio_client,
    BUCKET_NAME,
    check_bucket_object_exists,
)
from src.utils import format_timestamp


def search_similar_frames(text: str, top_k: int = 5) -> List[Dict]:
    """
    Searches for the most similar video frames to the input text in the Milvus database.

    Args:
        text (str): The input text to search for.
        top_k (int): The number of most similar frames to return.

    Returns:
        List[Dict]: The list of search results as dicts, each containing the video name, timestamp, and similarity score.
    """
    assert top_k > 0, "The 'top_k' parameter must be greater than 0."

    # Load the collection if it's not already loaded
    collection.load()

    # Process the input text and generate embedding
    if os.getenv("EMBEDDING_MODEL") == "BLIP":
        inputs = emb_processor[1]["eval"](text)
        sample = {"image": None, "text_input": [inputs]}
        text_features = emb_model.extract_features(sample, mode="text")
        # Project from 768 to 256 dimensions (includes normalization)
        text_features = text_features.text_embeds_proj[:, 0, :]
    else:
        inputs = emb_processor(text=text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = emb_model.get_text_features(**inputs)
        # Normalize the embedding
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Convert the text embedding to 1-D numpy array
    query_embedding = text_features.cpu().numpy().flatten()

    # Search in Milvus
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "top_k": top_k},
        consistency_level="Strong",
        output_fields=["video_name", "timestamp"],
        limit=top_k,
    )[0]

    return search_results


def get_frame_from_video(video_name: str, timestamp: float) -> bytes:
    """
    Retrieves a frame image for the specified timestamp from a video stored in the Minio bucket.
    It uses a presigned URL so that ffmpeg can access the video directly, without a local download.

    Args:
        video_name (str): The name of the video file in the Minio bucket.
        timestamp (float): The timestamp in seconds for the frame to extract.

    Returns:
        bytes: The image data in bytes.
    """
    if not check_bucket_object_exists(video_name):
        raise FileNotFoundError(f"Video file not found in the bucket: {video_name}")

    # Generate a presigned URL for the video valid for 1 minute
    url = minio_client.presigned_get_object(
        BUCKET_NAME, video_name, expires=datetime.timedelta(minutes=1)
    )

    # Probe the video to obtain metadata (including duration)
    try:
        probe = ffmpeg.probe(url)
        video_duration = float(probe["format"]["duration"])
    except Exception as e:
        raise RuntimeError(f"Error retrieving video metadata: {e}")

    # Check if the timestamp is not out of bounds
    if timestamp < 0 or timestamp > video_duration:
        raise ValueError(
            f"Timestamp {timestamp} seconds is out of bounds for video '{video_name}' lasting {video_duration} seconds."
        )

    # Extract frame at the specified timestamp
    try:
        image_data, err = (
            ffmpeg.input(url, ss=timestamp)
            .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        error_message = e.stderr.decode()
        raise RuntimeError(
            f"Error extracting frame from '{video_name}' at {format_timestamp(timestamp)}: {error_message}"
        )

    return image_data
