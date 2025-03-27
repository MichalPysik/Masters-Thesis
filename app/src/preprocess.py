import torch
from PIL import Image
import os
import ffmpeg
import logging

from src import device, emb_model, emb_processor
from src.db_and_storage import (
    collection,
    COLLECTION_NAME,
    get_bucket_video_url,
    list_all_data,
    create_collection,
)
from src.utils import format_timestamp


def extract_and_store_embeddings(video_name: str, sampling_fps: float = 1.0) -> int:
    """
    Extracts frames from a video stored in the bucket, generates embeddings,
    and stores them in the Milvus database collection together with corresponding metadata.

    Args:
        video_name (str): The name of the video file in the Minio bucket.
        sampling_fps (float): How many frames to sample for each second of the video.

    Returns:
        int: The number of frames processed and stored in the Milvus database collection.
    """
    batch_size = int(os.getenv("PREPROCESS_BATCH_SIZE", 64))

    # Generate a presigned URL for the video valid for 1 hour
    url = get_bucket_video_url(video_name)

    # Probe the video to get metadata (duration, fps, total_frames, width, height)
    try:
        probe = ffmpeg.probe(url)
        video_duration = float(probe["format"]["duration"])
        # Handle fractional FPS values like "30000/1001"
        r_frame_rate = probe["streams"][0]["r_frame_rate"]
        if "/" in r_frame_rate:
            num, den = map(float, r_frame_rate.split("/"))
            video_fps = num / den
        else:
            video_fps = float(r_frame_rate)
        video_total_frames = int(probe["streams"][0]["nb_frames"])
        video_width = int(probe["streams"][0]["width"])
        video_height = int(probe["streams"][0]["height"])
    except Exception as e:
        raise RuntimeError(f"Error retrieving video metadata: {e}")

    logging.info(
        f"Video '{video_name}' has duration {format_timestamp(video_duration)} ({video_duration} seconds), {video_total_frames} frames total at {video_fps} FPS, and resolution {video_width}x{video_height}."
    )

    # Calculate how many frames we expect to extract at the selected sampling rate
    expected_frames = int(video_duration * sampling_fps)
    logging.info(
        f"Expecting to sample approximately {expected_frames} frames at {sampling_fps} FPS."
    )

    # Calculate the size of a single frame in bytes
    frame_size = video_width * video_height * 3

    # Start ffmpeg process to extract frames
    process = (
        ffmpeg.input(url)
        .filter("fps", fps=sampling_fps)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    total_processed = 0
    progress_step = max(expected_frames // 10, 1)

    try:
        while True:
            # Read a batch of frames
            batch_frames = []
            batch_timestamps = []

            # Try to fill the batch
            for i in range(batch_size):
                # Read a frame_size bytes of data
                raw_frame = process.stdout.read(frame_size)

                # If the frame is not full, it means we reached the end of the video
                if len(raw_frame) != frame_size:
                    break

                # Convert to PIL Image and calculate timestamp
                frame = Image.frombytes("RGB", (video_width, video_height), raw_frame)
                timestamp = (total_processed + i) / sampling_fps

                batch_frames.append(frame)
                batch_timestamps.append(timestamp)

            # If batch contains no frames, we reached the end of the video
            if not batch_frames:
                break

            logging.debug(f"Processing batch of {len(batch_frames)} frames...")

            # Process the batch of sampled frames
            for frame, timestamp in zip(batch_frames, batch_timestamps):
                # Process image and generate embedding using Multimodal embedding model
                if os.getenv("EMBEDDING_MODEL") == "BLIP":
                    inputs = emb_processor[0]["eval"](frame).unsqueeze(0).to(device)
                    sample = {"image": inputs, "text_input": None}
                    image_features = emb_model.extract_features(sample, mode="image")
                    # project from 768 to 256 dimensions (includes normalization)
                    image_features = image_features.image_embeds_proj[:, 0, :]
                else:
                    inputs = emb_processor(
                        images=frame, return_tensors="pt", padding=True
                    ).to(device)
                    with torch.no_grad():
                        image_features = emb_model.get_image_features(**inputs)
                    # Normalize the embedding
                    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

                # Convert the image embedding to 1-D numpy array
                embedding = image_features.cpu().numpy().flatten()

                # Insert into Milvus
                data = [[video_name], [timestamp], [embedding]]
                collection.insert(data)

                total_processed += 1

            # Report progress every 10 % (or on the last frame)
            if (
                total_processed % progress_step < len(batch_frames)
                or len(batch_frames) < batch_size
            ):
                progress_percent = min(total_processed * 100 / expected_frames, 100)
                logging.info(
                    f"Processed {total_processed} of ~{expected_frames} frames ({progress_percent:.0f} % complete)..."
                )

            # Free memory after processing a batch
            del batch_frames
            del batch_timestamps

    except Exception as e:
        # Try to get stderr output for better error reporting
        stderr_output = ""
        try:
            stderr_output = process.stderr.read().decode()
        except:
            pass

        if stderr_output:
            raise RuntimeError(
                f"Error processing video: {e}. FFmpeg error: {stderr_output}"
            )
        else:
            raise RuntimeError(f"Error processing video: {e}")

    finally:
        # Clean up
        process.stdout.close()
        if hasattr(process, "stderr") and process.stderr:
            process.stderr.close()

        # If process is still running
        if process.poll() is None:
            process.kill()
            process.wait()

    logging.info(
        f"Successfully processed {total_processed} frames from video '{video_name}' and stored the embeddings in the Milvus database collection."
    )

    return total_processed


def recreate_embeddings(video_name: str, sampling_fps: float = 1.0) -> int:
    """
    Deletes all embeddings for a specified video from the Milvus collection and recreates them.
    This is basically the same as synchronizing all data with force_bucket_mirror=True, but only for a single video.

    Args:
        video_name (str): The name of the video file in the Minio bucket.
        sampling_fps (float): How many frames to sample for each second of the video.

    Returns:
        int: The number of frames newly processed and stored in the Milvus database collection.
    """
    # Delete all embeddings for the video from the Milvus collection
    expr = f"video_name == '{video_name}'"
    collection.delete(expr=expr)
    collection.flush()
    logging.info(
        f"Deleted all embeddings for video '{video_name}' from the Milvus database collection."
    )

    # Extract and store new embeddings for the video
    return extract_and_store_embeddings(video_name, sampling_fps)


# This function has to be here (instead of db_and_storage.py) to avoid circular imports
def synchronize_embeddings_with_bucket(
    force_bucket_mirror: bool = False, sampling_fps: float = 1.0
):
    """
    Synchronizes the data in the Milvus collection with the videos in the Minio bucket.
    It deletes any entries in the Milvus collection that do not have a corresponding video in the bucket,
    and extracts and stores embeddings for any videos found in the bucket but not in the collection.

    In cases of configuring a different embedding model, or having lots of incomplete data in the Milvus database collection,
    use force_bucket_mirror=True to completely recreate the collection based on the current state of the bucket.

    Args:
        force_bucket_mirror (bool): If True, the Milvus collection will be recreated based on the current state of the bucket.
        sampling_fps (float): How many frames to sample for each second of a video.
    """
    minio_data, milvus_data = list_all_data()

    if force_bucket_mirror:
        # Delete all data in the Milvus collection by simply recreating it
        global collection
        collection.drop()
        collection = create_collection(COLLECTION_NAME)
        logging.info(
            f"Deleted and created new Milvus database collection '{COLLECTION_NAME}'."
        )

        # Insert all videos from the Minio bucket into the Milvus collection
        cnt = 0
        for video_name in minio_data.keys():
            cnt += 1
            logging.info(
                f"Processing video '{video_name}' number {cnt}/{len(minio_data)}..."
            )
            extract_and_store_embeddings(video_name, sampling_fps=sampling_fps)
        logging.info(
            f"Extracted and stored embeddings of all {len(minio_data)} videos from the Minio bucket into the Milvus database collection: {list(minio_data.keys())}"
        )
        return

    # Delete entries in Milvus that do not have a corresponding video in the Minio bucket
    milvus_exclusives = [
        video_name for video_name in milvus_data.keys() if video_name not in minio_data
    ]
    expr = f"video_name in {milvus_exclusives}"
    collection.delete(expr=expr)
    collection.flush()
    logging.info(
        f"Deleted all entries from the Milvus database collection for the following {len(milvus_exclusives)} videos not found in the Minio bucket: {milvus_exclusives}"
    )

    # Extract and store embeddings for videos found in Minio but not in Milvus
    minio_exclusives = [
        video_name for video_name in minio_data.keys() if video_name not in milvus_data
    ]
    cnt = 0
    for video_name in minio_exclusives:
        cnt += 1
        logging.info(
            f"Processing video '{video_name}' number {cnt}/{len(minio_exclusives)}..."
        )
        extract_and_store_embeddings(video_name, sampling_fps=sampling_fps)
    logging.info(
        f"Extracted and stored embeddings for the following {len(minio_exclusives)} videos found in the Minio bucket but not in the Milvus collection: {minio_exclusives}"
    )
