import torch
from PIL import Image
import os
import datetime
import ffmpeg
import logging


from src import device, emb_model, emb_processor
from src.db_and_storage import collection, minio_client, BUCKET_NAME, COLLECTION_NAME, check_bucket_object_exists, list_all_data, create_collection
from src.utils import format_timestamp


# Function to upload video to minio bucket
def upload_video_to_bucket(video_path: str) -> str:
    """
    Uploads a video file to the Minio bucket and returns the video name.
    
    Args:
        video_path (str): Path to the video file in the local filesystem.
    
    Returns:
        str: The name of the video file in the Minio bucket.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}.")
    
    video_name = os.path.basename(video_path)

    # Check if video already exists in Minio bucket
    if check_bucket_object_exists(video_name):
        raise FileExistsError(f"Video file '{video_name}' already exists in the Minio bucket. Plese delete the existing video from the system first, or rename the local file if the name conflict is coincidental.")

    # Upload video to Minio bucket
    minio_client.fput_object(BUCKET_NAME, video_name, video_path)
    logging.info(f"Video '{video_name}' was uploaded to the Minio bucket.")

    return video_name


# Function to sample frames from a video and store the embeddings in the DB
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
    # Check that video exists in the Minio bucket
    if not check_bucket_object_exists(video_name):
        raise FileNotFoundError(f"Video file '{video_name}' not found in the Minio bucket.")

    # Generate a presigned URL for the video valid for 1 hour
    url = minio_client.presigned_get_object(BUCKET_NAME, video_name, expires=datetime.timedelta(hours=1))

    # Probe the video to get metadata (duration, fps, total_frames, width, height)
    try:
        probe = ffmpeg.probe(url)
        video_duration = float(probe['format']['duration'])
        video_fps = float(probe['streams'][0]['r_frame_rate'].split('/')[0])
        video_total_frames = int(probe['streams'][0]['nb_frames'])
        video_width = int(probe['streams'][0]['width'])
        video_height = int(probe['streams'][0]['height'])
    except Exception as e:
        raise RuntimeError(f"Error retrieving video metadata: {e}")
    
    logging.info(f"Video '{video_name}' has duration {format_timestamp(video_duration)} ({video_duration} seconds), {video_total_frames} frames total at {video_fps} FPS, and resolution {video_width}x{video_height}.")
    
    # Use ffmpeg to extract frames at the desired fps,
    # outputs a stream of raw video frames in RGB format
    try:
        out, err = (
            ffmpeg
            .input(url)
            .filter('fps', fps=sampling_fps)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        error_message = e.stderr.decode()
        raise RuntimeError(f"Error extracting frames from '{video_name}': {error_message}")
    
    # Calculate the size of a single frame (width x height x 3 bytes for RGB)
    frame_size = video_width * video_height * 3
    num_frames = len(out) // frame_size

    # Convert output to PIL images and calculate timestamps
    frames = [Image.frombytes('RGB', (video_width, video_height), out[i*frame_size:(i+1)*frame_size]) for i in range(num_frames)]
    assert len(frames) == num_frames, f"Number of frames extracted ({len(frames)}) does not match expected ({num_frames})."
    timestamps = [float(i) / sampling_fps for i in range(num_frames)]
    logging.info(f"Sampled {num_frames} frames at {sampling_fps} FPS, starting embedding generation...")

    # Step for reporting progress every 10 % of frames processed
    progress_step = max(num_frames // 10, 1)

    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        # Process image and generate embedding using Multimodal embedding model
        if os.getenv("EMBEDDING_MODEL") == "Blip":
            inputs = emb_processor[0]["eval"](frame).unsqueeze(0).to(device)
            sample = {"image": inputs, "text_input": None}
            image_features = emb_model.extract_features(sample, mode="image")
            # project from 768 to 256 dimensions (includes normalization)
            image_features = image_features.image_embeds_proj[:, 0, :]
        else:
            inputs = emb_processor(images=frame, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = emb_model.get_image_features(**inputs)
            # Normalize the embedding
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        # Convert the image embedding to 1-D numpy array
        embedding = image_features.cpu().numpy().flatten()

         # Insert into Milvus
        data = [[video_name], [timestamp], [embedding]]
        collection.insert(data)

        # Report progress every 10 % (or on the last frame)
        if (i + 1) % progress_step == 0 or (i + 1) == num_frames:
            progress_percent = (i + 1) * 100 / num_frames
            logging.info(f"Processed {i + 1} of {num_frames} frames ({progress_percent:.0f} % complete)...")

    logging.info(f"Successfully processed {num_frames} frames from video '{video_name}' and stored the embeddings in the Milvus database collection.")
    return num_frames


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
    logging.info(f"Deleted all embeddings for video '{video_name}' from the Milvus database collection.")

    # Extract and store new embeddings for the video
    return extract_and_store_embeddings(video_name, sampling_fps)



# This function has to be here (instead of db_and_storage.py) to avoid circular imports
def synchronize_all_data(force_bucket_mirror: bool = False):
    """
    Synchronizes the data in the Milvus collection with the videos in the Minio bucket.
    It deletes any entries in the Milvus collection that do not have a corresponding video in the bucket,
    and extracts and stores embeddings for any videos found in the bucket but not in the collection.

    In cases of configuring a different embedding model, or having lots of incomplete data in the Milvus database collection,
    use force_bucket_mirror=True to completely recreate the collection based on the current state of the bucket.

    Args:
        force_bucket_mirror (bool): If True, the Milvus collection will be recreated based on the current state of the bucket.
    """
    minio_data, milvus_data = list_all_data()

    if force_bucket_mirror:
        # Delete all data in the Milvus collection by simply recreating it
        global collection
        collection.drop()
        collection = create_collection(COLLECTION_NAME)
        logging.info(f"Deleted and created new Milvus database collection '{COLLECTION_NAME}'.")

        # Insert all videos from the Minio bucket into the Milvus collection
        cnt = 0
        for video_name in minio_data.keys():
            cnt += 1
            logging.info(f"Processing video '{video_name}' number {cnt}/{len(minio_data)}...")
            extract_and_store_embeddings(video_name)
        logging.info(f"Extracted and stored embeddings of all {len(minio_data)} videos from the Minio bucket into the Milvus database collection: {list(minio_data.keys())}")
        return

    # Delete entries in Milvus that do not have a corresponding video in the Minio bucket
    milvus_exclusives = [video_name for video_name in milvus_data.keys() if video_name not in minio_data]
    expr = f"video_name in {milvus_exclusives}"
    collection.delete(expr=expr)
    collection.flush()
    logging.info(f"Deleted all entries from the Milvus database collection for the following {len(milvus_exclusives)} videos not found in the Minio bucket: {milvus_exclusives}")

    # Extract and store embeddings for videos found in Minio but not in Milvus
    minio_exclusives = [video_name for video_name in minio_data.keys() if video_name not in milvus_data]
    cnt = 0
    for video_name in minio_exclusives:
        cnt += 1
        logging.info(f"Processing video '{video_name}' number {cnt}/{len(minio_exclusives)}...")
        extract_and_store_embeddings(video_name)
    logging.info(f"Extracted and stored embeddings for the following {len(minio_exclusives)} videos found in the Minio bucket but not in the Milvus collection: {minio_exclusives}")

