import torch
from PIL import Image
import cv2
import os

from src import device, emb_model, emb_processor
from src.db_and_storage import collection, minio_client, BUCKET_NAME, check_bucket_object_exists


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
    print(f"Video '{video_name}' uploaded to the Minio bucket.")

    return video_name


# Function to sample frames from a video and store the embeddings in the DB
def extract_and_store_embeddings(video_name: str, frame_interval: int = 1) -> int:
    """
    Extracts frames from a video stored in the bucket, generates embeddings,
    and stores them in the Milvus database together with corresponding metadata.

    Args:
        video_name (str): The name of the video file in the Minio bucket.
        frame_interval (int): The interval between frames to process [seconds].

    Returns:
        int: The number of frames processed and stored in the Milvus database.
    """
    # check that video exists in the Minio bucket
    if not check_bucket_object_exists(video_name):
        raise FileNotFoundError(f"Video file '{video_name}' not found in the Minio bucket.")

    # Download video (temporarily) from Minio bucket
    temp_video_path = f"./tmp/{video_name}"
    minio_client.fget_object(BUCKET_NAME, video_name, temp_video_path)
    print(f"Video '{video_name}' successfully downloaded from the Minio bucket.")

    # Open the video file using OpenCV
    video_capture = cv2.VideoCapture(temp_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = total_frames // (frame_interval * fps)
    ten_percent_frames = frames_to_extract // 10
    print(f"Video '{video_name}' has {total_frames} frames at {fps} FPS, sampling {frames_to_extract} frames at {1.0/float(frame_interval)} FPS...")

    # Process every frame at 'frame_interval' intervals (e.g., 1 frame per second)
    frame_idx = 0
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Calculate the timestamp of this frame
        timestamp = float(frame_idx) / float(fps)

        # Process the frame only if it's at the right interval
        if frame_idx % (fps * frame_interval) == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Process image and generate embedding using Multimodal embedding model
            if os.getenv("EMBEDDING_MODEL") == "Blip":
                inputs = emb_processor[0]["eval"](pil_image).unsqueeze(0).to(device)
                sample = {"image": inputs, "text_input": None}
                image_features = emb_model.extract_features(sample, mode="image")
                # project from 768 to 256 dimensions (includes normalization)
                image_features = image_features.image_embeds_proj[:, 0, :]
            else:
                inputs = emb_processor(images=pil_image, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    image_features = emb_model.get_image_features(**inputs)
                # Normalize the embedding
                image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

            # Convert the image embedding to 1-D numpy array
            embedding = image_features.cpu().numpy().flatten()

             # Insert into Milvus
            data = [[video_name], [timestamp], [embedding]]
            collection.insert(data)

            frame_count += 1

            # Print progress every 10 % of frames processed (if possible with long enough video)
            if ten_percent_frames and (frame_count % ten_percent_frames == 0):
                print(f"Processed {frame_count} frames ({frame_count / frames_to_extract:.0%}) from video {video_name}.")

        frame_idx += 1

    video_capture.release()
    os.remove(temp_video_path)
    print(f"Processed {frame_count} frames from video {video_name} and inserted them into the Milvus collection.")

    return frame_count


