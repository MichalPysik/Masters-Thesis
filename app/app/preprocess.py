import torch
from PIL import Image
import cv2
import os

from app import device, model, processor, collection


def get_image_text_similarity(image_path: str, text: str) -> float:
    """
    Computes the cosine similarity between an image and a text description using CLIP.

    Args:
        image_path (str): Path to the input image.
        text (str): Text description to compare with the image.

    Returns:
        float: Cosine similarity score between the image and the text.
    """
    # Preprocess the image and text
    inputs = processor(
        text=[text], 
        images=Image.open(image_path), 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # Extract features from the image and text
    with torch.no_grad():
        image_features = model.get_image_features(inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        )
    
    # Normalize features to unit vectors
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Debug: Print features
    print("Image Features Shape:", image_features.shape)
    print("Text Features Shape:", text_features.shape)
    print("Image Features:", image_features)
    print("Text Features:", text_features)

    # Compute cosine similarity
    similarity = torch.matmul(image_features, text_features.T).item()
    
    return similarity


# Function to sample frames from the video and store embeddings
def extract_video_frames(video_path: str, frame_interval: int = 1):
    # check that video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open the video file using OpenCV
    video_capture = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)  # Use video file name for identification
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

    print(f"Video '{video_name}' has {total_frames} frames at {fps} fps.")

    frame_idx = 0
    frame_count = 0

    # Process every frame at 'frame_interval' intervals (e.g., 1 frame per second)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Calculate the timestamp of this frame
        timestamp = float(frame_idx) / float(fps)

        # Process the frame only if it's at the right interval
        if frame_idx % (fps * frame_interval) == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=pil_image, return_tensors="pt", padding=True).to(device)

            # Generate embedding using CLIP
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            # Normalize the embedding
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

            # Insert into Milvus
            embedding = image_features.cpu().numpy().flatten()
            data = [
                [video_name],   # Video name field
                [timestamp],    # Timestamp field
                [embedding]     # CLIP embedding
            ]
            collection.insert(data)

            print(f"Frame {frame_count} processed and inserted into Milvus.")

            frame_count += 1

        frame_idx += 1

    video_capture.release()
    print(f"Processed {frame_count} frames from video {video_name} and inserted them into Milvus.")



