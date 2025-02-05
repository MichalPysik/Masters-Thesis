import ffmpeg
import datetime
from typing import List, Tuple
import torch
import os
from PIL import Image
import io
import base64
from openai import OpenAI, AzureOpenAI

from src import device, mllm_model, mllm_processor
from src.db_and_storage import minio_client, check_bucket_object_exists, BUCKET_NAME
from src.utils import format_timestamp


# Function to extract video frames for MLLM to analyze
def extract_frames_for_video_analysis(video_name: str, start_timestamp: float, end_timestamp : float, num_frames : int = 16) -> List[Tuple[bytes, float]]:
    """
    Extracts a set of frames from a video stored in the Minio bucket around a specified timestamp.

    Args:
        video_name (str): The name of the video file in the bucket.
        start_timestamp (float): The timestamp in seconds marking the start of the sampling interval.
        end_timestamp (float): The timestamp in seconds marking the end of the sampling interval.
        num_frames (int): The number of frames to extract (automatically gets smaller when needed).

    Returns:
        List[Tuple[bytes, float]]: A list of tuples, each containing frame image bytes and the corresponding timestamp in seconds.
    """
    assert start_timestamp < end_timestamp, "The start timestamp must be before the end timestamp."

    if not check_bucket_object_exists(video_name):
        raise FileNotFoundError(f"Video file '{video_name}' not found in the Minio bucket.")
    
    # Generate a presigned URL for the video valid for 5 minutes
    url = minio_client.presigned_get_object(BUCKET_NAME, video_name, expires=datetime.timedelta(minutes=5))

    # Probe the video to obtain metadata (including duration)
    try:
        probe = ffmpeg.probe(url)
        video_duration = float(probe['format']['duration'])
        video_fps = float(probe['streams'][0]['r_frame_rate'].split('/')[0])
        print(f"Video duration: {video_duration} seconds at {video_fps} FPS.")
    except Exception as e:
        raise RuntimeError(f"Error retrieving video metadata: {e}")
    
    # Check if the timestamp is not out of bounds
    if start_timestamp < 0 or end_timestamp > video_duration:
        raise ValueError(f"Interval <{start_timestamp}, {end_timestamp}> seconds is out of bounds for video '{video_name}' lasting {video_duration} seconds.")

    sampling_interval_duration = end_timestamp - start_timestamp
    if sampling_interval_duration * video_fps < num_frames:
        print(f"Adjusting the number of sampled frames from {num_frames} to {max(int(sampling_interval_duration * video_fps), 1)}.")
        num_frames = int(sampling_interval_duration * video_fps)
        # Very rare edge case where the interval is too short for even one frame
        if num_frames == 0:
            num_frames = 1
            sampling_interval_duration = 1.0 / video_fps

    sampling_fps = float(num_frames) / sampling_interval_duration

    # Seek start_time in video and read for interval_duration seconds
    process = (
        ffmpeg
        .input(url, ss=start_timestamp, t=sampling_interval_duration)
        .output("pipe:", format="image2pipe", vcodec="mjpeg", vf=f"fps={sampling_fps}", vframes=num_frames)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    out_bytes = process.stdout.read()
    process.wait()

    # Split the concatenated JPEG stream into individual frames
    frames = []
    start_marker = b'\xff\xd8'
    end_marker = b'\xff\xd9'
    pos = 0
    while True:
        start_idx = out_bytes.find(start_marker, pos)
        if start_idx == -1:
            break
        end_idx = out_bytes.find(end_marker, start_idx)
        if end_idx == -1:
            break
        end_idx += len(end_marker)
        frames.append(out_bytes[start_idx:end_idx])
        pos = end_idx
    assert len(frames) == num_frames, f"Expected {num_frames} frames, but sampled {len(frames)} frames."

    # Calculate the timestamps for each frame
    sample_timestamps = [
        start_timestamp + i * (sampling_interval_duration / float(num_frames - 1))
        for i in range(num_frames)
    ]

    return list(zip(frames, sample_timestamps))


def create_initial_prompt(video_name: str, timestamps: List[float], user_query: str) -> str:
    """
    Creates the initial prompt for a video analysis conversation with MLLM.
    This prompt includes the video name, the timestamps of the frames, and the user's query.

    Args:
        video_name (str): The name of the video file.
        timestamps (List[float]): The list of timestamps (in seconds) for the frames.
        user_query (str): The user's query to the system.

    Returns:
        str: The formatted initial prompt text.
    """
    return f"""You are a video analysis assistant inside a specialized system for analyzing traffic footage.
This prompt includes {len(timestamps)} images consecutively sampled from video named '{video_name}', each taken at a different timestamp. The timestamps (in the same ascending order) are: {', '.join([format_timestamp(ts) for ts in timestamps])}.
Your goal is to have a conversation with the user about the situation in video section shown to you, don't talk about anything else or make up information about the video.
Don't include any comments about your possibly limited video processing abilities, and do your best to interpret the images as a continous video.
Don't blindly agree to everything the user says unless you saw proof in the footage. Try to be helpful and informative.

This is the user's query: {user_query}"""


# Function to start chatting with MLLM about video (including showing it the frames)
def perform_video_analysis(user_query: str, video_name: str, start_timestamp: float, end_timestamp: float, num_frames: int = 16):
    timestamped_frames = extract_frames_for_video_analysis(video_name, start_timestamp, end_timestamp, num_frames)
    initial_prompt_text = create_initial_prompt(video_name, [ts for _, ts in timestamped_frames], user_query)

    current_MLLM = os.getenv("MLLM_MODEL")

    if current_MLLM == "LLaVA-OneVision":
        # Convert JPEG byte strings to PIL images
        frames = [Image.open(io.BytesIO(frame_bytes)) for frame_bytes, _ in timestamped_frames]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": initial_prompt_text},
                ],
            },
        ]
        initial_prompt = mllm_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = mllm_processor(videos=[frames], text=initial_prompt, return_tensors="pt").to(device, torch.float16)

        output = mllm_model.generate(**inputs, max_new_tokens=500)
        decoded_output = mllm_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return decoded_output.split("assistant\n", 1)[1]
    
    elif current_MLLM == "GPT-4o":
        # Convert JPEG byte strings to base64-encoded image URLs
        base64_frames = [f"data:image/jpeg;base64,{base64.b64encode(frame_bytes).decode('utf-8')}" for frame_bytes, _ in timestamped_frames]

        if os.getenv("OPENAI_USE_AZURE", "false").lower() == "true":
            openai_client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                azure_endpoint=os.getenv("OPENAI_CUSTOM_ENDPOINT"),
                api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            )
        else:
            # TODO check for custom endpoint
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What happens in the video you just saw?"},
                    *[{"type": "image_url", "image_url": {"url": img}} for img in base64_frames],
                ],
            },
        ]
        params = {
            "model": os.getenv("OPENAI_AZURE_DEPLOYMENT_NAME") if os.getenv("OPENAI_AZURE_DEPLOYMENT_NAME", "false").lower() == "true" else "gpt-4o",
            "messages": conversation,
            "max_tokens": 500,
        }

        output = openai_client.chat.completions.create(**params)
        return output.choices[0].message.content
    
    else:
        raise NotImplementedError("This function is not implemented for the configured MLLM yet.")


# Function to continue chatting with MLLM about video
def continue_video_analysis():
    pass