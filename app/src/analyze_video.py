import ffmpeg
import datetime
from typing import List, Dict, Tuple, Optional
import torch
import os
from PIL import Image
import io
import base64
from openai import OpenAI, AzureOpenAI
from qwen_vl_utils import process_vision_info as qwen_process_vision_info

from src import device, mllm_model, mllm_processor
from src.db_and_storage import minio_client, check_bucket_object_exists, BUCKET_NAME
from src.utils import format_timestamp


def extract_frames_for_video_analysis(
    video_name: str, start_timestamp: float, end_timestamp: float, num_frames: int
) -> List[Tuple[bytes, float]]:
    """
    Extracts a set of frames from a video stored in the Minio bucket around a specified timestamp.

    Args:
        video_name (str): The name of the video file in the Minio bucket.
        start_timestamp (float): The timestamp in seconds marking the start of the sampling interval.
        end_timestamp (float): The timestamp in seconds marking the end of the sampling interval.
        num_frames (int): The number of frames to extract.
    Returns:
        List[Tuple[bytes, float]]: A list of tuples, each containing frame image bytes and the corresponding timestamp in seconds.
    """
    assert (
        start_timestamp < end_timestamp
    ), "The start timestamp must be before the end timestamp."

    if not check_bucket_object_exists(video_name):
        raise FileNotFoundError(
            f"Video file '{video_name}' not found in the Minio bucket."
        )

    # Generate a presigned URL for the video valid for 5 minutes
    url = minio_client.presigned_get_object(
        BUCKET_NAME, video_name, expires=datetime.timedelta(minutes=5)
    )

    # Probe the video to obtain metadata (including duration)
    try:
        probe = ffmpeg.probe(url)
        video_duration = float(probe["format"]["duration"])
        video_fps = float(probe["streams"][0]["r_frame_rate"].split("/")[0])
    except Exception as e:
        raise RuntimeError(f"Error retrieving video metadata: {e}")

    # Check if the timestamp is not out of bounds
    if start_timestamp < 0 or end_timestamp > video_duration:
        raise ValueError(
            f"Interval <{start_timestamp}, {end_timestamp}> seconds is out of bounds for video '{video_name}' lasting {video_duration} seconds."
        )

    sampling_interval_duration = end_timestamp - start_timestamp
    if sampling_interval_duration * video_fps < num_frames:
        raise ValueError(
            f"Cannot sample {num_frames} unique frames from interval <{start_timestamp}, {end_timestamp}> in a video that has {video_fps} FPS."
        )

    sampling_fps = float(num_frames) / sampling_interval_duration

    # Seek start_time in video and read for interval_duration seconds
    process = (
        ffmpeg.input(url, ss=start_timestamp, t=sampling_interval_duration)
        .output(
            "pipe:",
            format="image2pipe",
            vcodec="mjpeg",
            vf=f"fps={sampling_fps}",
            vframes=num_frames,
        )
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    out_bytes = process.stdout.read()
    process.wait()

    # Split the concatenated JPEG stream into individual frames
    frames = []
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"
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
    assert (
        len(frames) == num_frames
    ), f"Expected {num_frames} frames, but sampled {len(frames)} frames."

    # Calculate the timestamps for each frame
    sample_timestamps = [
        start_timestamp + i * (sampling_interval_duration / float(num_frames - 1))
        for i in range(num_frames)
    ]

    return list(zip(frames, sample_timestamps))


def create_initial_prompt(
    video_name: str, timestamps: List[float], user_query: str
) -> str:
    """
    Creates the initial prompt for a video analysis conversation with MLLM.
    This prompt includes the video name, the timestamps of the sampled frames, and the user's query.

    Args:
        video_name (str): The name of the video file.
        timestamps (List[float]): The list of timestamps (in seconds) for the sampled frames.
        user_query (str): The user's query for the assistant.

    Returns:
        str: The formatted initial prompt text.
    """
    return f"""You are a video analysis assistant inside a specialized system for analyzing traffic footage.
This prompt includes {len(timestamps)} images consecutively sampled from video named '{video_name}', each taken at a different timestamp. The timestamps (in the corresponding order and in HH:MM:SS.sss format) are: {', '.join([format_timestamp(ts) for ts in timestamps])}.
Your goal is to have a conversation with the user about the situation in video section shown to you, don't talk about anything else or make up information about the video.
Don't include any comments about your possibly limited video processing abilities, and do your best to interpret the images as a continous video.
Don't blindly agree to everything the user says unless you saw proof in the footage. Try to be helpful and informative.

This is the user's query: {user_query}"""


def perform_video_analysis(
    user_query: str,
    video_name: Optional[str] = None,
    start_timestamp: Optional[float] = None,
    end_timestamp: Optional[float] = None,
    num_frames: Optional[int] = None,
    existing_conversation: Optional[List[Dict]] = None,
) -> Tuple[str, List[Dict]]:
    """
    Starts or continues a video-analysis conversation with the configured MLLM.
    New conversations require the user's query, the video name, the start and end timestamps, and the number of frames to sample.
    Existing conversations require the user's query and the conversation history to continue from.

    Args:
        user_query (str): The user's query for the assistant.
        video_name (str): The name of the video file in the Minio bucket.
        start_timestamp (float): The timestamp in seconds marking the start of the sampling interval.
        end_timestamp (float): The timestamp in seconds marking the end of the sampling interval.
        num_frames (int): The number of frames to extract.
        existing_conversation (List[Dict]): The existing conversation history to continue from.

    Returns:
        Tuple[str, List[Dict]]: The assistant's response text and the updated conversation history.
    """
    current_MLLM = os.getenv("MLLM")

    # Extract frames and create initial prompt when starting a new conversation
    if existing_conversation is None:
        if None in (video_name, start_timestamp, end_timestamp, num_frames):
            raise ValueError(
                "The video name, the start and end timestamps, and the number of frames to sample must be provided for a new video analysis conversation."
            )
        timestamped_frames = extract_frames_for_video_analysis(
            video_name, start_timestamp, end_timestamp, num_frames
        )
        initial_prompt_text = create_initial_prompt(
            video_name, [ts for _, ts in timestamped_frames], user_query
        )

        # Problem: Some models cannot remember/store previously seen video (images) in the conversation
        # Hack: Store metadata in the conversation so the frames can be resampled every time
        if current_MLLM in ("LLaVA-OneVision", "VideoLLaMA-3", "Qwen2.5-VL"):
            conversation_hack = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "metadata",
                            "video_name": video_name,
                            "start_timestamp": start_timestamp,
                            "end_timestamp": end_timestamp,
                            "num_frames": num_frames,
                        },
                    ],
                },
            ]

    if current_MLLM == "LLaVA-OneVision":
        if existing_conversation is not None:
            return chat_with_llava_onevision(
                user_query, conversation=existing_conversation
            )
        return chat_with_llava_onevision(
            initial_prompt_text,
            new_conversation=True,
            timestamped_frames=timestamped_frames,
            conversation=conversation_hack,
        )

    elif current_MLLM == "GPT-4o":
        if existing_conversation is not None:
            return chat_with_gpt4o(user_query, conversation=existing_conversation)
        return chat_with_gpt4o(
            initial_prompt_text,
            new_conversation=True,
            timestamped_frames=timestamped_frames,
        )

    elif current_MLLM == "VideoLLaMA-3":
        if existing_conversation is not None:
            return chat_with_videollama_3(
                user_query, conversation=existing_conversation
            )
        return chat_with_videollama_3(
            initial_prompt_text,
            new_conversation=True,
            timestamped_frames=timestamped_frames,
            conversation=conversation_hack,
            video_name=video_name,
        )

    elif current_MLLM == "Qwen2.5-VL":
        if existing_conversation is not None:
            return chat_with_qwen2_5_vl(user_query, conversation=existing_conversation)
        return chat_with_qwen2_5_vl(
            initial_prompt_text,
            new_conversation=True,
            timestamped_frames=timestamped_frames,
            conversation=conversation_hack,
            video_name=video_name,
        )

    else:
        raise NotImplementedError(
            f"The chatting function for the configured MLLM '{current_MLLM}' is not implemented yet."
        )


def chat_with_llava_onevision(
    prompt_text: str,
    conversation: List[Dict] = [],
    new_conversation: bool = False,
    timestamped_frames: List[Tuple[bytes, float]] = [],
) -> Tuple[str, List[Dict]]:
    """
    Chat with the LLaVA-OneVision model using the provided prompt text, conversation history, and video frames.

    Args:
        prompt_text (str): The user's query for the assistant.
        conversation (List[Dict]): The conversation history to continue from.
        new_conversation (bool): Whether the conversation is new or continuing.
        timestamped_frames (List[Tuple[bytes, float]]): The list of frame bytes and corresponding timestamps (new conversation only).

    Returns:
        Tuple[str, List[Dict]]: The assistant's response text and the updated conversation history.
    """
    if not new_conversation and conversation and conversation[0]["role"] == "system":
        # Conversation that continues must provide the video segment metadata in conversation history
        video_segment_metadata = conversation[0]["content"][0]
        timestamped_frames = extract_frames_for_video_analysis(
            video_segment_metadata["video_name"],
            video_segment_metadata["start_timestamp"],
            video_segment_metadata["end_timestamp"],
            video_segment_metadata["num_frames"],
        )
    elif not new_conversation:
        raise ValueError(
            "Cannot continue conversation with LLaVA-OneVision without the video segment metadata as the first conversation entry."
        )

    # Convert JPEG byte strings to PIL images
    frames = [
        Image.open(io.BytesIO(frame_bytes)) for frame_bytes, _ in timestamped_frames
    ]

    conversation.append(
        {
            "role": "user",
            "content": [
                # {"type": "video"}, <- This will be inserted for new conversation only
                {"type": "text", "text": prompt_text},
            ],
        }
    )
    if new_conversation:
        conversation[-1]["content"].insert(0, {"type": "video"})
    prompt = mllm_processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = mllm_processor(videos=[frames], text=prompt, return_tensors="pt").to(
        device, torch.float16
    )

    output = mllm_model.generate(**inputs, max_new_tokens=512)
    decoded_output = mllm_processor.batch_decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    assistant_text_response = decoded_output.split("assistant\n", 1)[1]

    # Append the assistant's response to the conversation
    conversation.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text_response},
            ],
        }
    )
    return assistant_text_response, conversation


def chat_with_gpt4o(
    prompt_text: str,
    conversation: List[Dict] = [],
    new_conversation: bool = False,
    timestamped_frames: List[Tuple[bytes, float]] = [],
) -> Tuple[str, List[Dict]]:
    """
    Chat with the GPT-4o model using the provided prompt text, conversation history, and video frames.

    Args:
        prompt_text (str): The user's query for the assistant.
        conversation (List[Dict]): The conversation history to continue from.
        new_conversation (bool): Whether the conversation is new or continuing.
        timestamped_frames (List[Tuple[bytes, float]]): The list of frame bytes and corresponding timestamps (new conversation only).

    Returns:
        Tuple[str, List[Dict]]: The assistant's response text and the updated conversation history.
    """
    if new_conversation:
        # Convert JPEG byte strings to base64-encoded image URLs
        base64_frames = [
            f"data:image/jpeg;base64,{base64.b64encode(frame_bytes).decode('utf-8')}"
            for frame_bytes, _ in timestamped_frames
        ]

    if os.getenv("OPENAI_USE_AZURE", "false").lower() == "true":
        openai_client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_CUSTOM_ENDPOINT"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
        )
    else:
        openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=(
                os.getenv("OPENAI_CUSTOM_ENDPOINT")
                if (
                    os.getenv("OPENAI_CUSTOM_ENDPOINT")
                    and os.getenv("OPENAI_CUSTOM_ENDPOINT").lower() != "none"
                )
                else None
            ),
        )

    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                # *[{"type": "image_url", "image_url": {"url": img}} for img in base64_frames] <- This will be inserted for new conversation only
            ],
        }
    )
    if new_conversation:
        conversation[-1]["content"].extend(
            [{"type": "image_url", "image_url": {"url": img}} for img in base64_frames]
        )

    params = {
        "model": (
            os.getenv("OPENAI_AZURE_DEPLOYMENT_NAME")
            if os.getenv("OPENAI_USE_AZURE", "false").lower() == "true"
            else "gpt-4o"
        ),
        "messages": conversation,
        "max_tokens": 512,
    }
    output = openai_client.chat.completions.create(**params)

    # Append the assistant's response to the conversation
    conversation.append(
        {
            "role": output.choices[0].message.role,
            "content": [
                {"type": "text", "text": output.choices[0].message.content},
            ],
        }
    )
    return output.choices[0].message.content, conversation


def chat_with_videollama_3(
    prompt_text: str,
    conversation: List[Dict] = [],
    new_conversation: bool = False,
    timestamped_frames: List[Tuple[bytes, float]] = [],
    video_name: Optional[str] = None,
) -> Tuple[str, List[Dict]]:
    """
    Chat with the VideoLLaMA 3 model using the provided prompt text, conversation history, and video frames.

    Args:
        prompt_text (str): The user's query for the assistant.
        conversation (List[Dict]): The conversation history to continue from.
        new_conversation (bool): Whether the conversation is new or continuing.
        timestamped_frames (List[Tuple[bytes, float]]): The list of frame bytes and corresponding timestamps (new conversation only).
        video_name (str): The name of the video file the chat is about (new conversation only).

    Returns:
        Tuple[str, List[Dict]]: The assistant's response text and the updated conversation history.
    """
    if not new_conversation and conversation and conversation[0]["role"] == "system":
        # Conversation that continues must provide the video segment metadata in conversation history
        video_segment_metadata = conversation[0]["content"][0]
        timestamped_frames = extract_frames_for_video_analysis(
            video_segment_metadata["video_name"],
            video_segment_metadata["start_timestamp"],
            video_segment_metadata["end_timestamp"],
            video_segment_metadata["num_frames"],
        )
    elif not new_conversation:
        raise ValueError(
            "Cannot continue conversation with VideoLLaMA 3 without the video segment metadata as the first conversation entry."
        )

    video_name = (
        video_name if new_conversation else video_segment_metadata["video_name"]
    )
    if not video_name:
        raise ValueError(
            "The video name must be provided for chatting with VideoLLaMA 3."
        )
    # Convert JPEG byte strings to PIL images
    frames = [
        Image.open(io.BytesIO(frame_bytes)) for frame_bytes, _ in timestamped_frames
    ]
    timestamps = [ts for _, ts in timestamped_frames]
    # Temporarily store the frames as "{video_name}_timestamp.jpg" to tmp folder
    # Even though they will be deleted, they can be recondstructed again with the same file names easily
    tmp_folder = "./tmp"
    tmp_frame_paths = []
    for i, frame in enumerate(frames):
        tmp_frame_path = os.path.join(
            tmp_folder, f"{video_name}_{timestamps[i]:.3f}.jpg"
        )
        frame.save(tmp_frame_path, format="JPEG")
        tmp_frame_paths.append(tmp_frame_path)

    content = []
    # Only the first message from user contains reference to the video frames (images)
    if new_conversation:
        for frame_path in tmp_frame_paths:
            content.append({"type": "image", "image": {"image_path": frame_path}})
    content.append({"type": "text", "text": prompt_text})
    conversation.append(
        {
            "role": "user",
            "content": content,
        }
    )

    inputs = mllm_processor(conversation=conversation, return_tensors="pt")
    inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = mllm_model.generate(**inputs, max_new_tokens=512)
    assistant_text_response = mllm_processor.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    # Delete the temporary files
    for tmp_frame_path in tmp_frame_paths:
        os.remove(tmp_frame_path)

    # Append the assistant's response to the conversation
    conversation.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text_response},
            ],
        }
    )
    return assistant_text_response, conversation


def chat_with_qwen2_5_vl(
    prompt_text: str,
    conversation: List[Dict] = [],
    new_conversation: bool = False,
    timestamped_frames: List[Tuple[bytes, float]] = [],
    video_name: Optional[str] = None,
) -> Tuple[str, List[Dict]]:
    """
    Chat with the Qwen-2.5-VL model using the provided prompt text, conversation history, and video frames.

    Args:
        prompt_text (str): The user's query for the assistant.
        conversation (List[Dict]): The conversation history to continue from.
        new_conversation (bool): Whether the conversation is new or continuing.
        timestamped_frames (List[Tuple[bytes, float]]): The list of frame bytes and corresponding timestamps (new conversation only).
        video_name (str): The name of the video file the chat is about (new conversation only).

    Returns:
        Tuple[str, List[Dict]]: The assistant's response text and the updated conversation history.
    """
    if not new_conversation and conversation and conversation[0]["role"] == "system":
        # Conversation that continues must provide the video segment metadata in conversation history
        video_segment_metadata = conversation[0]["content"][0]
        timestamped_frames = extract_frames_for_video_analysis(
            video_segment_metadata["video_name"],
            video_segment_metadata["start_timestamp"],
            video_segment_metadata["end_timestamp"],
            video_segment_metadata["num_frames"],
        )
    elif not new_conversation:
        raise ValueError(
            "Cannot continue conversation with Qwen2.5-VL without the video segment metadata as the first conversation entry."
        )

    video_name = (
        video_name if new_conversation else video_segment_metadata["video_name"]
    )
    if not video_name:
        raise ValueError(
            "The video name must be provided for chatting with Qwen2.5-VL."
        )
    # Convert JPEG byte strings to PIL images
    frames = [
        Image.open(io.BytesIO(frame_bytes)) for frame_bytes, _ in timestamped_frames
    ]
    timestamps = [ts for _, ts in timestamped_frames]
    # Temporarily store the frames as "{video_name}_timestamp.jpg" to tmp folder
    # Even though they will be deleted, they can be recondstructed again with the same file names easily
    tmp_folder = "./tmp"
    tmp_frame_paths = []
    for i, frame in enumerate(frames):
        tmp_frame_path = os.path.join(
            tmp_folder, f"{video_name}_{timestamps[i]:.3f}.jpg"
        )
        frame.save(tmp_frame_path, format="JPEG")
        tmp_frame_paths.append(tmp_frame_path)

    content = []
    # Only the first message from user contains reference to the video frames (images)
    if new_conversation:
        for frame_path in tmp_frame_paths:
            content.append({"type": "image", "image": frame_path})
    content.append({"type": "text", "text": prompt_text})
    conversation.append(
        {
            "role": "user",
            "content": content,
        }
    )

    text = mllm_processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_process_vision_info(conversation)
    inputs = mllm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = mllm_model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    assistant_text_response = mllm_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Delete the temporary files
    for tmp_frame_path in tmp_frame_paths:
        os.remove(tmp_frame_path)

    # Append the assistant's response to the conversation
    conversation.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text_response},
            ],
        }
    )
    return assistant_text_response, conversation
