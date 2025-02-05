import json
from openai import OpenAI, AzureOpenAI
import os
import cv2
import base64
import time
import requests
import numpy as np
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoModel, BitsAndBytesConfig
import av


client = AzureOpenAI(
        api_key="",
        azure_endpoint="https://lakmoosgpt.openai.azure.com/",
        api_version="2024-02-01",
)

# specify how to quantize the LLaVA
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model in half-precision
model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config)
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")


def read_and_parse_annotations():
    dataset_dir = "../../../dp-demos/datasets/mllm/SUTD-TrafficQA"
    with open(f"{dataset_dir}/annotations/R2_all.jsonl", "r") as f:
        lines = f.readlines()

    # extract legend and data
    legend = json.loads(lines[0])
    print(legend)
    raw_data = [json.loads(line) for line in lines[1:]]
    structured_data = [dict(zip(legend, row)) for row in raw_data]

    return structured_data


def sample_uniform_frames_from_video_base64(video_path, num_frames = 12):
    video = cv2.VideoCapture(video_path)
    total_frame_cnt = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frame_cnt < num_frames:
        num_frames = total_frame_cnt
    print(f"Video {video_path} has {total_frame_cnt} frames. Sampling {num_frames} frames.")

    frame_indices = np.linspace(0, total_frame_cnt - 1, num_frames, dtype=int)
    base64_frames = []
    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_string = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(f"data:image/jpeg;base64,{base64_string}")
    video.release()

    return base64_frames


def sample_uniform_frames_from_video_pyav(video_path, num_frames = 12):
    container = av.open(video_path)
    total_frame_cnt = container.streams.video[0].frames
    if total_frame_cnt < num_frames:
        num_frames = total_frame_cnt
    print(f"Video {video_path} has {total_frame_cnt} frames. Sampling {num_frames} frames.")

    frame_indices = np.linspace(0, total_frame_cnt - 1, num_frames, dtype=int)
    frames = []
    container.seek(0)
    start_index = frame_indices[0]
    end_index = frame_indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in frame_indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_substring_after(text, substring):
    index = text.find(substring)
    if index != -1:
        return text[index + len(substring):]  # Extract part after substring
    return ""  # Return empty string if substring is not found


data = read_and_parse_annotations()

total = 50
correct = 0

for sample in data[:total]:
    video_path = f"../../../dp-demos/datasets/mllm/SUTD-TrafficQA/raw_videos/compressed_videos/{sample['vid_filename']}"
    #base64_frames = sample_uniform_frames_from_video_base64(video_path, num_frames=12)
    frames = sample_uniform_frames_from_video_pyav(video_path, num_frames=12)

    question = sample["q_body"]
    options_str = ""
    for i in [0,1,2,3]:
        opt = sample[f"option{i}"]
        if opt != "":
            options_str += f"Option {i}: {opt}\n"

    query_text = f"""You just saw consecutive frames sampled from a video. Please answer the following question about the video: {question}
    You must choose an answer from the following options:\n{options_str}
    Only answer either '0', '1', '2', or '3' (if there is such option corresponding to the number)."""    

    #PROMPT_MESSAGES = [
    #    {
    #        "role": "user",
    #        "content": [
    #            {"type": "text", "text": query_text},
    #            *[{"type": "image_url", "image_url": {"url": img}} for img in base64_frames],
    #        ],
    #    },
    #]
    #params = {
    #    "model": "Lakmoos-gpt4-o",
    #    "messages": PROMPT_MESSAGES,
    #    "max_tokens": 200,
    #}

    #result = client.chat.completions.create(**params)
    #llm_answer = result.choices[0].message.content

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": query_text},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(videos=list(frames), text=prompt, return_tensors="pt").to("cuda:0", torch.float16)

    out = model.generate(**inputs, max_new_tokens=60)
    llm_answer = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    llm_answer = get_substring_after(llm_answer[0], "assistant\n")

    print(f"Video: {sample['vid_filename']}, Question: {question}, Options: {options_str}, Correct_answer: {sample['answer']}, LLM_answer: {llm_answer}")
    print("\n\n")

    # look for first number (0, 1, 2, 3) in the answer
    llm_choice = "9"
    for char in llm_answer.lower():
        if char in ["0", "1", "2", "3"]:
            llm_choice = char
            break

    if llm_choice == str(sample["answer"]):
        correct += 1


print(f"Total: {total}, Correct: {correct}, Accuracy: {float(correct)/float(total)}")