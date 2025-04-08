import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI, AzureOpenAI, BadRequestError
import data
from PIL import Image
import io
import base64
import os
import json
from qwen_vl_utils import process_vision_info as qwen_process_vision_info

# 0 = LLaVA-OneVision, 1 = GPT-4o, 2 = VideoLLaMA3, 3 = Qwen2.5-VL - Configure!
ACTIVE_MLLM = 3
MLLM_NAMES = ["LLaVA-OneVision", "GPT-4o", "VideoLLaMA-3", "Qwen2.5-VL"]


# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load selected MLLM
model, processor, openai_client, azure_deployment_name = None, None, None, None
if ACTIVE_MLLM in [0, 2]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

if ACTIVE_MLLM == 0:
    llava_onevision_version = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    mllm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        llava_onevision_version,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    mllm_processor = AutoProcessor.from_pretrained(llava_onevision_version)

elif ACTIVE_MLLM == 1:
    openai_client = AzureOpenAI(
        api_key="",
        azure_endpoint="https://lakmoosgpt.openai.azure.com/",
        api_version="2024-02-01",
    )
    azure_deployment_name = "Lakmoos-gpt4-o"
    
elif ACTIVE_MLLM == 2:
    videollama_3_version = "DAMO-NLP-SG/VideoLLaMA3-7B-Image"
    mllm_model = AutoModelForCausalLM.from_pretrained(
        videollama_3_version,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    mllm_processor = AutoProcessor.from_pretrained(
        videollama_3_version, trust_remote_code=True
    )

elif ACTIVE_MLLM == 3:
    from transformers import Qwen2_5_VLForConditionalGeneration
    qwen2_5_vl_version = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    mllm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen2_5_vl_version,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # Lower max_pixels to avoid torch.OutOfMemoryError (not enough VRAM)
    mllm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=(256 * 28 *28), max_pixels=(600 * 28 * 28))

else:
    raise ValueError("Invalid MLLM selection. Choose 0, 1, 2, or 3.")



def prompt_llava_onevision(prompt_text, video_frames):
    # Convert JPEG byte strings to PIL images
    images = [Image.open(io.BytesIO(frame)) for frame in video_frames]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = mllm_processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = mllm_processor(videos=[images], text=prompt, return_tensors="pt").to(
        device, torch.float16
    )

    output = mllm_model.generate(**inputs, max_new_tokens=128)
    decoded_output = mllm_processor.batch_decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    assistant_text_response = decoded_output.split("assistant\n", 1)[1]

    return assistant_text_response


def prompt_gpt4o(prompt_text, video_frames):
    base64_frames = [
            f"data:image/jpeg;base64,{base64.b64encode(frame).decode('utf-8')}"
            for frame in video_frames
        ]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                # *[{"type": "image_url", "image_url": {"url": img}} for img in base64_frames] <- This will be inserted for new conversation only
            ],
        }
    ]
    conversation[-1]["content"].extend(
        [{"type": "image_url", "image_url": {"url": img}} for img in base64_frames]
    )

    params = {
        "model": azure_deployment_name,
        "messages": conversation,
        "max_tokens": 128,
    }
    output = openai_client.chat.completions.create(**params)

    return output.choices[0].message.content


def prompt_videollama3(prompt_text, video_frames):
    # Convert JPEG byte strings to PIL images
    images = [Image.open(io.BytesIO(frame)) for frame in video_frames]
   
    # Temporarily store the images to the ./tmp folder
    tmp_folder = "./tmp"
    tmp_image_paths = []
    for i, image in enumerate(images):
        tmp_image_path = os.path.join(tmp_folder, f"{i}.jpg")
        image.save(tmp_image_path, format="JPEG")
        tmp_image_paths.append(tmp_image_path)

    content = []
    for image_path in tmp_image_paths:
        content.append({"type": "image", "image": {"image_path": image_path}})
    content.append({"type": "text", "text": prompt_text})
    conversation = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = mllm_processor(conversation=conversation, return_tensors="pt")
    inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = mllm_model.generate(**inputs, max_new_tokens=128)
    assistant_text_response = mllm_processor.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    # Delete the temporary files
    for tmp_image_path in tmp_image_paths:
        os.remove(tmp_image_path)

    return assistant_text_response


def prompt_qwen2_5_vl(prompt_text, video_frames):
    # Convert JPEG byte strings to PIL images
    images = [Image.open(io.BytesIO(frame)) for frame in video_frames]
   
    # Temporarily store the images to the ./tmp folder
    tmp_folder = "./tmp"
    tmp_image_paths = []
    for i, image in enumerate(images):
        tmp_image_path = os.path.join(tmp_folder, f"{i}.jpg")
        image.save(tmp_image_path, format="JPEG")
        tmp_image_paths.append(tmp_image_path)

    content = []
    for image_path in tmp_image_paths:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt_text})
    conversation = [
        {
            "role": "user",
            "content": content,
        }
    ]

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
    generated_ids = mllm_model.generate(**inputs, max_new_tokens=128)
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
    for tmp_image_path in tmp_image_paths:
        os.remove(tmp_image_path)

    return assistant_text_response




def create_prompt_text(annotation_sample):
    question = annotation_sample["q_body"]
    options_str = ""
    for i in [0,1,2,3]:
        opt = annotation_sample[f"option{i}"]
        if opt != "":
            options_str += f"Option {i}: {opt}\n"
    if options_str.endswith("\n"):
        options_str = options_str[:-1]

    query_text = f"""Please answer the following question about the consecutive video frames: '{question}'.
You must choose exactly one answer from the following options:\n{options_str}
Only answer either '0', '1', '2', or '3' (if there is an available option corresponding to the number).
Your answer will be parsed so that the first occurence of a number ('0', '1', '2', '3') in your answer will be considered as the final answer.
    """
    return query_text



entries = []
annotations = data.read_and_parse_annotations()
total_correct = 0
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

for index, sample in enumerate(annotations):    
    frames = data.sample_uniform_frames_from_video(f"./dataset/SUTD-TrafficQA/compressed_videos/{sample['vid_filename']}")
    #frames = data.sample_uniform_frames_from_video(f"./dataset/SUTD-TrafficQA/compressed_videos/b_1b4411R7AX_clip_043.mp4") # Longest video for VRAM test
    prompt_text = create_prompt_text(sample)

    if ACTIVE_MLLM == 0:
        raw_response = prompt_llava_onevision(prompt_text, frames)
    elif ACTIVE_MLLM == 1:
        try:
            raw_response = prompt_gpt4o(prompt_text, frames)
        except BadRequestError as e:
            print(f"Skipping question {index + 1} due to OpenAI policy rules: {e}")
            continue
    elif ACTIVE_MLLM == 2:
        raw_response = prompt_videollama3(prompt_text, frames)
    else: # ACTIVE_MLLM == 3
        raw_response = prompt_qwen2_5_vl(prompt_text, frames)

    mllm_choice = 9
    for char in raw_response.lower():
        if char in ["0", "1", "2", "3"]:
            mllm_choice = int(char)
            break

    available_choices = []
    for i in range(4):
        if sample[f"option{i}"] != "":
            available_choices.append(i)
    entry = {
        "question_number": index + 1,
        "available_choices": available_choices,
        "correct_choice": int(sample["answer"]),
        "selected_choice": mllm_choice,
        "raw_answer": raw_response,
    }
    entries.append(entry)

    print(f"Question {index + 1}: {sample["q_body"]}")
    print(f"Available choices: {available_choices}")
    print(f"Correct choice: {sample["answer"]}")
    print(f"Raw MLLM answer: {mllm_choice}\n")

    if mllm_choice == int(sample["answer"]):
        total_correct += 1

    # Dump results to JSON file every 50 samples as checkpoint
    if (index + 1) % 50 == 0 and (index + 1) < len(annotations):
        output_filename = f"sutd_traffic_qa_{MLLM_NAMES[ACTIVE_MLLM]}_results.json"
        results_file = os.path.join(results_dir, output_filename)
        with open(results_file, "w") as f:
            json.dump(entries, f, indent=4)
        print(f"Checkpoint of benchmark results saved to {results_file}\n")


print(f"Total correct answers: {total_correct}/{len(annotations)}")

# Save results to JSON file
output_filename = f"sutd_traffic_qa_{MLLM_NAMES[ACTIVE_MLLM]}_results.json"
results_file = os.path.join(results_dir, output_filename)
with open(results_file, "w") as f:
    json.dump(entries, f, indent=4)
print(f"Benchmark results saved to {results_file}")

    


