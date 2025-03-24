import os
import dotenv
import logging
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
)
# When BLIP is configured, older version of transformers is used
try:
    from transformers import BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
except ImportError:
    pass


# Load environment variables
dotenv.load_dotenv()


# Setup logging
log_level_str = os.getenv("LOGGING_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper())
if not isinstance(log_level, int):
    raise ValueError(f"Invalid log level: {log_level_str}")
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


# Configure torch device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load configured embedding model
emb_model_name = os.getenv("EMBEDDING_MODEL")
emb_model, emb_processor = None, None
logging.info(f"Loading {emb_model_name} embedding model...")

if emb_model_name == "CLIP":
    clip_versions = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
    emb_model = CLIPModel.from_pretrained(clip_versions[1]).eval().to(device)
    emb_processor = CLIPProcessor.from_pretrained(clip_versions[1])

elif emb_model_name == "SigLIP":
    siglip_version = "google/siglip-so400m-patch14-384"  # most downloaded version on hugging face transformers
    emb_model = AutoModel.from_pretrained(siglip_version).eval().to(device)
    emb_processor = AutoProcessor.from_pretrained(siglip_version)

elif emb_model_name == "ALIGN":
    align_version = "kakaobrain/align-base"
    emb_model = AutoModel.from_pretrained(align_version).eval().to(device)
    emb_processor = AutoProcessor.from_pretrained(align_version)

elif emb_model_name == "BLIP":
    from lavis.models import load_model_and_preprocess

    emb_processor = [None, None]  # Separate image ([0]) and text ([1]) processors
    emb_model, emb_processor[0], emb_processor[1] = load_model_and_preprocess(
        name="blip_feature_extractor", model_type="base", is_eval=True, device=device
    )

elif emb_model_name.lower() != "none":
    error_message = f"Invalid embedding model: {emb_model_name}"
    logging.error(error_message)
    raise ValueError(error_message)

logging.info(f"Loaded {emb_model_name} embedding model.")


# Load configured MLLM
mllm_model_name = os.getenv("MLLM")
mllm_model, mllm_processor = None, None
logging.info(f"Loading {mllm_model_name} MLLM...")

# BLIP can only be used with GPT-4o (or None)
if mllm_model_name in ["LLaVA-OneVision", "VideoLLaMA-3", "Qwen2.5-VL"]:
    if emb_model_name == "BLIP":
        raise ValueError("BLIP is not compatible with LLaVA-OneVision, VideoLLaMA-3, or Qwen2.5-VL.")
    # Local models can use the BitsAndBytesConfig for 4-bit quantization,
    # pass 'quantization_config=quantization_config' to the model's {Model}.from_pretrained() method
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

if mllm_model_name == "LLaVA-OneVision":
    llava_onevision_version = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    mllm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        llava_onevision_version,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    mllm_processor = AutoProcessor.from_pretrained(llava_onevision_version)

elif mllm_model_name == "GPT-4o":
    pass  # GPT-4o communicates through OpenAI API

elif mllm_model_name == "VideoLLaMA-3":
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

elif mllm_model_name == "Qwen2.5-VL":
    qwen2_5_vl_version = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    mllm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen2_5_vl_version,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    mllm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

elif mllm_model_name.lower() != "none":
    error_message = f"Invalid MLLM: {mllm_model_name}"
    logging.error(error_message)
    raise ValueError(error_message)

logging.info(f"Loaded {mllm_model_name} MLLM.")
