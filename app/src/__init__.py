import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration
import os
import dotenv


# Load environment variables
dotenv.load_dotenv()


# Configure torch device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load configured embedding model
emb_model_name = os.getenv("EMBEDDING_MODEL")
emb_model, emb_processor = None, None
print(f"Loading {emb_model_name} embedding model...")
if emb_model_name == "CLIP":
    clip_versions = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
    emb_model = CLIPModel.from_pretrained(clip_versions[1]).eval().to(device)
    emb_processor = CLIPProcessor.from_pretrained(clip_versions[1])
elif emb_model_name == "SigLIP":
    siglip_version = "google/siglip-so400m-patch14-384" # most downloaded version on hugging face transformers
    emb_model = AutoModel.from_pretrained(siglip_version).eval().to(device)
    emb_processor = AutoProcessor.from_pretrained(siglip_version)
elif emb_model_name == "ALIGN":
    align_version = "kakaobrain/align-base"
    emb_model = AutoModel.from_pretrained(align_version).eval().to(device)
    emb_processor = AutoProcessor.from_pretrained(align_version)
elif emb_model_name == "BLIP":
    from lavis.models import load_model_and_preprocess
    emb_processor = [None, None] # Separate image ([0]) and text ([1]) processors
    emb_model, emb_processor[0], emb_processor[1] = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
elif emb_model_name.lower() != "none":
    raise ValueError(f"Invalid embedding model: {emb_model_name}")


# Load configured MLLM
mllm_model_name = os.getenv("MLLM_MODEL")
mllm_model, mllm_processor = None, None
print(f"Loading {mllm_model_name} MLLM model...")
# Some models will use the BitsAndBytesConfig for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
if mllm_model_name == "LLaVA-OneVision":
    llava_onevision_version = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    mllm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(llava_onevision_version, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config)#, attn_implementation="flash_attention_2")
    mllm_processor = AutoProcessor.from_pretrained(llava_onevision_version)
elif mllm_model_name == "GPT-4o":
    pass # GPT-4o communicates through OpenAI API
elif mllm_model_name.lower() != "none":
    raise ValueError(f"Invalid MLLM model: {mllm_model_name}")

    