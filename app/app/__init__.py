import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from lavis.models import load_model_and_preprocess
from pymilvus import connections, Collection, utility, FieldSchema, DataType, CollectionSchema
import os
import dotenv


# Load environment variables
dotenv.load_dotenv()

# torch device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding models
embedding_model = os.getenv("EMBEDDING_MODEL")
emb_model, emb_processor = None, None
print(f"Loading {embedding_model} model...")
if embedding_model == "CLIP":
    clip_versions = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
    emb_model = CLIPModel.from_pretrained(clip_versions[1]).eval().to(device)
    emb_processor = CLIPProcessor.from_pretrained(clip_versions[1])
elif embedding_model == "SigLIP":
    siglip_version = "google/siglip-so400m-patch14-384" # most downloaded version on hugging face transformers
    emb_model = AutoModel.from_pretrained(siglip_version).eval().to(device)
    emb_processor = AutoProcessor.from_pretrained(siglip_version)
elif embedding_model == "ALIGN":
    align_version = "kakaobrain/align-base"
    emb_model = AutoModel.from_pretrained(align_version).eval().to(device)
    emb_processor = AutoProcessor.from_pretrained(align_version)
elif embedding_model == "Blip":
    emb_processor = [None, None] # Separate image ([0]) and text ([1]) processors
    emb_model, emb_processor[0], emb_processor[1] = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
else:
    raise ValueError(f"Invalid embedding model: {embedding_model}")

    

# Database
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")
connections.connect(host="localhost", port="19530")
COLLECTION_NAME = "video_frames_collection"


def create_collection(name):
    print(f"Creating a new collection {name}.")

    # Define the schema for the collection (fields: id, video_name, timestamp, embedding)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)  # Primary key
    video_name_field = FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=255)
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.FLOAT)
    if embedding_model == "CLIP":
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    elif embedding_model == "SigLIP":
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1152)
    elif embedding_model == "ALIGN":
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=640)
    elif embedding_model == "Blip":
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=256)
    else:
        raise ValueError(f"Invalid embedding model: {embedding_model}")

    schema = CollectionSchema(
        fields=[id_field, video_name_field, timestamp_field, embedding_field],
        description="Video frames with multimodal embeddings",
    )
    collection = Collection(name=name, schema=schema)

    # Create an index for the embedding field
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    # Create the index for the embedding field
    collection.create_index(field_name="embedding", index_params=index_params)

    return collection


if utility.has_collection(COLLECTION_NAME):
    print("Collection already exists, deleting...")
    collection = Collection(name=COLLECTION_NAME)
    collection.drop()

collection = create_collection(COLLECTION_NAME)
    