import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, Collection, utility, FieldSchema, DataType, CollectionSchema
import os
from dotenv import load_dotenv


# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


# Database
load_dotenv()
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")
# Milvus connection
connections.connect(host="localhost", port="19530")
COLLECTION_NAME = "video_frames_collection"


def create_collection(name):
    print(f"Creating a new collection {name}.")

    # Define the schema for the collection (fields: id, video_name, timestamp, embedding)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)  # Primary key
    video_name_field = FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=255)
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.FLOAT)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)  # CLIP embedding size

    schema = CollectionSchema(
        fields=[id_field, video_name_field, timestamp_field, embedding_field],
        description="Video frames with CLIP embeddings",
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
    