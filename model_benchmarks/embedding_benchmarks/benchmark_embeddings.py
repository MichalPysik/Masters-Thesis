import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from pymilvus import connections, Collection, utility, FieldSchema, DataType, CollectionSchema
import os
import pandas as pd
from PIL import Image


# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load multimodal embedding model
clip_versions = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
model = CLIPModel.from_pretrained(clip_versions[1]).eval().to(device)
processor = CLIPProcessor.from_pretrained(clip_versions[1])

# Connect to Milvus
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")
connections.connect(host="localhost", port="19530")
COLLECTION_NAME = "embedding_benchmark_collection"


def create_collection(name):
    print(f"Creating a new collection {name}.")

    # Define the schema for the collection (fields: id, video_name, timestamp, embedding)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)  # Primary key
    filename_field = FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255)
    class_field = FieldSchema(name="class", dtype=DataType.VARCHAR, max_length=255)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)

    schema = CollectionSchema(
        fields=[id_field, filename_field, class_field, embedding_field],
        description="Images for retrieval benchmarking.",
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
    print(f"Collection {COLLECTION_NAME} already exists")
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    #collection.drop()
else:
    collection = create_collection(COLLECTION_NAME)


def insert_data(collection):
    dataset_dir = "../../../dp-demos/datasets/embedding/CARS196"
    df = pd.read_excel(f"{dataset_dir}/stanford_cars_with_class_names.xlsx")
    
    for sample in df.iterrows():
        filename = sample[1]["image"]
        class_num = sample[1]["class"]
        class_name = sample[1]["ture_class_name"]

        # open PIL image
        img = Image.open(f"{dataset_dir}/cars_train/cars_train/{filename}")
        inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize and flatten the embedding
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().flatten()

        data = [[filename], [class_name], [embedding]]
        collection.insert(data)
        print(f"Inserted {filename} into Milvus.")
        
# check if collection is empty
if collection.num_entities == 0:
    insert_data(collection)
else:
    print(f"Collection {COLLECTION_NAME} has {collection.num_entities} entities")



# Load class names
df = pd.read_excel("../../../dp-demos/datasets/embedding/CARS196/stanford_cars_with_class_names.xlsx")
class_names = df["ture_class_name"].unique()

print(len(class_names), "model classes loaded.")

# Benchmark model for all 196 classes
for class_name in class_names:
    inputs = processor(text=class_name, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # Normalize the embedding
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Convert the text embedding to 1-D numpy array
    query_embedding = text_features.cpu().numpy().flatten()

    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "top_k": 5},
        consistency_level="Strong",
        output_fields=["class"],
        limit=5,
    )

    # Calculate how many of 5 predictions were correct
    correct_model = 0
    correct_maker = 0
    for result in search_results[0]:
        predicted_class = result.entity.get("class")
        if predicted_class == class_name:
            correct_model += 1
        if predicted_class.split()[0] == class_name.split()[0]:
            correct_maker += 1

    print(f"Class: {class_name}, Correct models: {correct_model}/5, Correct makers: {correct_maker}/5")