import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from pymilvus import connections, Collection, utility
import os
import pandas as pd
import json
import data

# 0 = CLIP, 1 = SigLIP, 2 = ALIGN, 3 = BLIP - Configure!
ACTIVE_EMB_MODEL = 0
EMBEDDING_DIMS = [768, 1152, 640, 256]
EMB_MODEL_NAMES = ["CLIP", "SigLIP", "ALIGN", "BLIP"]

# 0 = CARS196, 1 = Czech traffic signs - Configure!
ACTIVE_DATASET = 0

collection_name = None
if ACTIVE_DATASET == 0:
    collection_name = "cars196_embedding_benchmark_collection"
elif ACTIVE_DATASET == 1:
    collection_name = "czech_traffic_signs_embedding_benchmark_collection"
else:
    raise ValueError("Invalid dataset selection. Please choose 0 for CARS196 or 1 for Czech traffic signs.")


# Connect to Milvus
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")
connections.connect(host="localhost", port="19530")


# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load multimodal embedding model
model, processor = None, None
if ACTIVE_EMB_MODEL == 0:
    clip_versions = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
    model = CLIPModel.from_pretrained(clip_versions[1]).eval().to(device)
    processor = CLIPProcessor.from_pretrained(clip_versions[1])

elif ACTIVE_EMB_MODEL == 1:
    siglip_version = "google/siglip-so400m-patch14-384"  # most downloaded version on hugging face transformers
    model = AutoModel.from_pretrained(siglip_version).eval().to(device)
    processor = AutoProcessor.from_pretrained(siglip_version)

elif ACTIVE_EMB_MODEL == 2:
    align_version = "kakaobrain/align-base"
    model = AutoModel.from_pretrained(align_version).eval().to(device)
    processor = AutoProcessor.from_pretrained(align_version)

elif ACTIVE_EMB_MODEL == 3:
    from lavis.models import load_model_and_preprocess
    processor = [None, None]  # Separate image ([0]) and text ([1]) processors
    model, processor[0], processor[1] = load_model_and_preprocess(
        name="blip_feature_extractor", model_type="base", is_eval=True, device=device
    )


# Load collection or create new
if utility.has_collection(collection_name):
    print(f"Collection {collection_name} already exists")
    collection = Collection(name=collection_name)
    
    for field in collection.schema.fields:
        if field.name == "embedding":
            embedding_dim = int(field.params["dim"])
            break
    if embedding_dim != EMBEDDING_DIMS[ACTIVE_EMB_MODEL]:
        print(f"Embedding dimension mismatch: {embedding_dim} != {EMBEDDING_DIMS[ACTIVE_EMB_MODEL]}")
        collection.drop()
        collection = data.create_collection(collection_name, EMBEDDING_DIMS[ACTIVE_EMB_MODEL])
else:
    collection = data.create_collection(collection_name, EMBEDDING_DIMS[ACTIVE_EMB_MODEL])


# Fill the collection with data if needed
if not data.check_correct_entity_count(collection, 8144 if ACTIVE_DATASET == 0 else 2294):
    if ACTIVE_DATASET == 0:
        data.insert_cars196_data(collection, model, processor, device, blip=(ACTIVE_EMB_MODEL == 3))
    else: # ACTIVE_DATASET == 1:
        data.insert_czech_traffic_signs_data(collection, model, processor, device, blip=(ACTIVE_EMB_MODEL == 3))
else:
    print(f"Collection {collection_name} already has {8144 if ACTIVE_DATASET == 0 else 2294} entities")


def run_cars196_benchmark():
    # Load CARS196 dataset annotations
    dataset_dir = "./datasets/CARS196"
    df = pd.read_excel(f"{dataset_dir}/stanford_cars_with_class_names.xlsx")

    # Get list of all 196 unique class names
    class_names = list(df["ture_class_name"].unique())
    class_names.sort()

    # Results will be list of dicts, later saved as JSON
    benchmark_results = []

    # Query for each class name and collect results (top-k = 25)
    for i, class_name in enumerate(class_names):
        if ACTIVE_EMB_MODEL == 3: # BLIP
            inputs = processor[1]["eval"](class_name)
            sample = {"image": None, "text_input": [inputs]}
            text_features = model.extract_features(sample, mode="text")
            # Project from 768 to 256 dimensions (includes normalization)
            text_features = text_features.text_embeds_proj[:, 0, :]
        else:
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
            param={"metric_type": "COSINE", "top_k": 68},
            consistency_level="Strong",
            output_fields=["class"],
            limit=68,
        )

        # Store the predictions
        predictions = []
        prev_similarity = 1.0
        for result in search_results[0]:
            # Make sure the predictions are sorted by similarity in descending order
            assert result.distance <= prev_similarity
            prev_similarity = result.distance
            predictions.append(result.entity.get("class"))

        # Create dict entry for the current query
        sample_result = {
            "query": class_name,
            "predictions": predictions,
        }
        benchmark_results.append(sample_result)
        print(f"Processed {i + 1}/{len(class_names)}: {class_name}")

    # Save results to JSON file
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"cars196_{EMB_MODEL_NAMES[ACTIVE_EMB_MODEL]}_results.json")
    with open(results_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"Benchmark results saved to {results_file}")
        

def run_czech_traffic_signs_benchmark():
    # All 152 unique class names are already stored (sorted) in the dictionary
    class_names = data.CODES_TO_NAMES.values()

    # Results will be list of dicts, later saved as JSON
    benchmark_results = []

    # Query for each class name and collect results (top-k = 25)
    for i, class_name in enumerate(class_names):
        if ACTIVE_EMB_MODEL == 3: # BLIP
            inputs = processor[1]["eval"](class_name)
            sample = {"image": None, "text_input": [inputs]}
            text_features = model.extract_features(sample, mode="text")
            # Project from 768 to 256 dimensions (includes normalization)
            text_features = text_features.text_embeds_proj[:, 0, :]
        else:
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
            param={"metric_type": "COSINE", "top_k": 27},
            consistency_level="Strong",
            output_fields=["class"],
            limit=27,
        )

        # Store the predictions
        predictions = []
        prev_similarity = 1.0
        for result in search_results[0]:
            # Make sure the predictions are sorted by similarity in descending order
            assert result.distance <= prev_similarity
            prev_similarity = result.distance

            # Each prediction is a list by itself, since more signs can be present in one image
            multiclass_prediction = result.entity.get("class").split(";")[:-1]
            predictions.append(multiclass_prediction)

        # Create dict entry for the current query
        sample_result = {
            "query": class_name,
            "query_code": data.NAMES_TO_CODES[class_name],
            "predictions": predictions,
        }
        benchmark_results.append(sample_result)
        print(f"Processed {i + 1}/{len(class_names)}: {class_name}")

    # Save results to JSON file
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"czech_traffic_signs_{EMB_MODEL_NAMES[ACTIVE_EMB_MODEL]}_results.json")
    with open(results_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"Benchmark results saved to {results_file}")



# Run the benchmark
collection.load()
if ACTIVE_DATASET == 0:
    run_cars196_benchmark()
elif ACTIVE_DATASET == 1:
    run_czech_traffic_signs_benchmark()
else:
    raise ValueError("Invalid dataset selection. Please choose 0 for CARS196 or 1 for Czech traffic signs.")