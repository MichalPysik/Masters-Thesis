import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from pymilvus import connections, Collection, utility, FieldSchema, DataType, CollectionSchema
import os
import pandas as pd
from PIL import Image
import json


# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load multimodal embedding model
clip_version = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(clip_version).eval().to(device)
processor = CLIPProcessor.from_pretrained(clip_version)

# Connect to Milvus
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")
connections.connect(host="localhost", port="19530")
COLLECTION_NAME = "benchmark_roadsigns_collection"


CODES_TO_NAMES = {
    "A10": "Traffic semaphore warning traffic sign",
    "A11": "Pedestrian crossing warning traffic sign",
    "A12a": "Pedestrians warning traffic sign",
    "A12b": "Children warning traffic sign",
    "A13": "Farm animals (cow) warning traffic sign",
    "A14": "Wild animals (deer) warning traffic sign",
    "A15": "Roadworks warning traffic sign",
    "A18": "Falling rocks warning traffic sign",
    "A19": "Cyclists warning traffic sign",
    "A1a": "Sharp curve to the right warning traffic sign",
    "A1b": "Sharp curve to the left warning traffic sign",
    "A22": "Other danger warning traffic sign",
    "A24": "Slippery ice warning traffic sign",
    "A28": "Soft verge warning traffic sign",
    "A29": "Level crossing with gates or barriers warning traffic sign",
    "A2a": "Double curve, first to the right warning traffic sign",
    "A2b": "Double curve, first to the left warning traffic sign",
    "A30": "Level crossing without gates or barriers warning traffic sign",
    "A31a": "Distance marker (240 m) warning traffic sign",
    "A31b": "Distance marker (160 m) warning traffic sign",
    "A31c": "Distance marker (80 m) warning traffic sign",
    "A32a": "Warning cross for single-track railway crossing warning traffic sign",
    "A32b": "Warning cross, multiple-track railway crossing warning traffic sign",
    "A5a": "Steep descent warning traffic sign",
    "A5b": "Steep ascent warning traffic sign",
    "A6a": "Road narrows from both sides warning traffic sign",
    "A6b": "Road narrows from the right side warning traffic sign",
    "A7a": "Uneven road warning traffic sign",
    "A7b": "Speed bumps warning traffic sign",
    "A8": "Slippery road warning traffic sign",
    "A9": "Two-way traffic warning traffic sign",

    "B1": "No entry for vehicles (both directions) prohibitory traffic sign",
    "B11": "No motor vehicles prohibitory traffic sign",
    "B12": "No entry to the vehicles indicated prohibitory traffic sign",
    "B13": "No vehicles heavier than indicated prohibitory traffic sign",
    "B14": "No vehicles with axle weight greater than indicated prohibitory traffic sign",
    "B15": "No vehicles wider than indicated prohibitory traffic sign",
    "B16": "No vehicles taller than indicated prohibitory traffic sign",
    "B17": "No vehicles or combinations longer than indicated prohibitory traffic sign",
    "B2": "No entry for vehicles prohibitory traffic sign",
    "B20a": "Speed limit prohibitory traffic sign",
    "B20b": "End of speed limit prohibitory traffic sign",
    "B21a": "No overtaking prohibitory traffic sign",
    "B21b": "End of no-overtaking zone prohibitory traffic sign",
    "B23a": "No use of audible warning signals prohibitory traffic sign",
    "B24a": "No right turn prohibitory traffic sign",
    "B24b": "No left turn prohibitory traffic sign",
    "B26": "End of all prohibitions prohibitory traffic sign",
    "B28": "No stopping and no parking prohibitory traffic sign",
    "B29": "No parking prohibitory traffic sign",
    "B30": "No pedestrians prohibitory traffic sign",
    "B32": "Other prohibition (Driving through forbidden) prohibitory traffic sign",
    "B34": "Minimum distance between vehicles prohibitory traffic sign",
    "B3b": "No cars prohibitory traffic sign",
    "B4": "No lorries prohibitory traffic sign",
    "B5": "No buses prohibitory traffic sign",

    "C1": "Roundabout mandatory traffic sign",
    "C2a": "Proceed straight mandatory traffic sign",
    "C3a": "Turn right here mandatory traffic sign",
    "C3b": "Turn left here mandatory traffic sign",
    "C4a": "Pass on right mandatory traffic sign",
    "C4c": "Pass on left or right mandatory traffic sign",
    "C7a": "Pedestrian path mandatory traffic sign",
    "C8a": "Cycleway mandatory traffic sign",
    "C9a": "Shared pedestrian and cycle path mandatory traffic sign",

    "E1": "Number supplementary plate",
    "E13": "Text or symbol supplementary plate",
    "E14": "Transit supplementary plate",
    "E2b": "Intersection shape supplementary plate",
    "E2d": "The shape of two intersections supplementary plate",
    "E3a": "Distance supplementary plate",
    "E3b": "Distance supplementary plate",
    "E4": "Length of section supplementary plate",
    "E5": "Maximum permissible weight supplementary plate",
    "E6": "In the wet (in the rain) supplementary plate",
    "E7a": "Directional arrow for straight direction supplementary plate",
    "E7b": "Directional arrow for turning supplementary plate",
    "E8a": "Start of section supplementary plate",
    "E8b": "Course of the section supplementary plate",
    "E8c": "End of section supplementary plate",
    "E8d": "Validity period supplementary plate",
    "E8e": "Validity period supplementary plate",
    "E9": "Vehicle type supplementary plate",

    "IJ14c": "Campsite for tents and caravans informative traffic sign",
    "IJ2": "Hospital informative traffic sign",
    "IJ4b": "Stop marker informative traffic sign",
    "IJ4c": "Bus stop informative traffic sign",
    "IJ5": "Information informative traffic sign",
    "IJ7": "Gas station informative traffic sign",
    "IJ9": "Technical inspection station informative traffic sign",

    "IP10a": "Dead end road information sign for traffic",
    "IP10b": "Advance about a dead end road information sign for traffic",
    "IP11a": "Parking information sign for traffic",
    "IP11b": "Perpendicular or inclined parking information sign for traffic",
    "IP11c": "Longitudinal parking lot information sign for traffic",
    "IP11g": "Parking lot partial parking on the sidewalk longitudinal information sign for traffic",
    "IP12": "Reserved parking information sign for traffic",
    "IP13a": "Covered parking information sign for traffic",
    "IP13b": "Parking lot with parking disc information sign for traffic",
    "IP13c": "Parking lot with parking meter information sign for traffic",
    "IP19": "Shifting lanes information sign for traffic",
    "IP2": "Slowdown threshold information sign for traffic",
    "IP22": "Change in transport organization information sign for traffic",
    "IP30": "State border information sign for traffic",
    "IP4b": "One-way traffic information sign for traffic",
    "IP5": "Recommended speed information sign for traffic",
    "IP6": "Pedestrian crossing information sign for traffic",

    "IS11b": "Directional sign for detour informative directional traffic sign",
    "IS14": "Borders of the territorial unit informative directional traffic sign",
    "IS15a": "Other name informative directional traffic sign",
    "IS16b": "Road number informative directional traffic sign",
    "IS19a": "Directional sign for cyclists informative directional traffic sign",
    "IS19b": "Directional sign for cyclists informative directional traffic sign",
    "IS19c": "Directional sign for cyclists informative directional traffic sign",
    "IS20": "Signal for cyclists informative directional traffic sign",
    "IS21a": "Directional sign for cyclists straight informative directional traffic sign",
    "IS21b": "Directional sign for cyclists left informative directional traffic sign",
    "IS21c": "Directional sign for cyclists right informative directional traffic sign",
    "IS21d": "End of the cycling route informative directional traffic sign",
    "IS22a": "Designation of the name of a street or other public space informative directional traffic sign",
    "IS22b": "Designation of the name of a street or other public space informative directional traffic sign",
    "IS22c": "Designation of the name of a street or other public space informative directional traffic sign",
    "IS22d": "Designation of the name of a street or other public space informative directional traffic sign",
    "IS22e": "Designation of the name of a street or other public space informative directional traffic sign",
    "IS22f": "Designation of the name of a street or other public space informative directional traffic sign",
    "IS24a": "Cultural or tourist destination informative directional traffic sign",
    "IS24b": "Signpost for a cultural or tourist destination informative directional traffic sign",
    "IS24c": "Communal goal informative directional traffic sign",
    "IS3a": "Direction sign with destination straight informative directional traffic sign",
    "IS3b": "Direction sign with destination left informative directional traffic sign",
    "IS3c": "Direction sign with destination right informative directional traffic sign",
    "IS4c": "Direction sign with local destination right informative directional traffic sign",
    "IS5": "Signpost with a different destination informative directional traffic sign",
    "IS9a": "Signal before a level crossing informative directional traffic sign",
    "IS9c": "Traffic light before a restricted intersection informative directional traffic sign",

    "IZ4a": "Town informative zone traffic sign",
    "IZ4b": "End of town informative zone traffic sign",
    "IZ5a": "Residential zone informative zone traffic sign",
    "IZ5b": "End of residential zone informative zone traffic sign",
    "IZ8a": "Traffic restriction zone informative zone traffic sign",
    "IZ8b": "End of traffic restriction zone informative zone traffic sign",

    "MOST": "Bridge sign",

    "P1": "Intersection with a secondary road traffic sign regulating priority",
    "P2": "Main road traffic sign regulating priority",
    "P4": "Give way! traffic sign regulating priority",
    "P6": "Stop, give way! traffic sign regulating priority",
    "P7": "Priority to oncoming vehicles traffic sign regulating priority",
    "P8": "Priority over oncoming vehicles traffic sign regulating priority",

    "RADAR": "Traffic radar",

    "Z3": "Guidance board transportation equipment",
    "Z4a": "Directional plate with diagonal stripes with a left slope transportation equipment",

    "ZRCADLO": "Traffic mirror",
}

NAMES_TO_CODES = {v: k for k, v in CODES_TO_NAMES.items()}


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
    #collection = create_collection(COLLECTION_NAME)
else:
    collection = create_collection(COLLECTION_NAME)


def insert_data(collection):
    dataset_dir = "../../../dp-demos/datasets/embedding/znacky"
    geojson_file = dataset_dir + "/ORP_Trutnov.geojson"
    with open(geojson_file, "r", encoding="utf8") as f:
        data = json.load(f)
        data_features = data["features"]

    for sample in data_features:
        filename = sample["properties"]["FOTO"]

        # Extract the class names
        class_names = ""
        for i in range(1, 11):
            tab = sample["properties"][f"tab{i}"]
            if tab != "":
                class_names += tab + ";"
        
        # Open PIL image
        img = Image.open(f"{dataset_dir}/sdz_II/{filename}")
        inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalize and flatten the embedding
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().flatten()
    
        data = [[filename], [class_names], [embedding]]
        collection.insert(data)
        print(f"Inserted {filename} into Milvus.")

        
# check if collection is empty
if collection.num_entities == 0:
    insert_data(collection)
else:
    print(f"Collection {COLLECTION_NAME} has {collection.num_entities} entities")

# Search for each class name
total_correct = 0
for query_name in CODES_TO_NAMES.values():
    inputs = processor(text=query_name, return_tensors="pt", padding=True).to(device)
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

    num_correct = 0
    for result in search_results[0]:
        true_class_codes = result.entity.get("class").split(";")[:-1]
        query_code = NAMES_TO_CODES[query_name]
        if query_code in true_class_codes:
            num_correct += 1

    print(f"Query: {query_name}, Correct predictions: {num_correct}/5")
    total_correct += num_correct

print(f"Total correct predictions: {total_correct}/{len(CODES_TO_NAMES) * 5}")













