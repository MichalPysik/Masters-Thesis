import torch
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema
import pandas as pd
from PIL import Image
import json

# Converts traffic sign codes to general class names
def convert_code_to_general_class(code):
    """
    Convert the code to a general class name.
    """
    if code.startswith("A"):
        return "Warning traffic sign"
    elif code.startswith("B"):
        return "Prohibitory traffic sign"
    elif code.startswith("C"):
        return "Mandatory traffic sign"
    elif code.startswith("E"):
        return "Supplementary plate traffic sign"
    elif code.startswith("IJ"):
        return "Other informative traffic sign"
    elif code.startswith("IP"):
        return "Operational information sign for traffic"
    elif code.startswith("IS"):
        return "Directional informative traffic sign"
    elif code.startswith("IZ"):
        return "Informative zone traffic sign"
    elif code.startswith("P"):
        return "Traffic sign regulating priority"
    elif code in ["MOST", "RADAR", "Z3", "Z4a", "ZRCADLO"]:
        return "Transportation equipment"
    else:
        raise ValueError(f"Unknown code: {code}")

# Maps traffic sign codes to specific class names
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

    "E1": "Number supplementary plate traffic sign",
    "E13": "Text or symbol supplementary plate traffic sign",
    "E14": "Transit supplementary plate traffic sign",
    "E2b": "Intersection shape supplementary plate traffic sign",
    "E2d": "The shape of two intersections supplementary plate traffic sign",
    "E3a": "Distance supplementary plate traffic sign",
    "E3b": "Distance supplementary plate traffic sign",
    "E4": "Length of section supplementary plate traffic sign",
    "E5": "Maximum permissible weight supplementary plate traffic sign",
    "E6": "In the wet (in the rain) supplementary plate traffic sign",
    "E7a": "Directional arrow for straight direction supplementary plate traffic sign",
    "E7b": "Directional arrow for turning supplementary plate traffic sign",
    "E8a": "Start of section supplementary plate traffic sign",
    "E8b": "Course of the section supplementary plate traffic sign",
    "E8c": "End of section supplementary plate traffic sign",
    "E8d": "Validity period supplementary plate traffic sign",
    "E8e": "Validity period supplementary plate traffic sign",
    "E9": "Vehicle type supplementary plate traffic sign",

    "IJ14c": "Campsite for tents and caravans informative traffic sign",
    "IJ2": "Hospital informative traffic sign",
    "IJ4b": "Stop marker informative traffic sign",
    "IJ4c": "Bus stop informative traffic sign",
    "IJ5": "Information informative traffic sign",
    "IJ7": "Gas station informative traffic sign",
    "IJ9": "Technical inspection station informative traffic sign",

    "IP10a": "Dead end road operational information sign for traffic",
    "IP10b": "Advance about a dead end road operational information sign for traffic",
    "IP11a": "Parking operational information sign for traffic",
    "IP11b": "Perpendicular or inclined parking operational information sign for traffic",
    "IP11c": "Longitudinal parking lot operational information sign for traffic",
    "IP11g": "Parking lot partial parking on the sidewalk longitudinal operational information sign for traffic",
    "IP12": "Reserved parking operational information sign for traffic",
    "IP13a": "Covered parking operational information sign for traffic",
    "IP13b": "Parking lot with parking disc operational information sign for traffic",
    "IP13c": "Parking lot with parking meter operational information sign for traffic",
    "IP19": "Shifting lanes operational information sign for traffic",
    "IP2": "Slowdown threshold operational information sign for traffic",
    "IP22": "Change in transport organization operational information sign for traffic",
    "IP30": "State border operational information sign for traffic",
    "IP4b": "One-way traffic operational information sign for traffic",
    "IP5": "Recommended speed operational information sign for traffic",
    "IP6": "Pedestrian crossing operational information sign for traffic",

    "IS11b": "Directional sign for detour directional informative traffic sign",
    "IS14": "Borders of the territorial unit directional informative traffic sign",
    "IS15a": "Other name directional informative traffic sign",
    "IS16b": "Road number directional informative traffic sign",
    "IS19a": "Directional sign for cyclists directional informative traffic sign",
    "IS19b": "Directional sign for cyclists directional informative traffic sign",
    "IS19c": "Directional sign for cyclists directional informative traffic sign",
    "IS20": "Signal for cyclists directional informative traffic sign",
    "IS21a": "Directional sign for cyclists straight directional informative traffic sign",
    "IS21b": "Directional sign for cyclists left directional informative traffic sign",
    "IS21c": "Directional sign for cyclists right directional informative traffic sign",
    "IS21d": "End of the cycling route directional informative traffic sign",
    "IS22a": "Designation of the name of a street or other public space directional informative traffic sign",
    "IS22b": "Designation of the name of a street or other public space directional informative traffic sign",
    "IS22c": "Designation of the name of a street or other public space directional informative traffic sign",
    "IS22d": "Designation of the name of a street or other public space directional informative traffic sign",
    "IS22e": "Designation of the name of a street or other public space directional informative traffic sign",
    "IS22f": "Designation of the name of a street or other public space directional informative traffic sign",
    "IS24a": "Cultural or tourist destination directional informative traffic sign",
    "IS24b": "Signpost for a cultural or tourist destination directional informative traffic sign",
    "IS24c": "Communal goal directional informative traffic sign",
    "IS3a": "Direction sign with destination straight directional informative traffic sign",
    "IS3b": "Direction sign with destination left directional informative traffic sign",
    "IS3c": "Direction sign with destination right directional informative traffic sign",
    "IS4c": "Direction sign with local destination right directional informative traffic sign",
    "IS5": "Signpost with a different destination directional informative traffic sign",
    "IS9a": "Signal before a level crossing directional informative traffic sign",
    "IS9c": "Traffic light before a restricted intersection directional informative traffic sign",

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

# Maps the same in reverse
NAMES_TO_CODES = {v: k for k, v in CODES_TO_NAMES.items()}


def create_collection(name, embedding_dim):
    print(f"Creating a new collection {name}.")

    # Define the schema for the collection (fields: id, video_name, timestamp, embedding)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)  # Primary key
    filename_field = FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255)
    class_field = FieldSchema(name="class", dtype=DataType.VARCHAR, max_length=255)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)

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


def check_correct_entity_count(collection, expected_count):
    """
    Check if the number of entities in the collection matches the expected count, via a query.
    """
    collection.load()
    result = collection.query(expr="", output_fields=["count(*)"])
    count = result[0]["count(*)"]
    
    return count == expected_count


def insert_cars196_data(collection, model, processor, device, blip=False):
    dataset_dir = "./datasets/CARS196"
    df = pd.read_excel(f"{dataset_dir}/stanford_cars_with_class_names.xlsx")
    
    for sample in df.iterrows():
        filename = sample[1]["image"]
        #class_num = sample[1]["class"]
        class_name = sample[1]["ture_class_name"]

        # open PIL image
        img = Image.open(f"{dataset_dir}/cars_train_images/{filename}")
        # Grayscale images need to be converted to RGB for certain models to work
        if img.mode != "RGB":
            print(f"Converting {filename} from {img.mode} to RGB.")
            img = img.convert("RGB")

        if blip:
            inputs = processor[0]["eval"](img).unsqueeze(0).to(device)
            sample = {"image": inputs, "text_input": None}
            image_features = model.extract_features(sample, mode="image")
            # project from 768 to 256 dimensions (includes normalization)
            image_features = image_features.image_embeds_proj[:, 0, :]
        else:
            inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        embedding = image_features.cpu().numpy().flatten()

        data = [[filename], [class_name], [embedding]]
        collection.insert(data)
        print(f"Inserted {filename} into Milvus.")


def insert_czech_traffic_signs_data(collection, model, processor, device, blip=False):
    dataset_dir = "./datasets/czech_traffic_signs"
    geojson_file = dataset_dir + "/annotations.geojson"
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
        img = Image.open(f"{dataset_dir}/images/{filename}")

        if blip:
            inputs = processor[0]["eval"](img).unsqueeze(0).to(device)
            sample = {"image": inputs, "text_input": None}
            image_features = model.extract_features(sample, mode="image")
            # project from 768 to 256 dimensions (includes normalization)
            image_features = image_features.image_embeds_proj[:, 0, :]
        else:
            inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().flatten()
    
        data = [[filename], [class_names], [embedding]]
        collection.insert(data)
        print(f"Inserted {filename} into Milvus.")

# Prints important statistics about the CARS196 dataset (only the used training part)
def describe_cars196_dataset():
    dataset_dir = "./datasets/CARS196"
    df = pd.read_excel(f"{dataset_dir}/stanford_cars_with_class_names.xlsx")
    print("----------------CARS196----------------")
    print(f"Number of images in CARS196: {len(df)}\n")

    # Stats for specific car models
    print(f"Number of unique car models in CARS196: {len(df['ture_class_name'].unique())}")
    class_counts = df["ture_class_name"].value_counts()
    most_common_class = class_counts.idxmax()
    most_common_count = class_counts.max()
    least_common_class = class_counts.idxmin()
    least_common_count = class_counts.min()
    print(f"Most common car model: {most_common_class} ({most_common_count} images)")
    print(f"Least common car model: {least_common_class} ({least_common_count} images)")
    avg_per_class = len(df) / len(df["ture_class_name"].unique())
    print(f"Average number of images per unique car model: {avg_per_class:.2f}\n")

    # Group the car models by car brands (first substring without whitespace) - stats
    df["brand"] = df["ture_class_name"].apply(lambda x: x.split(" ")[0])
    # "AM" and "HUMMER" should be the same brand in this case
    df.loc[df["brand"] == "AM", "brand"] = "HUMMER"
    print(f"Number of unique car brands in CARS196: {len(df['brand'].unique())} (when AM and HUMMER are merged)")
    brand_counts = df["brand"].value_counts()
    most_common_brand = brand_counts.idxmax()
    most_common_count = brand_counts.max()
    least_common_brand = brand_counts.idxmin()
    least_common_count = brand_counts.min()
    print(f"Most common car brand: {most_common_brand} ({most_common_count} images)")
    print(f"Least common car brand: {least_common_brand} ({least_common_count} images)")
    avg_per_brand = len(df) / len(df["brand"].unique())
    print(f"Average number of images per unique car brand: {avg_per_brand:.2f}\n")
    

def describe_czech_traffic_signs_dataset():
    dataset_dir = "./datasets/czech_traffic_signs"
    geojson_file = dataset_dir + "/annotations.geojson"
    with open(geojson_file, "r", encoding="utf8") as f:
        data = json.load(f)
        samples = data["features"]
    print("----------------Czech Traffic Signs----------------")
    print(f"Number of annotated images in Czech Traffic Signs: {len(samples)}\n")

    # Stats for specific traffic signs
    avg_unique_clases_per_sample = 0
    for sample in samples:
        class_names = []
        for i in range(1, 11):
            tab = sample["properties"][f"tab{i}"]
            if tab != "":
                class_names.append(tab)
        sample["unique_class_names"] = list(set(class_names))
        avg_unique_clases_per_sample += len(sample["unique_class_names"])
    avg_unique_clases_per_sample /= len(samples)
    print(f"Number of unique traffic signs in Czech Traffic Signs: {len(CODES_TO_NAMES)}")
    print(f"Average number of unique traffic signs per image: {avg_unique_clases_per_sample:.3f}")
    # Most commonly and least occuring traffic sign
    class_counts = {}
    for sample in samples:
        for class_name in sample["unique_class_names"]:
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
    most_common_class = max(class_counts, key=class_counts.get)
    most_common_count = class_counts[most_common_class]
    least_common_class = min(class_counts, key=class_counts.get)
    least_common_count = class_counts[least_common_class]
    least_common_classes = [k for k, v in class_counts.items() if v == least_common_count]
    print(f"Most common traffic sign: {most_common_class} = {CODES_TO_NAMES[most_common_class]} ({most_common_count} images)")
    print(f"Least common traffic signs: {least_common_classes} ({least_common_count} image each)")
    avg_image_occurences_per_class = sum(class_counts.values()) / len(class_counts)
    print("Each unique traffic sign appears on average in", avg_image_occurences_per_class, "images.\n")

    # Group the traffic signs by their general class - stats
    print("Number of unique traffic sign categories in Czech Traffic Signs: 10")
    avg_unique_categories_per_sample = 0
    for sample in samples:
        sample["unique_class_categories"] = []
        for class_name in sample["unique_class_names"]:
            general_class = convert_code_to_general_class(class_name)
            if general_class not in sample["unique_class_categories"]:
                sample["unique_class_categories"].append(general_class)
        avg_unique_categories_per_sample += len(sample["unique_class_categories"])
    avg_unique_categories_per_sample /= len(samples)
    print(f"Average number of unique traffic sign categories per image: {avg_unique_categories_per_sample:.3f}")
    # Most commonly and least occuring traffic sign category
    category_counts = {}
    for sample in samples:
        for class_name in sample["unique_class_categories"]:
            if class_name not in category_counts:
                category_counts[class_name] = 0
            category_counts[class_name] += 1
    most_common_category = max(category_counts, key=category_counts.get)
    most_common_count = category_counts[most_common_category]
    least_common_category = min(category_counts, key=category_counts.get)
    least_common_count = category_counts[least_common_category]
    print(f"Most common traffic sign category: {most_common_category} ({most_common_count} images)")
    print(f"Least common traffic sign category: {least_common_category} ({least_common_count} images)")
    avg_image_occurences_per_category = sum(category_counts.values()) / len(category_counts)
    print("Each unique traffic sign category appears on average in", avg_image_occurences_per_category, "images.\n")


# Describe both datasets if the script is executed directly
if __name__ == "__main__":
    describe_cars196_dataset()
    describe_czech_traffic_signs_dataset()