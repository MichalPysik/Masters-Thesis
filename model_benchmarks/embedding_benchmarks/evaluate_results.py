import json
import pandas as pd
import data

# CARS196 - Car models
# Precision@1, Precision@5, Precision@10, Precision@24, Precision@42
# AP@1, AP@5, AP@10, AP@24, AP@42
# Recall@42, Recall@68

# CARS196 - Car brands
# Precision@1, Precision@5, Precision@10, Precision@29, Precision@68
# AP@1, AP@5, AP@10, AP@29, AP@68

# Czech traffic signs - Signs
# Precision@1, Precision@5, Precision@10, Precision@23
# AP@1, AP@5, AP@10, AP@23
# Recall@23

# Czech traffic signs - Sign categories
# Precision@1, Precision@5, Precision@10, Precision@27
# AP@1, AP@5, AP@10, AP@27


def evaluate_cars196_results():
    # Load the CARS196 dataset to get class counts for recall calculations
    dataset_dir = "./datasets/CARS196"
    df = pd.read_excel(f"{dataset_dir}/stanford_cars_with_class_names.xlsx")
    
    car_model_names = list(df["ture_class_name"].unique())
    car_model_names.sort()
    car_model_counts = {name: len(df[df["ture_class_name"] == name]) for name in car_model_names}

    final_results = {}

    for model_name in ["CLIP", "SigLIP", "ALIGN", "BLIP"]:
        # Load the results from the JSON file
        try:
            with open(f"results/cars196_{model_name}_results.json", "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"CARS196 results file for {model_name} not found, skipping...")
            continue

        # Create dictionaries to accumulate metrics for each query
        car_model_metrics = {
            "Precision@1": [],
            "Precision@5": [],
            "Precision@10": [],
            "Precision@24": [],
            "Precision@42": [],
            "AP@1": [],
            "AP@5": [],
            "AP@10": [],
            "AP@24": [],
            "AP@42": [],
            "Recall@42": [],
            "Recall@68": [],
        }
        car_brand_metrics = {
            "Precision@1": [],
            "Precision@5": [],
            "Precision@10": [],
            "Precision@29": [],
            "Precision@68": [],
            "AP@1": [],
            "AP@5": [],
            "AP@10": [],
            "AP@29": [],
            "AP@68": [],
        }

        for result in results:
            # Create array of binary prediction for car models - each prediction matching query is 1, otherwise 0
            predictions = result["predictions"]
            boolean_predictions = [1 if pred == result["query"] else 0 for pred in predictions]

            # Calculate precision for car models
            for limit in [1, 5, 10, 24, 42]:
                car_model_metrics[f"Precision@{limit}"].append(sum(boolean_predictions[:limit]) / float(limit))

            # Calculate recall for car models
            for limit in [42, 68]:
                car_model_metrics[f"Recall@{limit}"].append(sum(boolean_predictions[:limit]) / float(car_model_counts[result["query"]]))

            # Calculate average precision for car models
            ap_sum = 0
            for i in range(42):
                if boolean_predictions[i] == 1:
                    ap_sum += sum(boolean_predictions[:i + 1]) / float(i + 1)
                if (i + 1) in [1, 5, 10, 24, 42]:
                    car_model_metrics[f"AP@{i + 1}"].append(ap_sum / float(i + 1))

            # Create array of binary prediction for car brands - interpret HUMMER and AM as the same brand
            query_brand = result["query"].split(" ")[0]
            brand_predictions = [1 if pred.startswith(query_brand) or (query_brand in ["HUMMER", "AM"] and pred.startswith(("HUMMER", "AM"))) else 0 for pred in predictions]

            # Calculate precision for car brands
            for limit in [1, 5, 10, 29, 68]:
                car_brand_metrics[f"Precision@{limit}"].append(sum(brand_predictions[:limit]) / float(limit))

            # Calculate average precision for car brands
            ap_sum = 0
            for i in range(68):
                if brand_predictions[i] == 1:
                    ap_sum += sum(brand_predictions[:i + 1]) / float(i + 1)
                if (i + 1) in [1, 5, 10, 29, 68]:
                    car_brand_metrics[f"AP@{i + 1}"].append(ap_sum / float(i + 1))

        # Calculate mean values for each metric
        car_model_metrics_mean = {("m" + metric): sum(values) / len(values) for metric, values in car_model_metrics.items()}
        car_brand_metrics_mean = {("m" + metric): sum(values) / len(values) for metric, values in car_brand_metrics.items()}

        model_results = {"car_models": car_model_metrics_mean, "car_brands": car_brand_metrics_mean}
        final_results[model_name] = model_results
        print(f"CARS196 results for {model_name} processed.")

    # Save the final results to a JSON file
    with open("results/cars196_evaluation.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("CARS196 evaluation results saved to results/cars196_evaluation.json")


def evaluate_czech_traffic_signs_results():
    # Load the Czech Traffic Signs dataset to get class counts for recall calculations
    dataset_dir = "./datasets/czech_traffic_signs"
    geojson_file = dataset_dir + "/annotations.geojson"
    with open(geojson_file, "r", encoding="utf8") as f:
        geojson_data = json.load(f)
        samples = geojson_data["features"]

    # Create a dictionary to count the number of occurences for each class
    class_counts = {}
    for sample in samples:
        current_tabs = [] # count multiple occurences of the same sign in a single image as one
        for i in range(1, 11):
            tab = sample["properties"][f"tab{i}"]
            if tab != "" and tab not in current_tabs:
                current_tabs.append(tab)
                if tab not in class_counts:
                    class_counts[tab] = 0
                class_counts[tab] += 1

    final_results = {}

    for model_name in ["CLIP", "SigLIP", "ALIGN", "BLIP"]:
        # Load the results from the JSON file
        try:
            with open(f"results/czech_traffic_signs_{model_name}_results.json", "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Czech Traffic Signs results file for {model_name} not found, skipping...")
            continue

        # Create dictionaries to accumulate metrics for each query
        sign_metrics = {
            "Precision@1": [],
            "Precision@5": [],
            "Precision@10": [],
            "Precision@23": [],
            "AP@1": [],
            "AP@5": [],
            "AP@10": [],
            "AP@23": [],
            "Recall@23": [],
        }
        sign_category_metrics = {
            "Precision@1": [],
            "Precision@5": [],
            "Precision@10": [],
            "Precision@27": [],
            "AP@1": [],
            "AP@5": [],
            "AP@10": [],
            "AP@27": [],
        }

        for result in results:
            # Create array of binary prediction for signs - each prediction matching query is 1, otherwise 0
            # This time, each prediction is an array instead of a single value (one images can contiain multiple signs)
            predictions = result["predictions"]
            boolean_predictions = [1 if result["query_code"] in pred else 0 for pred in predictions]

            # Calculate precision for signs
            for limit in [1, 5, 10, 23]:
                sign_metrics[f"Precision@{limit}"].append(sum(boolean_predictions[:limit]) / float(limit))

            # Calculate recall for signs
            sign_metrics["Recall@23"].append(sum(boolean_predictions[:23]) / float(class_counts[result["query_code"]]))

            # Calculate average precision for signs
            ap_sum = 0
            for i in range(23):
                if boolean_predictions[i] == 1:
                    ap_sum += sum(boolean_predictions[:i + 1]) / float(i + 1)
                if (i + 1) in [1, 5, 10, 23]:
                    sign_metrics[f"AP@{i + 1}"].append(ap_sum / float(i + 1))

            # Create array of binary predictions for sign categories
            query_category = data.convert_code_to_general_class(result["query_code"])
            category_predictions = [1 if query_category in [data.convert_code_to_general_class(pred) for pred in pred_list] else 0 for pred_list in predictions]

            # Calculate precision for sign categories
            for limit in [1, 5, 10, 27]:
                sign_category_metrics[f"Precision@{limit}"].append(sum(category_predictions[:limit]) / float(limit))

            # Calculate average precision for sign categories
            ap_sum = 0
            for i in range(27):
                if category_predictions[i] == 1:
                    ap_sum += sum(category_predictions[:i + 1]) / float(i + 1)
                if (i + 1) in [1, 5, 10, 27]:
                    sign_category_metrics[f"AP@{i + 1}"].append(ap_sum / float(i + 1))

        # Calculate mean values for each metric
        sign_metrics_mean = {("m" + metric): sum(values) / len(values) for metric, values in sign_metrics.items()}
        sign_category_metrics_mean = {("m" + metric): sum(values) / len(values) for metric, values in sign_category_metrics.items()}

        model_results = {"signs": sign_metrics_mean, "sign_categories": sign_category_metrics_mean}
        final_results[model_name] = model_results
        print(f"Czech traffic sign results for {model_name} processed.")

    # Save the final results to a JSON file
    with open("results/czech_traffic_signs_evaluation.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("Czech Traffic Signs evaluation results saved to results/czech_traffic_signs_evaluation.json")








evaluate_cars196_results()
evaluate_czech_traffic_signs_results()