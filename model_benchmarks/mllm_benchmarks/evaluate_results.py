import json

evaluated_results = {}

for model_name in ["LLaVA-OneVision", "GPT-4o", "VideoLLaMA-3", "Qwen2.5-VL"]:
    benchmark_output_file = f"results/sutd_traffic_qa_{model_name}_results.json"

    # check if exists, if not continue
    try:
        with open(benchmark_output_file, "r") as f:
            benchmark_output = json.load(f)
    except FileNotFoundError:
        print(f"Results for {model_name} not found, skipping...")
        continue

    model_evaluation = {}
    total_questions = 0
    total_correct = 0
    # count how many times did the model incorrectly format the answer
    incorrectly_formatted = 0

    for question in benchmark_output:
        total_questions += 1
        if question["selected_choice"] == question["correct_choice"]:
            total_correct += 1

        if len(question["raw_answer"]) != 1 and question["raw_answer"] not in ["0", "1", "2", "3"]:
            incorrectly_formatted += 1

    model_evaluation["total_questions"] = total_questions
    model_evaluation["total_correct"] = total_correct
    model_evaluation["accuracy"] = float(total_correct) / float(total_questions)
    model_evaluation["incorrectly_formatted"] = incorrectly_formatted

    # add the model evaluation to the evaluated results
    evaluated_results[model_name] = model_evaluation

# save the evaluated results to a json file
with open("results/evaluated_results.json", "w") as f:
    json.dump(evaluated_results, f, indent=4)
print("Evaluated results saved to results/evaluated_results.json")
    


