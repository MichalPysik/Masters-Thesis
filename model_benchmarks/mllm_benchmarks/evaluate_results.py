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
    # weighted accuracy metric (1, 1.5 or 2 points based on the number of available choices)
    total_weight = 0.0
    total_weighted_correct = 0.0
    # 4-choice questions only
    total_4choice_questions = 0
    total_4choice_correct = 0
    # count how many times did the model incorrectly format the answer
    incorrectly_formatted = 0
    unavailable_choices_chosen = 0

    for question in benchmark_output:
        total_questions += 1
        question_weight = float(len(question["available_choices"]))/2.0
        total_weight += question_weight
        total_4choice_questions += 1 if len(question["available_choices"]) == 4 else 0

        if question["selected_choice"] == question["correct_choice"]:
            total_correct += 1
            total_weighted_correct += question_weight
            total_4choice_correct += 1 if len(question["available_choices"]) == 4 else 0

        if len(question["raw_answer"]) != 1 or question["raw_answer"] not in ["0", "1", "2", "3"]:
            incorrectly_formatted += 1
            print(f"Incorrectly formatted answer from model {model_name}: {question['raw_answer']}")
        elif int(question["raw_answer"]) not in question["available_choices"]:
            unavailable_choices_chosen += 1

    model_evaluation["total_questions"] = total_questions
    model_evaluation["total_correct"] = total_correct
    model_evaluation["accuracy"] = float(total_correct) / float(total_questions)
    model_evaluation["weighted_accuracy"] = total_weighted_correct / total_weight
    model_evaluation["4choice_questions_accuracy"] = float(total_4choice_correct) / float(total_4choice_questions)
    model_evaluation["incorrectly_formatted"] = incorrectly_formatted
    model_evaluation["unavailable_choices_chosen"] = unavailable_choices_chosen

    # add the model evaluation to the evaluated results
    evaluated_results[model_name] = model_evaluation

# save the evaluated results to a json file
with open("results/evaluated_results.json", "w") as f:
    json.dump(evaluated_results, f, indent=4)
print("Evaluated results saved to results/evaluated_results.json")
    


