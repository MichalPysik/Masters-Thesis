import json
import numpy as np
import matplotlib.pyplot as plt

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



# Plot the accuracy
model_names = ['LLaVA-OneVision', 'GPT-4o', 'VideoLLaMA-3', 'Qwen2.5-VL']
accuracies = [evaluated_results[model]['accuracy'] for model in model_names]
colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:purple']  # Specific colors as requested

# Plotting
plt.figure(figsize=(12, 6))
plt.barh(model_names[::-1], accuracies[::-1], color=colors[::-1])
plt.xlabel('Accuracy', fontsize=16)
plt.ylabel('MLLM', fontsize=16)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/accuracy_plot.png')
    


