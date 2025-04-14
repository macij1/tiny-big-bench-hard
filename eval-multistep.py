import json
import ollama
import csv
from typing import List
import re

# === CONFIG ===
json_file = 'multistep_arithmetic_two.json'  # Replace with your path
models = ['deepseek-r1:1.5b', 'deepseek-r1:7b', 'deepseek-r1:8b', 'deepseek-r1:14b']  # or any aliases you use in Ollama
few_shot_preamble = "" #
max_examples = 10  # set to an integer to limit number of examples

# === FUNCTIONS ===

def load_examples(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['examples']

def extract_answer(text: str):
    # Extracts a single numerical value from the response, including negative numbers
    try:
        # Regular expression to match both integers and floats (including negative ones)
        numbers = [float(num) for num in re.findall(r'-?\d+\.?\d*', text)]
        
        if len(numbers) == 1:
            return numbers[0]  # Return the single numerical value
        else:
            print(f"Error: Multiple or no numerical values found in response: {text}")
            return None
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def evaluate_models(models: List[str], few_shot: str, examples: List[dict]):
    results = []
    print("\n🧠 Evaluating models...")

    for ex in examples:
        row = {
            "input": ex['input'],
            "expected": ex['target'],
            "Deepseek1.5": "",
            "Deepseek7": "",
            "Deepseek8": "",
            "Deepseek14": ""
        }

        for model in models:
            print(f"\nEvaluating model: {model}")
            prompt = f"{few_shot.strip()}\n\nQ:{ex['input']}\nA:"
            print(f"Prompt: {prompt}")
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            output = response['message']['content']
            print(f"Output from {model}: {output}")

            # Store the raw output in the corresponding model's column
            if "1.5b" in model:
                row["Deepseek1.5"] = output
            elif "7b" in model:
                row["Deepseek7"] = output
            elif "8b" in model:
                row["Deepseek8"] = output
            elif "14b" in model:
                row["Deepseek14"] = output

        results.append(row)

    return results

def write_results_to_csv(results, output_file):
    headers = [
        "Multiarithmetic question", "Correct Answer:",
        "Deepseek1.5 Answer", "Deepseek7 Answer",
        "Deepseek8 Answer", "Deepseek14 Answer"
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for result in results:
            row = [
                result['input'], result['expected'],
                result['Deepseek1.5'], result['Deepseek7'],
                result['Deepseek8'], result['Deepseek14']
            ]
            writer.writerow(row)

# === MAIN ===

if __name__ == "__main__":
    all_examples = load_examples(json_file)
    if max_examples:
        all_examples = all_examples[:max_examples]

    all_results = evaluate_models(models, few_shot_preamble, all_examples)
    write_results_to_csv(all_results, "evaluation_results_multistep.csv")
    print("Results saved to evaluation_results.csv")