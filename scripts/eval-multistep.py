import json
import csv
import ollama
from typing import List

# === CONFIG ===
json_file = 'multistep_arithmetic_two.json'  # Path to the JSON file
models = ['gemma3:4b', 'deepseek-r1:1.5b', 'deepseek-r1:8b', 'llama3.2:1b', 'llama3.1:8b']  # Models to evaluate
max_examples = 249  # Maximum number of examples to evaluate
few_shot_preamble = """
Solve multi-step arithmetic problems.

Q: ((-5 + 9 * -4 - 0) * (4 + -7 + 0 * -5)) =
A: Let's think step by step.
Let’s recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
This equation can be written as "A * B", where A = (-5 + 9 * -4 - 0) and B = (4 + -7 + 0 * -5).
Let's calculate A = (-5 + 9 * -4 - 0) = (-5 + (9 * -4) - 0) = (-5 + (-36) - 0) = (-5 + -36 - 0) = -5 - 36 = -41.
Let's calculate B = (4 + -7 + 0 * -5) = (4 + -7 + (0 * -5)) = (4 + -7 + 0) = (4 + -7) = (4 - 7) = -3.
Then, the final equation is A * B = -41 * -3 = (-61) * (-3) = 123. So the answer is 123.
"""

# === FUNCTIONS ===

def load_examples(json_path: str, max_examples: int = None):
    """Load examples from the JSON file, limiting to max_examples if specified."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    examples = data['examples']
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]
    return examples

def evaluate_models(models: List[str], few_shot: str, examples: List[dict]):
    """Evaluate all models on the given examples and save results incrementally."""
    print("\n🧠 Evaluating models...")

    # Define the output size limit
    OUTPUT_LIMIT = 30000

    # Loop through models first to avoid reloading models into memory
    for model in models:
        print(f"\nEvaluating model: {model}")

        # Open a separate CSV file for each model
        output_file = f"{model.replace(':', '_')}_multistep_results.csv"
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header for this model's CSV file
            headers = ["input", "expected", f"{model} output"]
            writer.writerow(headers)

            # Process examples in batches of 20
            batch = []
            for idx, ex in enumerate(examples, start=1):
                row = {
                    "input": ex['input'],
                    "expected": ex['target']
                }

                # Add a note to the prompt to limit the output size
                prompt = f"{few_shot.strip()}\n\nQ:{ex['input']}\nA: (Limit output to 30,000 characters)"
                # print(f"Prompt: {prompt}")

                try:
                    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                    output = response['message']['content']

                    # Check if the output exceeds the limit
                    if len(output) > OUTPUT_LIMIT:
                        output = "CLIPPED " + output[-OUTPUT_LIMIT:]  # Clip from the beginning and prepend "CLIPPED"

                    # print(f"Output from {model}: {output}")

                    # Add the model's output to the row
                    row[f"{model} output"] = output
                except Exception as e:
                    print(f"Error evaluating model {model}: {e}")
                    # Add a placeholder for the model's output in case of failure
                    row[f"{model} output"] = "Error: Model unavailable"

                # Add the row to the batch
                batch.append([row["input"], row["expected"], row[f"{model} output"]])
                print
                # Save to CSV after every 20 examples
                if idx % 20 == 0 or idx == len(examples):
                    writer.writerows(batch)
                    batch = []  # Clear the batch after saving
                    print(f"Saved {idx} examples for model {model} to {output_file}")

    print("All results saved.")

# === MAIN ===

if __name__ == "__main__":
    # Load examples from the JSON file with a limit on the number of examples
    all_examples = load_examples(json_file, max_examples)

    # Evaluate all models on the examples and save results incrementally
    evaluate_models(models, few_shot_preamble, all_examples)