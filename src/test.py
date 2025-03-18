import argparse
import json
from sklearn.metrics import accuracy_score, classification_report
from litgpt import LLM

def load_test_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_model(model_path, test_data, max_tokens):
    model = LLM.load(model_path)
    model.eval()

    preds, labels, results = [], [], []
    for sample in test_data:
        prompt = f"{sample['instruction']}\n\nInput: {sample['input']}\n\nResponse:"
        pred = model.generate(prompt, max_new_tokens=max_tokens).strip().split("\n")[0]
        
        preds.append(pred)
        labels.append(sample["output"].strip())
        results.append({"input": sample["input"], "expected_output": sample["output"], "predicted_output": pred})

    return preds, labels, results

def save_results(accuracy, report, detailed_results, filename="evaluation_results.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "classification_report": report, "detailed_results": detailed_results}, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLM.")
    parser.add_argument("--model_path", required=True, help="Path to the model directory or .pth file.")
    parser.add_argument("--test_json", required=True, help="Path to the test set JSON file.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens per response.")
    args = parser.parse_args()

    test_data = load_test_data(args.test_json)
    print(f"Loaded {len(test_data)} test samples.")

    preds, labels, results = evaluate_model(args.model_path, test_data, args.max_new_tokens)

    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, zero_division=0, output_dict=True)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(classification_report(labels, preds, zero_division=0))

    save_results(accuracy, report, results)

if __name__ == "__main__":
    main()
