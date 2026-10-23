from __future__ import annotations

from model import train_and_evaluate
from student_data import load_uci_records


def main() -> None:
    records = load_uci_records()
    result = train_and_evaluate(records)
    metrics = result["metrics"]
    print(f"Dataset records: {len(records)}")
    print(f"Training records: {result['train_count']}")
    print(f"Holdout records: {result['test_count']}")
    print("Stratified split: 80/20")
    print("Pass threshold: G3 >= 10/20")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Pass precision: {metrics['precision']:.3f}")
    print(f"Pass recall: {metrics['recall']:.3f}")
    print(f"Pass F1: {metrics['f1']:.3f}")
    print("\nHoldout confusion matrix:")
    print(result["confusion_matrix"].to_string())


if __name__ == "__main__":
    main()
