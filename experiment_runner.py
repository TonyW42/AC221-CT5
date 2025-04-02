import os
import pandas as pd
from tabulate import tabulate
from alpha_prototype import AlphaPrototype

def run_experiment():
    system = AlphaPrototype(model_name="VGG-Face")

    # Force re-initialization to ensure consistency
    print(system.initialize(force_reinitialize=True))

    results = []
    probe_dir = "probe_db"

    for person in os.listdir(probe_dir):
        probe_path = os.path.join(probe_dir, person)
        if not os.path.isdir(probe_path):
            continue

        for image in os.listdir(probe_path):
            img_path = os.path.join(probe_path, image)

            predicted_name, confidence = system.find_match(img_path)

            results.append({
                "Actual": person,
                "Predicted": predicted_name,
                "Confidence": round(confidence, 3) if confidence else None,
                "Correct": person == predicted_name
            })

    df = pd.DataFrame(results)
    accuracy = (df["Correct"].sum() / len(df)) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print("\nDetailed Results:")
    print(tabulate(df, headers="keys", tablefmt="grid"))

    df.to_csv("experiment_results.csv", index=False)
    print("\nResults saved to experiment_results.csv")

if __name__ == "__main__":
    run_experiment()
