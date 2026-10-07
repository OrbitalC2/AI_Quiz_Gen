"""Report distractor quality and fallback behavior on RACE examples."""

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import loadPipeline


def evaluateDistractorQuality(dataPath, limit=25):
    pipeline = loadPipeline()
    dataPath = Path(dataPath)
    if not dataPath.exists():
        raise FileNotFoundError(f"Could not find {dataPath}")

    dataFrame = pd.read_csv(dataPath).head(limit)
    rows = []
    for rowIndex, row in dataFrame.iterrows():
        quiz = pipeline.generateQuiz(str(row["article"]))
        diagnostics = quiz.get("optionDiagnostics", {})
        selected = diagnostics.get("selectedDistractors", [])
        rows.append(
            {
                "row": int(rowIndex),
                "question": quiz["question"],
                "correctAnswer": quiz["correctAnswer"],
                "optionQuality": quiz.get("optionQuality", "unknown"),
                "fallbackUsed": bool(diagnostics.get("fallbackUsed", False)),
                "verifierMatched": quiz["prediction"]["answer"].lower() == quiz["correctAnswer"].lower(),
                "options": " | ".join(quiz["options"].values()),
                "selectedDistractors": " | ".join(item.get("text", "") for item in selected),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate corpus-backed distractor generation.")
    parser.add_argument("--data", default="data/raw/train.csv")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    results = evaluateDistractorQuality(args.data, args.limit)
    print(results[["row", "optionQuality", "fallbackUsed", "verifierMatched", "correctAnswer"]])
    print({"qualityCounts": results["optionQuality"].value_counts().to_dict()})
    print({"verifierMatchRate": round(float(results["verifierMatched"].mean()), 4)})

    weak = results[results["optionQuality"].isin(["relaxed", "last_resort"])]
    if not weak.empty:
        print("\nWeak fallback examples:")
        for _, row in weak.head(10).iterrows():
            print(f"- row={row['row']} quality={row['optionQuality']} answer={row['correctAnswer']}")
            print(f"  question={row['question']}")
            print(f"  options={row['options']}")


if __name__ == "__main__":
    main()
