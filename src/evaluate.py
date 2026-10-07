"""Evaluate generated quizzes against a slice of the RACE dataset."""

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import loadPipeline


def evaluateGeneratedQuizRows(dataPath, limit=25):
    pipeline = loadPipeline()
    dataPath = Path(dataPath)
    if not dataPath.exists():
        raise FileNotFoundError(f"Could not find {dataPath}")

    dataFrame = pd.read_csv(dataPath).head(limit)
    rows = []
    for _, row in dataFrame.iterrows():
        quiz = pipeline.generateQuiz(str(row["article"]))
        rows.append(
            {
                "question": quiz["question"],
                "correctAnswer": quiz["correctAnswer"],
                "predictedAnswer": quiz["prediction"]["answer"],
                "matched": quiz["prediction"]["answer"].lower() == quiz["correctAnswer"].lower(),
                "latencySec": quiz["latencySec"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the unified inference pipeline.")
    parser.add_argument("--data", default="data/raw/train.csv")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    results = evaluateGeneratedQuizRows(args.data, args.limit)
    print(results)
    print({"verifierMatchRate": round(results["matched"].mean(), 4)})


if __name__ == "__main__":
    main()
