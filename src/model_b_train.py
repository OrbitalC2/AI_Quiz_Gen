from pathlib import Path
import argparse
import string

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


OPTION_LABELS = ["A", "B", "C", "D"]


def cleanText(text):
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def loadRaceData(csvPath):
    dataFrame = pd.read_csv(csvPath)
    requiredColumns = {"article", "A", "B", "C", "D", "answer"}
    missingColumns = requiredColumns.difference(dataFrame.columns)
    if missingColumns:
        raise ValueError(f"Missing required columns: {sorted(missingColumns)}")
    return dataFrame


def editDistance(leftText, rightText):
    previousRow = list(range(len(rightText) + 1))
    for leftIndex, leftChar in enumerate(leftText, start=1):
        currentRow = [leftIndex]
        for rightIndex, rightChar in enumerate(rightText, start=1):
            currentRow.append(
                min(
                    currentRow[rightIndex - 1] + 1,
                    previousRow[rightIndex] + 1,
                    previousRow[rightIndex - 1] + (leftChar != rightChar),
                )
            )
        previousRow = currentRow
    return previousRow[-1]


def getDistractorFeatures(correctAnswer, candidateText, articleText, answerEmbedding, candidateEmbedding):
    answerNorm = np.linalg.norm(answerEmbedding)
    candidateNorm = np.linalg.norm(candidateEmbedding)
    cosineSimilarity = (
        np.dot(answerEmbedding, candidateEmbedding) / (answerNorm * candidateNorm)
        if answerNorm > 0 and candidateNorm > 0
        else 0.0
    )

    return [
        editDistance(correctAnswer.lower(), candidateText.lower()),
        abs(len(correctAnswer) - len(candidateText)),
        int(candidateText.lower() in articleText.lower()),
        cosineSimilarity,
    ]


def getFakeDistractors():
    fakeDistractors = []
    try:
        for synset in wordnet.synsets("object")[:2]:
            for lemma in synset.lemmas()[:2]:
                fakeDistractors.append(lemma.name().replace("_", " "))
    except LookupError:
        fakeDistractors = []

    return fakeDistractors or ["object", "thing", "unknown", "item"]


def buildDistractorTrainingRows(dataFrame, embedder):
    rows = []
    fakeDistractors = getFakeDistractors()

    for _, row in dataFrame.iterrows():
        articleText = str(row["article"])
        cleanArticle = cleanText(articleText)
        answerLabel = str(row["answer"])
        answerText = str(row[answerLabel])
        answerEmbedding = embedder.encode([answerText])[0]

        for optionLabel in OPTION_LABELS:
            if optionLabel == answerLabel:
                continue
            distractorText = str(row[optionLabel])
            distractorEmbedding = embedder.encode([distractorText])[0]
            rows.append(
                {
                    "features": getDistractorFeatures(
                        answerText,
                        distractorText,
                        cleanArticle,
                        answerEmbedding,
                        distractorEmbedding,
                    ),
                    "label": 1,
                }
            )

        for fakeText in fakeDistractors:
            fakeEmbedding = embedder.encode([fakeText])[0]
            rows.append(
                {
                    "features": getDistractorFeatures(
                        answerText,
                        fakeText,
                        cleanArticle,
                        answerEmbedding,
                        fakeEmbedding,
                    ),
                    "label": 0,
                }
            )

    return rows


def summarizeClassifier(name, model, xTest, yTest):
    predictions = model.predict(xTest)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yTest,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "modelName": name,
        "accuracy": round(accuracy_score(yTest, predictions), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusionMatrix": confusion_matrix(yTest, predictions).tolist(),
    }


def trainDistractorRanker(
    trainCsv,
    outputDir="models/model_b/traditional",
    sampleSize=3000,
    randomState=42,
):
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    dataFrame = loadRaceData(trainCsv)
    sampleSize = min(sampleSize, len(dataFrame))
    sampledData = dataFrame.sample(n=sampleSize, random_state=randomState).reset_index(drop=True)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    trainingRows = buildDistractorTrainingRows(sampledData, embedder)
    xData = np.array([row["features"] for row in trainingRows])
    yData = np.array([row["label"] for row in trainingRows])
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=randomState)

    rfModel = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=randomState)
    rfModel.fit(xTrain, yTrain)

    lrBaseline = LogisticRegression(max_iter=1000)
    lrBaseline.fit(xTrain, yTrain)

    joblib.dump(rfModel, outputDir / "rfDistractorRanker.joblib")
    joblib.dump(lrBaseline, outputDir / "lrDistractorBaseline.joblib")

    return {
        "randomForest": summarizeClassifier("Random Forest", rfModel, xTest, yTest),
        "logisticRegression": summarizeClassifier("Logistic Regression", lrBaseline, xTest, yTest),
        "rows": len(trainingRows),
        "outputDir": str(outputDir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Model B distractor ranker from RACE options.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--output-dir", default="models/model_b/traditional")
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args()

    metrics = trainDistractorRanker(args.train_csv, args.output_dir, args.sample_size)
    print(metrics)


if __name__ == "__main__":
    main()
