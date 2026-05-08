"""
question_ranker_train.py
------------------------
Train a traditional ML ranker for rule-generated question candidates.

The model learns to choose among template-generated candidates. It does not use
T5/BERT or any neural text generation. RACE human questions are used only as
training supervision.

Usage
-----
python3 src/question_ranker_train.py \
  --train-csv data/raw/train.csv \
  --output-dir models/model_a/traditional \
  --sample-size 30000
"""

from pathlib import Path
import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from question_ranker import (  # noqa: E402
    QUESTION_RANKER_FEATURE_NAMES,
    buildQuestionRankerFeatureVector,
    findBestSourceSentence,
    generateQuestionCandidates,
    goldQuestionSimilarity,
    inferAnswerKind,
    isObviouslyBadQuestion,
    tokenise,
)
from template_quiz_pipeline import (  # noqa: E402
    AnswerCandidate,
    EnsembleAnswerPredictor,
    findAnswerCandidates,
)


OPTION_LABELS = ["A", "B", "C", "D"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "model_a" / "traditional"


def loadRaceData(csvPath):
    dataFrame = pd.read_csv(csvPath)
    requiredColumns = {"article", "question", "A", "B", "C", "D", "answer"}
    missingColumns = requiredColumns.difference(dataFrame.columns)
    if missingColumns:
        raise ValueError(f"Missing required columns: {sorted(missingColumns)}")
    return dataFrame.dropna(subset=["article", "question", "answer"]).reset_index(drop=True)


def getCorrectAnswer(row):
    answerLabel = str(row["answer"]).strip()
    if answerLabel not in OPTION_LABELS:
        return None
    return str(row[answerLabel])


def getRaceOptions(row):
    return [str(row[label]) for label in OPTION_LABELS]


def buildOptionsForCandidate(candidateAnswer, row):
    """
    Build a four-option set that includes the candidate answer.

    During training this lets the Model A margin feature answer:
    "If this question/answer candidate were used, would Model A clearly prefer
    its intended answer over plausible RACE distractors?"
    """
    options = [str(candidateAnswer)]
    for option in getRaceOptions(row):
        if len(options) == 4:
            break
        if str(option).lower() not in {item.lower() for item in options}:
            options.append(str(option))

    fallbackOptions = ["related detail", "background fact", "supporting idea", "none of these"]
    for option in fallbackOptions:
        if len(options) == 4:
            break
        if option.lower() not in {item.lower() for item in options}:
            options.append(option)
    return options[:4]


def addGoldAnswerCandidate(articleText, correctAnswer, candidates):
    """
    Inject the RACE correct option as an answer candidate for training.

    RACE answers are not always exact named entities from the passage, so this
    makes the training candidate pool closer to the supervised target.
    """
    if not correctAnswer:
        return candidates

    if any(candidate.text.lower() == str(correctAnswer).lower() for candidate in candidates):
        return candidates

    sentence, _ = findBestSourceSentence(articleText, correctAnswer)
    kind = inferAnswerKind(correctAnswer, sentence)
    injected = AnswerCandidate(
        text=str(correctAnswer),
        sentence=sentence,
        kind=kind,
        score=7.5,
    )
    return [injected] + candidates


def isGoldClozeQuestion(goldQuestion):
    text = str(goldQuestion).lower()
    return "_" in text or "complete" in text or "fill" in text


def computeTrainingTarget(candidate, goldQuestion, correctAnswer):
    """
    Convert a generated candidate into a continuous quality score.

    The human RACE question is used only here, as supervision. At inference time
    the trained ranker sees only template/answer/source/Model-A features.
    """
    similarity = goldQuestionSimilarity(candidate.question, goldQuestion)

    answerMatchesGold = str(candidate.answer_text).lower() == str(correctAnswer).lower()
    if not answerMatchesGold:
        similarity *= 0.35

    if candidate.template_type == "cloze" and not isGoldClozeQuestion(goldQuestion):
        similarity *= 0.55

    if isObviouslyBadQuestion(candidate.question):
        similarity *= 0.20

    return float(similarity)


def labelGroup(groupFrame, positiveThreshold, negativeThreshold):
    """
    Create conservative binary labels from continuous target scores.

    Only the best clear candidate in a group is labelled positive. Everything
    weak, malformed, or cloze-by-default remains negative. This intentionally
    favours precision over recall for the ranker.
    """
    labels = np.zeros(len(groupFrame), dtype=np.int64)
    if groupFrame.empty:
        return labels

    bestIndex = int(groupFrame["targetScore"].astype(float).idxmax())
    bestScore = float(groupFrame.loc[bestIndex, "targetScore"])
    if bestScore >= positiveThreshold:
        labels[list(groupFrame.index).index(bestIndex)] = 1

    clearPositiveMask = groupFrame["targetScore"].astype(float) >= max(positiveThreshold + 0.08, 0.45)
    for localIndex, isClear in enumerate(clearPositiveMask.tolist()):
        if isClear:
            labels[localIndex] = 1

    clearNegativeMask = groupFrame["targetScore"].astype(float) <= negativeThreshold
    for localIndex, isClear in enumerate(clearNegativeMask.tolist()):
        if isClear:
            labels[localIndex] = 0
    return labels


def buildTrainingFrame(
    dataFrame,
    sampleSize=30000,
    maxAnswerCandidates=10,
    answerPredictor=None,
    randomState=42,
    positiveThreshold=0.22,
    negativeThreshold=0.08,
):
    if sampleSize and sampleSize < len(dataFrame):
        dataFrame = dataFrame.sample(n=sampleSize, random_state=randomState).reset_index(drop=True)

    rows = []
    skipped = 0
    startTime = time.perf_counter()

    for rowIndex, row in dataFrame.iterrows():
        articleText = str(row["article"])
        goldQuestion = str(row["question"])
        correctAnswer = getCorrectAnswer(row)
        if not correctAnswer:
            skipped += 1
            continue

        answerCandidates = findAnswerCandidates(articleText)
        answerCandidates = addGoldAnswerCandidate(articleText, correctAnswer, answerCandidates)
        questionCandidates = generateQuestionCandidates(
            articleText,
            answerCandidates,
            maxAnswerCandidates=maxAnswerCandidates,
        )
        if not questionCandidates:
            skipped += 1
            continue

        for candidateIndex, candidate in enumerate(questionCandidates):
            options = buildOptionsForCandidate(candidate.answer_text, row)
            featureVector = buildQuestionRankerFeatureVector(
                articleText,
                candidate,
                options=options,
                answerPredictor=answerPredictor,
            )
            targetScore = computeTrainingTarget(candidate, goldQuestion, correctAnswer)
            rows.append(
                {
                    "groupId": rowIndex,
                    "candidateIndex": candidateIndex,
                    "question": candidate.question,
                    "goldQuestion": goldQuestion,
                    "answerText": candidate.answer_text,
                    "correctAnswer": correctAnswer,
                    "templateType": candidate.template_type,
                    "targetScore": targetScore,
                    "featureVector": featureVector,
                }
            )

        if (rowIndex + 1) % 1000 == 0:
            elapsed = time.perf_counter() - startTime
            totalRows = len(dataFrame)
            done = rowIndex + 1
            left = totalRows - done
            percent = (done / totalRows) * 100
            print(f"[question-ranker] rows={done:,}/{totalRows:,} ({percent:.1f}%) left={left:,} candidates={len(rows):,} elapsed={elapsed:.1f}s")

    trainingFrame = pd.DataFrame(rows)
    if trainingFrame.empty:
        raise ValueError("No question-ranker training rows were generated.")

    labels = np.zeros(len(trainingFrame), dtype=np.int64)
    for _, groupFrame in trainingFrame.groupby("groupId", sort=False):
        groupLabels = labelGroup(groupFrame, positiveThreshold, negativeThreshold)
        labels[groupFrame.index.to_numpy()] = groupLabels
    trainingFrame["label"] = labels
    return trainingFrame, skipped


def toFeatureMatrix(trainingFrame):
    matrix = np.array(trainingFrame["featureVector"].tolist(), dtype=np.float32)
    if matrix.shape[1] != len(QUESTION_RANKER_FEATURE_NAMES):
        raise ValueError(
            f"Feature count mismatch: got {matrix.shape[1]}, "
            f"expected {len(QUESTION_RANKER_FEATURE_NAMES)}"
        )
    return matrix


def evaluateRanking(model, frame, xMatrix):
    frame = frame.copy()
    frame["predictedScore"] = model.predict_proba(xMatrix)[:, 1]

    selectedRows = []
    oracleRows = []
    for _, groupFrame in frame.groupby("groupId", sort=False):
        selectedRows.append(groupFrame.loc[groupFrame["predictedScore"].astype(float).idxmax()])
        oracleRows.append(groupFrame.loc[groupFrame["targetScore"].astype(float).idxmax()])

    selected = pd.DataFrame(selectedRows)
    oracle = pd.DataFrame(oracleRows)
    exactBest = (
        selected["candidateIndex"].astype(int).to_numpy()
        == oracle["candidateIndex"].astype(int).to_numpy()
    )

    return {
        "groups": int(frame["groupId"].nunique()),
        "top1OracleMatch": round(float(exactBest.mean()), 4) if len(exactBest) else 0.0,
        "selectedMeanGoldScore": round(float(selected["targetScore"].mean()), 4),
        "oracleMeanGoldScore": round(float(oracle["targetScore"].mean()), 4),
        "selectedClozeRate": round(float((selected["templateType"] == "cloze").mean()), 4),
        "oracleClozeRate": round(float((oracle["templateType"] == "cloze").mean()), 4),
    }


def summarizeBinary(model, xTest, yTest):
    predictions = model.predict(xTest)
    probabilities = model.predict_proba(xTest)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        yTest,
        predictions,
        average="binary",
        zero_division=0,
    )
    try:
        rocAuc = roc_auc_score(yTest, probabilities)
    except ValueError:
        rocAuc = 0.0
    return {
        "accuracy": round(float(accuracy_score(yTest, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "rocAuc": round(float(rocAuc), 4),
        "confusionMatrix": confusion_matrix(yTest, predictions).tolist(),
    }


def trainQuestionRanker(
    trainCsv,
    outputDir=DEFAULT_OUTPUT_DIR,
    sampleSize=30000,
    maxAnswerCandidates=10,
    useModelAFeature=True,
    randomState=42,
    positiveThreshold=0.22,
    negativeThreshold=0.08,
):
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    print(f"[question-ranker] loading {trainCsv}")
    dataFrame = loadRaceData(trainCsv)

    answerPredictor = None
    if useModelAFeature:
        print("[question-ranker] loading Model A verifier for margin features")
        answerPredictor = EnsembleAnswerPredictor.fromZip()

    trainingFrame, skipped = buildTrainingFrame(
        dataFrame,
        sampleSize=sampleSize,
        maxAnswerCandidates=maxAnswerCandidates,
        answerPredictor=answerPredictor,
        randomState=randomState,
        positiveThreshold=positiveThreshold,
        negativeThreshold=negativeThreshold,
    )

    positiveCount = int(trainingFrame["label"].sum())
    if positiveCount == 0:
        raise ValueError(
            "No positive question-ranker labels were produced. "
            "Lower --positive-threshold or inspect candidate generation."
        )

    xData = toFeatureMatrix(trainingFrame)
    yData = trainingFrame["label"].to_numpy(dtype=np.int64)
    groups = trainingFrame["groupId"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=randomState)
    trainIndex, testIndex = next(splitter.split(xData, yData, groups))
    xTrain, xTest = xData[trainIndex], xData[testIndex]
    yTrain, yTest = yData[trainIndex], yData[testIndex]
    trainFrame = trainingFrame.iloc[trainIndex].reset_index(drop=True)
    testFrame = trainingFrame.iloc[testIndex].reset_index(drop=True)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=randomState,
        n_jobs=-1,
    )
    model.fit(xTrain, yTrain)

    binaryMetrics = summarizeBinary(model, xTest, yTest)
    trainRankingMetrics = evaluateRanking(model, trainFrame, xTrain)
    testRankingMetrics = evaluateRanking(model, testFrame, xTest)

    modelPath = outputDir / "rfQuestionRanker.joblib"
    metadataPath = outputDir / "questionRankerMetadata.json"
    featurePath = outputDir / "questionRankerFeatures.json"
    previewPath = outputDir / "questionRankerTrainingPreview.csv"

    joblib.dump(model, modelPath)

    importances = sorted(
        zip(QUESTION_RANKER_FEATURE_NAMES, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    metadata = {
        "modelType": "RandomForestClassifier",
        "rows": int(len(trainingFrame)),
        "groups": int(trainingFrame["groupId"].nunique()),
        "skippedRows": int(skipped),
        "positiveRows": positiveCount,
        "negativeRows": int(len(trainingFrame) - positiveCount),
        "positiveThreshold": positiveThreshold,
        "negativeThreshold": negativeThreshold,
        "sampleSize": sampleSize,
        "maxAnswerCandidates": maxAnswerCandidates,
        "usesModelAFeature": bool(useModelAFeature),
        "binaryMetrics": binaryMetrics,
        "trainRankingMetrics": trainRankingMetrics,
        "testRankingMetrics": testRankingMetrics,
        "topFeatureImportances": [
            {"feature": name, "importance": round(float(value), 6)}
            for name, value in importances[:20]
        ],
    }

    metadataPath.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    featurePath.write_text(json.dumps(QUESTION_RANKER_FEATURE_NAMES, indent=2), encoding="utf-8")
    trainingFrame.drop(columns=["featureVector"]).head(1000).to_csv(previewPath, index=False)

    return {
        "modelPath": str(modelPath),
        "metadataPath": str(metadataPath),
        "featurePath": str(featurePath),
        "previewPath": str(previewPath),
        **metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a traditional ML ranker for generated questions.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--max-answer-candidates", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--positive-threshold", type=float, default=0.22)
    parser.add_argument("--negative-threshold", type=float, default=0.08)
    parser.add_argument(
        "--no-model-a-feature",
        action="store_true",
        help="Disable Model A score/margin features for faster experiments.",
    )
    args = parser.parse_args()

    metrics = trainQuestionRanker(
        trainCsv=args.train_csv,
        outputDir=args.output_dir,
        sampleSize=args.sample_size,
        maxAnswerCandidates=args.max_answer_candidates,
        useModelAFeature=not args.no_model_a_feature,
        randomState=args.random_state,
        positiveThreshold=args.positive_threshold,
        negativeThreshold=args.negative_threshold,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

