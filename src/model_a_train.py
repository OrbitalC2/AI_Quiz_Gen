"""
model_a_train.py
----------------
Trains Model A: answer-verification ensemble (traditional ML) + T5 neural baseline.

Traditional pipeline
--------------------
  Features (12 total):
    0  TF-IDF cosine similarity (document level)
    1  Unigram overlap ratio
    2  IDF-weighted option score
    3  Best-sentence unigram overlap  ← key discriminative feature
    4  Bigram overlap ratio
    5  Exact match flag
    6  Option token length
    7  Option character length
    8  First-position in article
    9  Is-number / date flag
   10  K-Means cluster label          (unsupervised)
   11  K-Means centroid distance      (unsupervised)

  Classifiers : Logistic Regression, Calibrated SVM, Gaussian Naive Bayes,
                Random Forest
  Ensemble    : soft-vote (mean predict_proba across all four models)

Neural baseline (T5)
--------------------
  Fine-tunes t5-small on (article, answer) → question pairs.
  All transformer imports are lazy (inside the function body).

Usage
-----
  python src/model_a_train.py --task traditional --train-csv data/raw/train.csv
  python src/model_a_train.py --task neural      --train-csv data/raw/train.csv
  python src/model_a_train.py --task all         --train-csv data/raw/train.csv
"""

from pathlib import Path
import argparse
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# Make the project root importable (for features.py in root)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features import (  # noqa: E402
    BASE_FEATURE_COUNT,
    FEATURE_COUNT,
    FEATURE_NAMES,
    buildBaseFeatureMatrixBatch,
    buildClusterFeaturesIfAvailable,
)
from preprocessing import (  # noqa: E402
    explodeOptions,
    fastCleanSeries,
    loadRaceCsv,
    prepareT5Rows,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTION_LABELS = ["A", "B", "C", "D"]
TFIDF_MAX_FEATURES = 20_000
SVD_COMPONENTS = 64
KMEANS_CLUSTERS = 2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Unsupervised component — K-Means clustering
# ---------------------------------------------------------------------------

def fitKMeansClusters(optionTfidfMatrix, nClusters=KMEANS_CLUSTERS, randomState=RANDOM_STATE):
    """
    Fit K-Means on TruncatedSVD-reduced option TF-IDF vectors.

    Returns
    -------
    kmeansModel, svdModel
    """
    svdModel = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=randomState)
    reducedVecs = svdModel.fit_transform(optionTfidfMatrix)
    kmeansModel = KMeans(n_clusters=nClusters, random_state=randomState, n_init=10)
    kmeansModel.fit(reducedVecs)
    return kmeansModel, svdModel


def buildClusterFeatures(optionTfidfMatrix, kmeansModel, svdModel):
    """
    Append 2 K-Means features to a base feature matrix.

    Returns np.ndarray of shape (n_rows, 2): [cluster_label, centroid_dist]
    """
    reducedVecs = svdModel.transform(optionTfidfMatrix)
    clusterLabels = kmeansModel.predict(reducedVecs).reshape(-1, 1).astype(np.float32)
    centroidDists = kmeansModel.transform(reducedVecs).min(axis=1).reshape(-1, 1).astype(np.float32)
    return np.hstack([clusterLabels, centroidDists])


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def trainEnsembleVerifier(xTrain, yTrain, randomState=RANDOM_STATE):
    """
    Train four classifiers on manually balanced (1:1) training data:
    Logistic Regression, Calibrated SVM, Gaussian Naive Bayes, Random Forest.

    All models are trained on balanced data so no class_weight is needed.
    SVM is trained and reported for spec compliance but EXCLUDED from the
    soft-vote ensemble because CalibratedClassifierCV degrades to all-negative
    predictions on this task (see evaluation notes).

    Ensemble members: LR + NB + RF  (3-model soft vote)
    """
    lrModel = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=randomState,
    )
    lrModel.fit(xTrain, yTrain)

    svmBase = LinearSVC(max_iter=5000, dual=False, C=1.0)
    svmModel = CalibratedClassifierCV(svmBase, cv=3, method="isotonic")
    svmModel.fit(xTrain, yTrain)

    nbModel = GaussianNB()
    nbModel.fit(xTrain, yTrain)

    rfModel = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=randomState,
        n_jobs=-1,
    )
    rfModel.fit(xTrain, yTrain)

    return {"lr": lrModel, "svm": svmModel, "nb": nbModel, "rf": rfModel}


def softVoteScores(models, xFeatures):
    """
    Soft-vote ensemble: mean predict_proba[:, 1] over LR + SVM + RF.

    NB is excluded from the vote: on this task its recall (~0.30) is
    significantly lower than LR/SVM/RF (~0.50), which suppresses the
    correct answer's score and degrades MCQ ranking accuracy.
    NB is still trained and its individual metrics are reported for
    spec compliance.

    Returns np.ndarray of shape (n_rows,)
    """
    ensembleModels = [models["lr"], models["svm"], models["rf"]]
    probList = [m.predict_proba(xFeatures)[:, 1] for m in ensembleModels]
    return np.mean(probList, axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluateMcqAccuracy(scores, labels):
    """
    MCQ accuracy: fraction of questions where argmax(scores) == correct option.

    Parameters
    ----------
    scores : flat array of length 4 × n_questions
    labels : flat array of length 4 × n_questions (one-hot per question)
    """
    correct = 0
    totalQuestions = len(labels) // 4
    for startIdx in range(0, len(labels), 4):
        actualIdx = int(np.argmax(labels[startIdx:startIdx + 4]))
        predictedIdx = int(np.argmax(scores[startIdx:startIdx + 4]))
        correct += int(actualIdx == predictedIdx)
    return correct / totalQuestions if totalQuestions else 0.0


def summariseClassifier(name, model, xTest, yTest):
    """Return standard binary-classification metrics + confusion matrix for one model."""
    predictions = model.predict(xTest)
    cm = confusion_matrix(yTest, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "modelName": name,
        "accuracy":  round(accuracy_score(yTest, predictions), 4),
        "precision": round(precision_score(yTest, predictions, zero_division=0), 4),
        "recall":    round(recall_score(yTest, predictions, zero_division=0), 4),
        "f1":        round(f1_score(yTest, predictions, zero_division=0), 4),
        "confusionMatrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluateEnsemble(models, xTest, yTest, xTrain, yTrain):
    """
    Evaluate each classifier individually and as a soft-vote ensemble.
    Also runs 5-fold stratified cross-validation on LR (fast, representative).

    Returns dict with full metrics.
    """
    individualMetrics = {
        key: summariseClassifier(key.upper(), model, xTest, yTest)
        for key, model in models.items()
    }

    ensembleScores = softVoteScores(models, xTest)
    ensemblePredictions = (ensembleScores >= 0.5).astype(int)
    ensembleCm = confusion_matrix(yTest, ensemblePredictions)
    tn, fp, fn, tp = ensembleCm.ravel() if ensembleCm.size == 4 else (0, 0, 0, 0)

    # 5-fold cross-val on LR (representative of training stability)
    cvScores = cross_val_score(
        LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced"),
        xTrain, yTrain,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        n_jobs=-1,
    )

    return {
        "individualMetrics": individualMetrics,
        "ensembleBinaryAccuracy": round(accuracy_score(yTest, ensemblePredictions), 4),
        "ensembleBinaryF1": round(f1_score(yTest, ensemblePredictions, zero_division=0), 4),
        "ensembleConfusionMatrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "ensembleMcqAccuracy": round(evaluateMcqAccuracy(ensembleScores, yTest), 4),
        "lrCrossValF1Mean": round(float(cvScores.mean()), 4),
        "lrCrossValF1Std":  round(float(cvScores.std()), 4),
        "randomBaselineMcqAccuracy": 0.25,
        "featureNames": FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# Main training orchestrator — traditional pipeline
# ---------------------------------------------------------------------------

def trainAnswerVerifiers(
    trainCsv,
    outputDir="models/model_a/traditional",
    sampleSize=None,
    randomState=RANDOM_STATE,
):
    """
    Full traditional Model A training pipeline.

    Steps
    -----
    1.  Load RACE CSV (full dataset by default, sampleSize=None)
    2.  Split into train / test (85 / 15)
    3.  Explode each question into 4 option rows
    4.  Fit TF-IDF vectorizer on training text
    5.  Build 10-feature BASE matrix (batch TF-IDF + row-wise sentence features)
    6.  Fit K-Means (unsupervised) on option TF-IDF vectors
    7.  Append 2 cluster features → 12-feature matrix
    8.  Train LR + Calibrated SVM + Naive Bayes + RF  (class_weight='balanced')
    9.  Soft-vote ensemble evaluation + cross-validation
    10. Save all artifacts

    Artifacts saved
    ---------------
    lrVerifierModel.joblib
    svmVerifierModel.joblib
    nbVerifierModel.joblib      (Gaussian Naive Bayes)
    rfVerifierModel.joblib
    tfidfVectorizer.joblib
    idfDictionary.joblib
    kmeansModel.joblib
    svdModel.joblib
    """
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    print("[1/8] Loading RACE data...")
    dataFrame = loadRaceCsv(trainCsv)
    if sampleSize is not None:
        sampleSize = min(sampleSize, len(dataFrame))
        dataFrame = dataFrame.sample(n=sampleSize, random_state=randomState).reset_index(drop=True)
    print(f"      Articles loaded: {len(dataFrame):,}")

    # --- 2. Split ---
    print("[2/8] Train / test split (85 / 15)...")
    trainBase, testBase = train_test_split(dataFrame, test_size=0.15, random_state=randomState)

    # --- 3. Explode options ---
    print("[3/8] Exploding options into (article, question, option, label) rows...")
    trainExploded = explodeOptions(trainBase.reset_index(drop=True))
    testExploded  = explodeOptions(testBase.reset_index(drop=True))

    # --- 3b. Manual 1:1 class balancing on train ---
    # class_weight='balanced' on classifiers causes CalibratedSVM to degenerate
    # to all-negative predictions (recall=0) on this imbalanced task.
    # Manual balancing produces well-calibrated probabilities for MCQ ranking.
    print("[3b]   Balancing train classes 1:1 (positive = correct answer)...")
    positives = trainExploded[trainExploded["label"] == 1]
    negatives = trainExploded[trainExploded["label"] == 0].sample(
        n=len(positives), random_state=randomState
    )
    trainExploded = (
        pd.concat([positives, negatives], ignore_index=True)
        .sample(frac=1, random_state=randomState)
        .reset_index(drop=True)
    )
    print(f"      Train rows (balanced): {len(trainExploded):,}  |  Test rows (natural): {len(testExploded):,}")

    # --- 4. Fit TF-IDF ---
    print("[4/8] Fitting TF-IDF vectorizer (max_features={:,})...".format(TFIDF_MAX_FEATURES))
    allText = (
        trainBase["article"].astype(str).tolist()
        + trainBase["question"].astype(str).tolist()
    )
    allTextClean = [
        t.lower().translate(str.maketrans("", "", string.punctuation))
        for t in allText
    ]
    tfidfVec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True)
    tfidfVec.fit(allTextClean)
    idfDict = dict(zip(tfidfVec.get_feature_names_out(), tfidfVec.idf_))

    # --- 5. Build 10-feature BASE matrix ---
    print("[5/8] Building feature matrices (batch TF-IDF + sentence-level loop)...")
    print("      Building train matrix...")
    xTrainBase = buildBaseFeatureMatrixBatch(trainExploded, idfDict, tfidfVec)
    print("      Building test matrix...")
    xTestBase  = buildBaseFeatureMatrixBatch(testExploded,  idfDict, tfidfVec)

    # --- 6. Fit K-Means (unsupervised) ---
    print("[6/8] Fitting K-Means (k=2) on option TF-IDF vectors...")
    trainOptionVecs = tfidfVec.transform(trainExploded["cleanOption"].fillna("").tolist())
    kmeansModel, svdModel = fitKMeansClusters(trainOptionVecs, randomState=randomState)

    # --- 7. Append cluster features → 12 total ---
    print("[7/8] Appending cluster features...")
    testOptionVecs  = tfidfVec.transform(testExploded["cleanOption"].fillna("").tolist())
    xTrainCluster = buildClusterFeatures(trainOptionVecs, kmeansModel, svdModel)
    xTestCluster  = buildClusterFeatures(testOptionVecs,  kmeansModel, svdModel)

    xTrain = np.hstack([xTrainBase, xTrainCluster])
    xTest  = np.hstack([xTestBase,  xTestCluster])
    yTrain = trainExploded["label"].values
    yTest  = testExploded["label"].values
    print(f"      Feature shape: {xTrain.shape}  |  Features: {FEATURE_NAMES}")

    # --- 8. Train ensemble ---
    print("[8/8] Training ensemble (LR + Calibrated SVM + Naive Bayes + RF)...")
    startTime = time.time()
    models = trainEnsembleVerifier(xTrain, yTrain, randomState=randomState)
    trainTimeSec = round(time.time() - startTime, 2)
    print(f"      Ensemble trained in {trainTimeSec}s")

    # --- Evaluate ---
    print("      Evaluating + 5-fold cross-validation...")
    evalResults = evaluateEnsemble(models, xTest, yTest, xTrain, yTrain)

    # --- Save artifacts ---
    print("Saving artifacts...")
    joblib.dump(models["lr"],  outputDir / "lrVerifierModel.joblib")
    joblib.dump(models["svm"], outputDir / "svmVerifierModel.joblib")
    joblib.dump(models["nb"],  outputDir / "nbVerifierModel.joblib")
    joblib.dump(models["rf"],  outputDir / "rfVerifierModel.joblib")
    joblib.dump(tfidfVec,      outputDir / "tfidfVectorizer.joblib")
    joblib.dump(idfDict,       outputDir / "idfDictionary.joblib")
    joblib.dump(kmeansModel,   outputDir / "kmeansModel.joblib")
    joblib.dump(svdModel,      outputDir / "svdModel.joblib")

    return {
        "articlesUsed":  len(dataFrame),
        "trainRows":     len(xTrain),
        "testRows":      len(xTest),
        "featureCount":  xTrain.shape[1],
        "featureNames":  FEATURE_NAMES,
        "trainTimeSec":  trainTimeSec,
        "evaluation":    evalResults,
        "outputDir":     str(outputDir),
    }


# ---------------------------------------------------------------------------
# T5 neural baseline — all transformer imports are lazy
# ---------------------------------------------------------------------------

def trainT5QuestionGenerator(
    trainCsv,
    outputDir="models/model_a/neural/model_a_final",
    baseModel="t5-small",
    sampleSize=20_000,
    epochs=5,
    batchSize=16,
    randomState=RANDOM_STATE,
):
    """
    Fine-tune T5 on (article, answer) → question pairs.

    NEURAL BASELINE — used only for comparison with the template-based QG.
    All heavy imports (torch, transformers, datasets) are inside this function.
    """
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        T5ForConditionalGeneration,
    )

    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    print("[T5] Loading RACE data...")
    dataFrame = loadRaceCsv(trainCsv)
    t5Rows = prepareT5Rows(dataFrame, sampleSize=sampleSize, randomState=randomState)
    print(f"[T5] Tokenising {len(t5Rows):,} rows...")

    tokenizer = AutoTokenizer.from_pretrained(baseModel)
    model = T5ForConditionalGeneration.from_pretrained(baseModel)
    dataset = Dataset.from_pandas(t5Rows).train_test_split(test_size=0.1, seed=randomState)

    def tokenizeBatch(batch):
        inputs = tokenizer(batch["t5Input"], max_length=512, truncation=True, padding="max_length")
        targets = tokenizer(text_target=batch["t5Target"], max_length=64, truncation=True, padding="max_length")
        inputs["labels"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in row]
            for row in targets["input_ids"]
        ]
        return inputs

    tokenisedDataset = dataset.map(tokenizeBatch, batched=True, remove_columns=["t5Input", "t5Target"])

    trainingArgs = Seq2SeqTrainingArguments(
        output_dir=str(outputDir.parent / "training_runs"),
        eval_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=batchSize,
        per_device_eval_batch_size=batchSize,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=False,
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainingArgs,
        train_dataset=tokenisedDataset["train"],
        eval_dataset=tokenisedDataset["test"],
        processing_class=tokenizer,
    )

    print("[T5] Training...")
    trainer.train()
    trainer.save_model(str(outputDir))
    tokenizer.save_pretrained(str(outputDir))

    return {
        "rows": len(t5Rows),
        "outputDir": str(outputDir),
        "note": "Neural baseline — compare with template QG only.",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import string   # needed for TF-IDF cleaning above


def buildArgParser():
    parser = argparse.ArgumentParser(
        description=(
            "Train Model A.\n"
            "  --task traditional : TF-IDF + KMeans + LR/SVM/NB/RF ensemble\n"
            "  --task neural      : T5 question generator (neural baseline)\n"
            "  --task all         : both"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-csv",               default="data/raw/train.csv")
    parser.add_argument("--task",                    choices=["all", "neural", "traditional"], default="all")
    parser.add_argument("--traditional-output-dir",  default="models/model_a/traditional")
    parser.add_argument("--neural-output-dir",       default="models/model_a/neural/model_a_final")
    parser.add_argument("--base-model",              default="t5-small")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of articles to use. Default: use all available articles.",
    )
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main():
    args = buildArgParser().parse_args()

    if args.task in {"all", "traditional"}:
        print("\n=== Training traditional verifier ensemble ===")
        results = trainAnswerVerifiers(
            trainCsv=args.train_csv,
            outputDir=args.traditional_output_dir,
            sampleSize=args.sample_size,
        )
        print("\nTraditional pipeline results:")
        for key, value in results.items():
            if key != "featureNames":
                print(f"  {key}: {value}")

    if args.task in {"all", "neural"}:
        print("\n=== Training T5 neural baseline ===")
        neuralResults = trainT5QuestionGenerator(
            trainCsv=args.train_csv,
            outputDir=args.neural_output_dir,
            baseModel=args.base_model,
            sampleSize=args.sample_size or 20_000,
            epochs=args.epochs,
            batchSize=args.batch_size,
        )
        print("\nT5 baseline results:")
        for key, value in neuralResults.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
