"""
preprocessing.py
----------------
Shared data-loading and cleaning utilities for Model A and Model B.

All functions are pure (no side effects) and return new DataFrames.
"""

from pathlib import Path
import string

import pandas as pd
from sklearn.model_selection import train_test_split


OPTION_LABELS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def cleanText(text):
    """Lowercase and strip punctuation from a single string."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def fastCleanSeries(textSeries):
    """Vectorised version of cleanText for a whole pandas Series."""
    return (
        textSeries.astype(str)
        .str.lower()
        .str.translate(str.maketrans("", "", string.punctuation))
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def loadRaceCsv(csvPath):
    """Load a RACE CSV and attach a correctAnswerText column."""
    csvPath = Path(csvPath)
    dataFrame = pd.read_csv(csvPath)
    requiredColumns = {"article", "question", "A", "B", "C", "D", "answer"}
    missingColumns = requiredColumns.difference(dataFrame.columns)
    if missingColumns:
        raise ValueError(f"Missing required columns: {sorted(missingColumns)}")
    return addCorrectAnswerText(dataFrame)


def addCorrectAnswerText(dataFrame):
    """Add a correctAnswerText column by resolving the answer label."""
    dataFrame = dataFrame.copy()

    def getAnswer(row):
        answerLabel = str(row["answer"]).strip()
        return str(row[answerLabel]) if answerLabel in OPTION_LABELS else ""

    dataFrame["correctAnswerText"] = dataFrame.apply(getAnswer, axis=1)
    return dataFrame


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filterWhQuestions(dataFrame):
    """Keep only rows whose question starts with a Wh-word."""
    whWords = ("who", "what", "where", "when", "why", "how")
    questionText = dataFrame["question"].astype(str).str.lower()
    return dataFrame[questionText.str.startswith(whWords)].copy()


# ---------------------------------------------------------------------------
# Row preparation for T5 baseline
# ---------------------------------------------------------------------------

def prepareT5Rows(dataFrame, sampleSize=20000, randomState=42):
    """
    Build input/target rows for fine-tuning the T5 neural baseline.

    Returns a DataFrame with columns:
        t5Input  — 'generate question: context: <article> answer: <answer>'
        t5Target — the cleaned question text
    """
    filteredData = filterWhQuestions(dataFrame)
    sampleSize = min(sampleSize, len(filteredData))
    sampledData = filteredData.sample(
        n=sampleSize, random_state=randomState
    ).reset_index(drop=True)

    sampledData["articleClean"] = fastCleanSeries(sampledData["article"])
    sampledData["questionClean"] = fastCleanSeries(sampledData["question"])
    sampledData["answerClean"] = fastCleanSeries(sampledData["correctAnswerText"])

    sampledData["t5Input"] = (
        "generate question: context: "
        + sampledData["articleClean"]
        + " answer: "
        + sampledData["answerClean"]
    )
    sampledData["t5Target"] = sampledData["questionClean"]
    return sampledData[["t5Input", "t5Target"]]


# ---------------------------------------------------------------------------
# Option explosion (verifier training)
# ---------------------------------------------------------------------------

def explodeOptions(dataFrame):
    """
    Expand each row into 4 option rows with a binary label.

    Returns a DataFrame with columns:
        cleanArticle, cleanQuestion, cleanOption, label
    """
    parts = []
    for optionLabel in OPTION_LABELS:
        tempData = pd.DataFrame({
            "cleanArticle":  fastCleanSeries(dataFrame["article"]),
            "cleanQuestion": fastCleanSeries(dataFrame["question"]),
            "cleanOption":   fastCleanSeries(dataFrame[optionLabel]),
            "label":         (dataFrame["answer"] == optionLabel).astype(int).values,
        })
        parts.append(tempData)
    return pd.concat(parts).sort_index().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def makeTrainValTestSplit(dataFrame, randomState=42):
    """80 / 10 / 10 stratified split. Returns (train, val, test)."""
    trainData, tempData = train_test_split(
        dataFrame, test_size=0.2, random_state=randomState
    )
    valData, testData = train_test_split(
        tempData, test_size=0.5, random_state=randomState
    )
    return trainData, valData, testData
