"""
state.py
--------
All session-state initialisation and mutating actions for the RACE Quiz UI.
No Streamlit rendering lives here — only data and side-effects.
"""

from pathlib import Path
import sys
import time

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import loadPipeline, loadRandomRaceArticle  # noqa: E402
from ui.components.sample_data import SAMPLE_ARTICLES           # noqa: E402


# ---------------------------------------------------------------------------
# Cached pipeline loader (loads models once per server process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def getPipeline():
    return loadPipeline()


# ---------------------------------------------------------------------------
# RACE lazy loader (load a random chunk, not the full 87k rows)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _loadRaceChunk(chunkSize=500, randomState=42):
    """
    Load a random sample of RACE articles for the dropdown.
    Lazy — only executed once per session; never loads the full dataset.
    """
    import pandas as pd
    import random

    csvPath = PROJECT_ROOT / "data" / "raw" / "train.csv"
    if not csvPath.exists():
        return {}

    df = pd.read_csv(csvPath, usecols=["article"])
    sample = df.sample(n=min(chunkSize, len(df)), random_state=randomState)

    articles = {}
    for i, row in enumerate(sample.itertuples(), start=1):
        text = str(row.article).strip()
        # Generate a friendly name from the first meaningful sentence
        firstSentence = text.split(".")[0].strip()
        friendlyName = firstSentence[:55] + "…" if len(firstSentence) > 55 else firstSentence
        if not friendlyName or len(friendlyName) < 5:
            friendlyName = f"RACE Article #{i}"
        articles[friendlyName] = text

    return articles


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def initSessionState():
    """Initialise all session-state keys with safe defaults."""
    st.session_state.setdefault("articleText",    SAMPLE_ARTICLES["Rosetta Stone"])
    st.session_state.setdefault("quizResult",     None)
    st.session_state.setdefault("selectedLabel",  None)
    st.session_state.setdefault("answerCheck",    None)
    st.session_state.setdefault("usedHints",      0)
    st.session_state.setdefault("revealAnswer",   False)
    st.session_state.setdefault("history",        [])
    st.session_state.setdefault("lastError",      None)
    st.session_state.setdefault("currentScreen",  "input")   # input | quiz | hints | dashboard
    st.session_state.setdefault("raceChunk",      None)      # lazy-loaded RACE sample


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------

def goTo(screen: str):
    """Switch the current screen without a page reload."""
    st.session_state.currentScreen = screen


# ---------------------------------------------------------------------------
# Article loading
# ---------------------------------------------------------------------------

def loadSampleArticle(sampleName: str):
    st.session_state.articleText = SAMPLE_ARTICLES[sampleName]
    st.session_state.lastError   = None
    _resetQuizState()


def loadRaceArticle(friendlyName: str):
    """Load a specific article from the lazy-loaded RACE chunk."""
    if st.session_state.raceChunk and friendlyName in st.session_state.raceChunk:
        st.session_state.articleText = st.session_state.raceChunk[friendlyName]
        st.session_state.lastError   = None
        _resetQuizState()


def loadRandomRace():
    """Load a random single article directly from the CSV."""
    st.session_state.articleText = loadRandomRaceArticle()
    st.session_state.lastError   = None
    _resetQuizState()


def ensureRaceChunk():
    """Populate the RACE chunk on first request (lazy)."""
    if st.session_state.raceChunk is None:
        with st.spinner("Loading RACE sample…"):
            st.session_state.raceChunk = _loadRaceChunk()


def _resetQuizState():
    st.session_state.quizResult    = None
    st.session_state.selectedLabel = None
    st.session_state.answerCheck   = None
    st.session_state.usedHints     = 0
    st.session_state.revealAnswer  = False


# ---------------------------------------------------------------------------
# Quiz generation
# ---------------------------------------------------------------------------

def generateQuiz(articleText: str) -> bool:
    """
    Run the full pipeline on articleText.
    Returns True on success, False on error.
    Stores result in session state and appends to history.
    """
    cleanedText = str(articleText).strip()
    if not cleanedText:
        st.session_state.lastError = "Please paste or load a passage before generating a quiz."
        return False

    startTime = time.perf_counter()
    try:
        pipeline    = getPipeline()
        quizResult  = pipeline.generateQuiz(cleanedText)
    except Exception as exc:
        st.session_state.lastError = f"Model error — {type(exc).__name__}: {exc}"
        return False

    elapsed = round(time.perf_counter() - startTime, 3)
    quizResult["uiLatencySec"] = elapsed

    st.session_state.articleText   = cleanedText
    st.session_state.quizResult    = quizResult
    st.session_state.selectedLabel = None
    st.session_state.answerCheck   = None
    st.session_state.usedHints     = 0
    st.session_state.revealAnswer  = False
    st.session_state.lastError     = None

    st.session_state.history.append({
        "question":        quizResult["question"],
        "correctAnswer":   f"{quizResult['correctLabel']}) {quizResult['correctAnswer']}",
        "verifierPick":    f"{quizResult['prediction']['label']}) {quizResult['prediction']['answer']}",
        "verifierCorrect": quizResult["prediction"]["answer"].lower() == quizResult["correctAnswer"].lower(),
        "optionQuality":   quizResult.get("optionQuality", "unknown"),
        "latencySec":      quizResult["latencySec"],
        "uiLatencySec":    elapsed,
    })
    return True


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def checkAnswer():
    """Check the currently selected label and store result."""
    quizResult    = st.session_state.quizResult
    selectedLabel = st.session_state.selectedLabel
    if quizResult is None or selectedLabel is None:
        return

    pipeline = getPipeline()
    result   = pipeline.checkAnswer(quizResult, selectedLabel)
    st.session_state.answerCheck = result
