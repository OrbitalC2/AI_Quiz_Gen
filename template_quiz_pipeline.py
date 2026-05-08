import math
import random
import re
import string
import sys
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

# Import shared feature engineering so inference == training exactly
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from features import buildBaseFeatureVector, buildClusterFeaturesIfAvailable  # noqa: E402


DEFAULT_ARCHIVE_DIR = Path.home() / "Desktop" / "race_rc_project_model_archives"
DEFAULT_ZIP_PATH = DEFAULT_ARCHIVE_DIR / "Model_A_Submission_Final.zip"
DEFAULT_MODEL_B_ZIP_PATH = DEFAULT_ARCHIVE_DIR / "Model_B_Assets.zip"
DEFAULT_QUESTION_RANKER_ZIP_PATH = DEFAULT_ARCHIVE_DIR / "My_Trained_Ranker.zip"
DEFAULT_MODEL_DIR = Path("models") / "model_a" / "traditional"
DEFAULT_MODEL_B_DIR = Path("models") / "model_b" / "traditional"
DEFAULT_QUESTION_RANKER_MODEL_DIR = Path("models") / "model_a" / "traditional"
DEFAULT_RACE_TRAIN_PATH = Path("data") / "raw" / "train.csv"
DEFAULT_DISTRACTOR_INDEX_PATH = DEFAULT_MODEL_B_DIR / "raceDistractorIndex.joblib"
DISTRACTOR_INDEX_VERSION = 6
OPTION_LABELS = ["A", "B", "C", "D"]
BAD_PROPER_NOUN_STARTS = {
    "in",
    "on",
    "at",
    "during",
    "because",
    "for",
    "eventually",
    "this",
    "that",
    "these",
    "those",
    "it",
}
BAD_QUESTION_TRAILING_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "because",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "than",
    "that",
    "the",
    "to",
    "with",
}
BAD_QUESTION_SECOND_WORDS = BAD_QUESTION_TRAILING_WORDS | {
    "now",
    "then",
    "there",
}
QUESTION_VERB_WORDS = {
    "am",
    "are",
    "be",
    "became",
    "become",
    "been",
    "being",
    "called",
    "can",
    "contains",
    "could",
    "decoded",
    "did",
    "discovered",
    "does",
    "do",
    "found",
    "generated",
    "generates",
    "had",
    "has",
    "have",
    "included",
    "is",
    "issued",
    "known",
    "launched",
    "lies",
    "may",
    "might",
    "must",
    "orbits",
    "provided",
    "should",
    "study",
    "took",
    "use",
    "used",
    "uses",
    "was",
    "were",
    "will",
    "would",
}

@dataclass
class AnswerCandidate:
    text: str
    sentence: str
    kind: str
    score: float


@dataclass
class DistractorRecord:
    text: str
    source_question: str
    source_correct_answer: str
    source_article_snippet: str
    option_word_count: int
    answer_category: str
    option_category: str
    tokens: tuple


def cleanText(text):
    text = str(text).lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def textTokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def normaliseOptionKey(text):
    return " ".join(textTokens(text))


def containsTokenPhrase(containerText, phraseText):
    phraseTokens = textTokens(phraseText)
    if not phraseTokens:
        return False

    containerTokens = textTokens(containerText)
    if len(phraseTokens) == 1:
        return phraseTokens[0] in set(containerTokens)

    windowSize = len(phraseTokens)
    for startIndex in range(0, len(containerTokens) - windowSize + 1):
        if containerTokens[startIndex:startIndex + windowSize] == phraseTokens:
            return True
    return False


def hasCorrectAnswerLeak(optionText, correctAnswer):
    correctTokens = textTokens(correctAnswer)
    optionTokens = textTokens(optionText)
    if not correctTokens or not optionTokens:
        return False

    if len(correctTokens) == 1:
        return correctTokens[0] in set(optionTokens)

    return containsTokenPhrase(optionText, correctAnswer)


def optionLooksUsable(optionText, correctAnswer=""):
    text = re.sub(r"\s+", " ", str(optionText)).strip()
    key = normaliseOptionKey(text)
    if not key or key in {"nan", "none", "null"}:
        return False
    if correctAnswer and hasCorrectAnswerLeak(text, correctAnswer):
        return False
    if len(text) > 220:
        return False
    if len(textTokens(text)) > 32:
        return False
    if re.fullmatch(r"[_\W]+", text):
        return False
    return True


def editDistance(leftText, rightText):
    leftText = str(leftText)
    rightText = str(rightText)
    previousRow = list(range(len(rightText) + 1))

    for leftIndex, leftChar in enumerate(leftText, start=1):
        currentRow = [leftIndex]
        for rightIndex, rightChar in enumerate(rightText, start=1):
            insertCost = currentRow[rightIndex - 1] + 1
            deleteCost = previousRow[rightIndex] + 1
            replaceCost = previousRow[rightIndex - 1] + (leftChar != rightChar)
            currentRow.append(min(insertCost, deleteCost, replaceCost))
        previousRow = currentRow

    return previousRow[-1]


def splitSentences(articleText):
    sentences = re.split(r"(?<=[.!?])\s+", str(articleText).strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalizeCandidate(text):
    text = re.sub(r"\s+", " ", str(text)).strip(" ,.;:()[]")
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def addCandidate(candidates, seenTexts, text, sentence, kind, baseScore):
    text = normalizeCandidate(text)
    if len(text) < 3:
        return

    firstWord = text.split()[0].lower().strip("'s")
    if kind == "properNoun" and firstWord in BAD_PROPER_NOUN_STARTS:
        return
    if kind == "properNoun" and "'s " in text:
        return

    key = text.lower()
    if key in seenTexts:
        return

    seenTexts.add(key)
    wordCount = len(text.split())
    score = baseScore + min(wordCount, 4) * 0.4
    if kind == "properNoun" and re.search(rf"\bnamed\s+{re.escape(text)}\b", sentence):
        score += 1.5
    if kind == "properNoun" and "-" in text:
        score += 0.8
    candidates.append(AnswerCandidate(text=text, sentence=sentence, kind=kind, score=score))


def findAnswerCandidates(articleText):
    candidates = []
    seenTexts = set()

    datePattern = r"\b(?:\d{4}|\d+\s*BC|[A-Z][a-z]+\s+\d{4})\b"
    acronymPattern = r"\b[A-Z]{2,}\b"
    possessiveNamePattern = r"\b([A-Z][a-z]{3,})'s\b"
    properNounPattern = (
        r"\b(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:'s)?\s+)"
        r"{1,5}[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:'s)?\b"
    )
    # Extracts the LABEL in 'known as / called / referred to as LABEL'
    # e.g. "famously known as the powerhouse of the cell"
    definitionPattern = (
        r"\b(?:known as|called|referred to as|named after)\s+"
        r"((?:the\s+|a\s+|an\s+)?[a-zA-Z][a-zA-Z\s'\-]{3,50}?)(?=\s*(?:[,.]|because|since|which|that|$))"
    )
    # Extracts location phrases: 'in/at/near the city of Rosetta', 'at Cambridge'
    locationPattern = (
        r"\b(?:in|at|near|from|within)\s+"
        r"((?:the\s+)?[A-Z][a-zA-Z]+(?:\s+of\s+[A-Z][a-zA-Z]+|(?:\s+[A-Z][a-zA-Z]+){0,3}))"
    )

    for sentenceIndex, sentence in enumerate(splitSentences(articleText)):
        positionBonus = max(0.0, 1.0 - sentenceIndex * 0.1)

        for match in re.finditer(datePattern, sentence):
            addCandidate(candidates, seenTexts, match.group(), sentence, "date", 4.2 + positionBonus)

        for match in re.finditer(acronymPattern, sentence):
            addCandidate(candidates, seenTexts, match.group(), sentence, "acronym", 3.8 + positionBonus)

        for match in re.finditer(possessiveNamePattern, sentence):
            addCandidate(candidates, seenTexts, match.group(1), sentence, "person", 3.6 + positionBonus)

        for match in re.finditer(properNounPattern, sentence):
            addCandidate(candidates, seenTexts, match.group(), sentence, "properNoun", 4.8 + positionBonus)

        for match in re.finditer(definitionPattern, sentence, re.IGNORECASE):
            phrase = match.group(1).strip(" ,;")
            if len(phrase.split()) >= 2:          # skip single-word labels
                addCandidate(candidates, seenTexts, phrase, sentence, "definition", 5.2 + positionBonus)

        for match in re.finditer(locationPattern, sentence):
            phrase = match.group(1).strip()
            if phrase.lower() in {"the", "a", "an"} or len(phrase) <= 3:
                continue
            words = phrase.split()
            # Accept multi-word phrases (e.g. 'the city of Rosetta', 'New York')
            # or single words that are clearly geographic (contain a geo indicator)
            _GEO_INDICATORS = {
                "city", "town", "village", "country", "island", "river",
                "street", "avenue", "road", "university", "college",
                "museum", "park", "ocean", "sea", "lake", "mountain",
            }
            isMultiWord = len(words) >= 2
            hasGeoIndicator = any(w.lower() in _GEO_INDICATORS for w in words)
            if isMultiWord or hasGeoIndicator:
                addCandidate(candidates, seenTexts, phrase, sentence, "location", 4.5 + positionBonus)


    if not candidates:
        for sentence in splitSentences(articleText):
            words = re.findall(r"\b[a-zA-Z]{7,}\b", sentence)
            for word in words:
                addCandidate(candidates, seenTexts, word, sentence, "keyword", 2.0)

    for candidate in candidates:
        if candidate.kind == "properNoun" and looksLikePerson(candidate):
            candidate.kind = "person"

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def looksLikePerson(candidate):
    if candidate.kind == "person":
        return True
    personClues = ["named", "soldier", "scholar", "king", "queen", "scientist", "inventor"]
    sentence = candidate.sentence.lower()
    if any(clue in sentence for clue in personClues):
        return True
    return bool(re.search(r"\b[A-Z][a-z]+-[A-Z][a-z]+\b", candidate.text))


def cleanQuestionEnd(text):
    text = re.sub(r"\s+", " ", str(text)).strip(" ,.;:")
    return text


def lowerFirstCharacter(text):
    text = str(text)
    if not text:
        return text
    if len(text) > 1 and text[0].isupper() and not text[1].islower():
        return text
    return text[0].lower() + text[1:]


def tokeniseWords(text):
    return re.findall(r"[A-Za-z']+", str(text).lower())


def hasVerbLikeToken(text):
    tokens = tokeniseWords(text)
    if any(token in QUESTION_VERB_WORDS for token in tokens):
        return True
    return any(token.endswith(("ed", "ing", "es")) for token in tokens if len(token) > 4)


def isReadableQuestion(question):
    """
    Lightweight quality gate for generated Wh-questions.

    This catches malformed fragments such as "What from now?" while keeping
    the generator fully rule-based and dependency-free.
    """
    question = cleanQuestionEnd(str(question).rstrip("?")) + "?"
    words = tokeniseWords(question)
    if len(words) < 5:
        return False
    if len(words) > 1 and words[1] in BAD_QUESTION_SECOND_WORDS:
        return False
    if words[-1] in BAD_QUESTION_TRAILING_WORDS:
        return False
    if not hasVerbLikeToken(" ".join(words[1:])):
        return False
    if re.search(r"\b(what|who|where|when|why|how)\s+(from|of|in|on|to|for)\b", question, re.IGNORECASE):
        return False
    return True


def buildClozeQuestion(candidate):
    sentence = cleanQuestionEnd(candidate.sentence)
    maskedSentence = re.sub(
        re.escape(candidate.text),
        "_____",
        sentence,
        count=1,
        flags=re.IGNORECASE,
    )
    return f"Which answer best completes this sentence: {maskedSentence}?"


def buildDateQuestion(candidate):
    return "When did the event described in the passage take place?"


def buildSubjectQuestion(candidate):
    sentence = cleanQuestionEnd(candidate.sentence)
    answerStart = sentence.lower().find(candidate.text.lower())

    if answerStart == -1:
        return buildClozeQuestion(candidate)

    afterAnswer = sentence[answerStart + len(candidate.text):].strip(" ,")

    if afterAnswer:
        whWord = "Who" if looksLikePerson(candidate) else "What"
        question = f"{whWord} {lowerFirstCharacter(cleanQuestionEnd(afterAnswer))}?"
        if isReadableQuestion(question):
            return question

    return buildClozeQuestion(candidate)


def buildTemplateQuestion(candidate):
    if candidate.kind == "date":
        return buildDateQuestion(candidate)
    return buildSubjectQuestion(candidate)


def buildHint(candidate):
    """
    Level-3 (specific) hint: answer sentence with the answer masked.
    Kept for backward compatibility.
    """
    maskedSentence = re.sub(
        re.escape(candidate.text), "__________", candidate.sentence, flags=re.IGNORECASE
    )
    return cleanQuestionEnd(maskedSentence)


def _findSentenceIndex(targetSentence, sentences):
    """
    Return the index of the sentence in sentences that best matches targetSentence.
    Falls back to 0 if no match is found.
    """
    targetLower = targetSentence.lower().strip()
    for index, sentence in enumerate(sentences):
        if sentence.lower().strip() == targetLower:
            return index
    # Fuzzy fallback: find the sentence with the most word overlap
    targetWords = set(targetLower.split())
    bestIndex, bestOverlap = 0, -1
    for index, sentence in enumerate(sentences):
        overlap = len(targetWords & set(sentence.lower().split()))
        if overlap > bestOverlap:
            bestOverlap = overlap
            bestIndex = index
    return bestIndex


def buildGraduatedHints(candidate, articleText):
    """
    Build three graduated hints moving from general to specific.

    Rule-based approach (no ML model required — spec allows 'rule-based overlap').

    Hint 1 — General
        The opening sentence(s) of the article, giving broad topic context
        without revealing where in the text the answer lives.

    Hint 2 — Intermediate
        The sentence immediately preceding the answer sentence, narrowing the
        reader's attention to the relevant part of the passage.

    Hint 3 — Specific
        The answer sentence itself with the answer replaced by __________,
        making it possible to fill in the blank with the right word.

    Returns
    -------
    list of three strings: [hint1, hint2, hint3]
    """
    sentences = splitSentences(str(articleText))

    # --- Hint 3: masked answer sentence (most specific) ---
    hint3 = buildHint(candidate)

    if not sentences:
        return [hint3, hint3, hint3]

    # --- Hint 1: first sentence that does NOT contain the answer (most general) ---
    answerLower = candidate.text.lower().strip()
    hint1 = None
    for sentence in sentences:
        # Skip any sentence that gives away the answer
        if answerLower and answerLower in sentence.lower():
            continue
        cleaned = cleanQuestionEnd(sentence)
        if len(cleaned) >= 20:
            hint1 = cleaned
            break
    # Fallback: use first sentence but mask the answer
    if hint1 is None:
        raw = cleanQuestionEnd(sentences[0]) if sentences else ""
        hint1 = re.sub(re.escape(candidate.text), "__________", raw, flags=re.IGNORECASE)

    # --- Hint 2: sentence before the answer sentence ---
    answerIdx = _findSentenceIndex(candidate.sentence, sentences)
    if answerIdx > 1:
        hint2 = cleanQuestionEnd(sentences[answerIdx - 1])
    elif answerIdx == 1:
        # Answer is in the 2nd sentence; use part of the first as hint 2
        hint2 = hint1
    else:
        # Answer is in the very first sentence; escalate to next sentence
        hint2 = cleanQuestionEnd(sentences[1]) if len(sentences) > 1 else hint3

    # Deduplicate: if hint2 == hint1, advance one sentence forward
    if hint2.lower() == hint1.lower() and len(sentences) > answerIdx + 1:
        hint2 = cleanQuestionEnd(sentences[min(answerIdx + 1, len(sentences) - 1)])

    return [hint1, hint2, hint3]


def buildAnswerOptions(correctCandidate, allCandidates):
    sameKind = [
        candidate.text for candidate in allCandidates
        if isCompatibleDistractor(correctCandidate, candidate)
    ]

    options = [correctCandidate.text]
    for option in sameKind:
        if normaliseOptionKey(option) not in {normaliseOptionKey(item) for item in options}:
            options.append(option)
        if len(options) == 4:
            break

    return options[:4]


def shuffleOptions(options, articleText, correctAnswer):
    seedText = f"{articleText[:500]}::{correctAnswer}"
    seed = sum(ord(char) for char in seedText)
    shuffledOptions = list(options)
    random.Random(seed).shuffle(shuffledOptions)
    correctIndex = next(
        index for index, option in enumerate(shuffledOptions)
        if option.lower() == correctAnswer.lower()
    )
    return shuffledOptions, OPTION_LABELS[correctIndex]


def inferOptionCategory(text, context="", questionText=""):
    text = str(text).strip()
    lowerText = text.lower()
    combined = f"{lowerText} {str(context).lower()} {str(questionText).lower()}"
    tokens = textTokens(text)

    degreeCues = (
        "degree",
        "mba",
        "phd",
        "ph.d",
        "bachelor",
        "master",
        "diploma",
    )
    if any(cue in lowerText for cue in degreeCues):
        return "educationDegree"

    if re.fullmatch(r"(?:\d{4}|\d+\s*bc|[a-z]+\s+\d{4})", lowerText):
        return "date"
    if re.fullmatch(r"\d[\d\s,.\-/]*", lowerText):
        return "number"

    questionStart = textTokens(questionText)[:1]
    if questionStart == ["when"]:
        return "date"
    if questionStart == ["who"]:
        return "person"
    if questionStart == ["where"]:
        return "location"

    if re.fullmatch(r"[A-Z]{2,}", text):
        return "acronym"
    if re.fullmatch(r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?){0,4}", text):
        return "properNoun"
    if len(tokens) >= 6:
        return "statement"
    return "term"


def getAnswerCategory(candidate, questionText=""):
    if candidate.kind == "date":
        return "date"
    if looksLikePerson(candidate):
        return "person"
    if candidate.kind == "location":
        return "location"
    inferred = inferOptionCategory(candidate.text, candidate.sentence, questionText)
    if candidate.kind == "acronym" and inferred == "term":
        return "acronym"
    if candidate.kind == "definition" and inferred == "term":
        return "definition"
    if candidate.kind == "properNoun" and inferred == "term":
        return "properNoun"
    return inferred


def isCompatibleDistractor(correctCandidate, distractorCandidate, questionText="", relaxed=False):
    if normaliseOptionKey(distractorCandidate.text) == normaliseOptionKey(correctCandidate.text):
        return False
    if hasCorrectAnswerLeak(distractorCandidate.text, correctCandidate.text):
        return False

    correctCategory = getAnswerCategory(correctCandidate, questionText)
    distractorCategory = getAnswerCategory(distractorCandidate, questionText)

    if correctCategory == distractorCategory:
        return True

    if not relaxed:
        return False

    if correctCategory in {"term", "definition", "statement"}:
        return distractorCategory in {"term", "definition", "statement"}
    if correctCategory == "properNoun":
        return distractorCategory in {"properNoun", "term"}
    if correctCategory == "educationDegree":
        return distractorCategory in {"educationDegree", "term", "statement"}
    return False


def categoryCompatibilityScore(correctCategory, candidateCategory, relaxed=False):
    if correctCategory == candidateCategory:
        return 1.0
    if relaxed and correctCategory in {"term", "definition", "statement"} and candidateCategory in {"term", "definition", "statement"}:
        return 0.65
    if relaxed and correctCategory == "properNoun" and candidateCategory == "term":
        return 0.55
    if relaxed and correctCategory == "educationDegree" and candidateCategory in {"term", "statement"}:
        return 0.45
    return 0.0


def lengthCompatibilityScore(correctAnswer, candidateText):
    correctWords = max(len(textTokens(correctAnswer)), 1)
    candidateWords = max(len(textTokens(candidateText)), 1)
    distance = abs(math.log1p(correctWords) - math.log1p(candidateWords))
    return max(0.0, 1.0 - distance / 2.4)


def ensureZipFiles(zipPath, modelDir, neededFiles):
    modelDir = Path(modelDir)
    neededPaths = [modelDir / fileName for fileName in neededFiles]

    if all(path.exists() for path in neededPaths):
        return modelDir

    if not Path(zipPath).exists():
        raise FileNotFoundError(f"Could not find {zipPath}")

    modelDir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zipPath, "r") as zipFile:
        zipFile.extractall(modelDir)

    return modelDir


def ensureModelFiles(zipPath=DEFAULT_ZIP_PATH, modelDir=DEFAULT_MODEL_DIR):
    """
    Extract Model A artifacts from zip if they are not already on disk.
    Only the LR model is strictly required; SVM/RF/KMeans are optional
    (falls back to LR-only when they are absent).
    """
    return ensureZipFiles(
        zipPath=zipPath,
        modelDir=modelDir,
        neededFiles=["lrVerifierModel.joblib"],
    )


def ensureModelBFiles(zipPath=DEFAULT_MODEL_B_ZIP_PATH, modelDir=DEFAULT_MODEL_B_DIR):
    return ensureZipFiles(
        zipPath=zipPath,
        modelDir=modelDir,
        neededFiles=["rfDistractorRanker.joblib"],
    )


def ensureQuestionRankerFiles(
    zipPath=DEFAULT_QUESTION_RANKER_ZIP_PATH,
    modelDir=DEFAULT_QUESTION_RANKER_MODEL_DIR,
):
    """Extract rfQuestionRanker.joblib from the ranker zip if not already present."""
    return ensureZipFiles(
        zipPath=zipPath,
        modelDir=modelDir,
        neededFiles=["rfQuestionRanker.joblib"],
    )


class EnsembleAnswerPredictor:
    """
    Pure traditional-ML answer verifier.

    Loads up to four trained classifiers (LR, Calibrated SVM, Naive Bayes,
    Random Forest) and K-Means cluster artifacts, then produces soft-vote
    probability scores for each answer option.

    Feature vector (12 dimensions) — defined in features.py
    ----------------------------------------------------------
    0  tfidf_cosine_sim        document-level TF-IDF cosine sim
    1  unigram_overlap_ratio   (q∪opt tokens) ∩ article / (q∪opt)
    2  idf_weighted_score      mean IDF of shared option–article tokens
    3  best_sentence_overlap   max unigram overlap over all article sentences
    4  bigram_overlap_ratio    bigram version of feature 1
    5  exact_match_flag        1 if option text appears verbatim in article
    6  option_token_length     token count
    7  option_char_length      character count
    8  first_position          normalised first-occurrence position
    9  is_number               1 if option is numeric / date-like
   10  kmeans_cluster_label    (0.0 when K-Means not available)
   11  kmeans_centroid_dist    (0.0 when K-Means not available)

    Backward compatibility
    ----------------------
    Falls back to LR-only scoring when SVM / RF / NB / KMeans artifacts
    are absent, so old checkpoints continue to work without retraining.
    """

    def __init__(self, modelDir=DEFAULT_MODEL_DIR, tieTolerance=0.003):
        self.modelDir = Path(modelDir)
        self.tieTolerance = tieTolerance
        self.embedder = None  # kept for external callers

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lrModel  = joblib.load(self.modelDir / "lrVerifierModel.joblib")
            self.svmModel = self._tryLoad("svmVerifierModel.joblib")
            self.nbModel  = self._tryLoad("nbVerifierModel.joblib")
            self.rfModel  = self._tryLoad("rfVerifierModel.joblib")
            self.tfidfVectorizer = self._tryLoad("tfidfVectorizer.joblib")
            self.idfDict  = self._tryLoad("idfDictionary.joblib") or {}
            self.kmeansModel = self._tryLoad("kmeansModel.joblib")
            self.svdModel    = self._tryLoad("svdModel.joblib")

        loadedModels = "LR" + ("+SVM" if self.svmModel else "") + ("+NB" if self.nbModel else "") + ("+RF" if self.rfModel else "")
        hasKmeans = self.kmeansModel is not None and self.svdModel is not None
        print(f"[EnsembleAnswerPredictor] {loadedModels}  kmeans={hasKmeans}  dir={self.modelDir}")


    @classmethod
    def fromZip(cls, zipPath=DEFAULT_ZIP_PATH, modelDir=DEFAULT_MODEL_DIR):
        extractedDir = ensureModelFiles(zipPath, modelDir)
        return cls(extractedDir)

    def _tryLoad(self, filename):
        """Load a joblib artifact; return None silently if file is absent."""
        path = self.modelDir / filename
        if not path.exists():
            return None
        return joblib.load(path)

    def getOptionFeatures(self, articleText, questionText, optionText):
        """
        Build the 12-dimensional feature vector for one (article, question, option) triple.
        Delegates to features.py to guarantee identity with the training features.
        """
        cArticle  = cleanText(articleText)
        cQuestion = cleanText(questionText)
        cOption   = cleanText(optionText)

        # BASE features 0-9 from features.py
        baseFeats = buildBaseFeatureVector(cArticle, cQuestion, cOption, self.idfDict, self.tfidfVectorizer)

        # CLUSTER features 10-11
        if self.tfidfVectorizer is not None:
            optionVec = self.tfidfVectorizer.transform([cOption])
            clusterFeats = buildClusterFeaturesIfAvailable(optionVec, self.kmeansModel, self.svdModel)
            clusterLabel = float(clusterFeats[0, 0])
            centroidDist = float(clusterFeats[0, 1])
        else:
            clusterLabel, centroidDist = 0.0, 0.0

        return baseFeats + [clusterLabel, centroidDist]

    # ------------------------------------------------------------------
    # Scoring and prediction
    # ------------------------------------------------------------------

    def _softVoteProba(self, features):
        """
        Soft-vote ensemble: mean predict_proba[:, 1] over LR + NB + RF.
        SVM is excluded — see EnsembleAnswerPredictor docstring.
        Gracefully skips any model that failed to load.
        """
        featArray = np.array(features, dtype=np.float32)
        # Ensemble: LR + SVM + RF  (NB excluded — low recall degrades MCQ ranking)
        probsList = [self.lrModel.predict_proba(featArray)[:, 1]]
        for model in (self.svmModel, self.rfModel):
            if model is not None:
                probsList.append(model.predict_proba(featArray)[:, 1])
        return np.mean(probsList, axis=0)

    def scoreOptions(self, articleText, questionText, options):
        """
        Return a dict mapping option labels (A-D) to soft-vote probability scores.
        """
        features = [
            self.getOptionFeatures(articleText, questionText, optionText)
            for optionText in options
        ]
        probabilities = self._softVoteProba(features)
        return {label: float(score) for label, score in zip(OPTION_LABELS, probabilities)}

    def predictAnswer(self, articleText, questionText, options, sourceSentence=""):
        """
        Predict the correct answer label and text for a given quiz item.

        Returns
        -------
        (bestLabel, bestOptionText, scores_dict)
        """
        scores = self.scoreOptions(articleText, questionText, options)
        bestLabel = self._chooseBestLabel(scores, options, articleText, sourceSentence)
        bestIndex = OPTION_LABELS.index(bestLabel)
        return bestLabel, options[bestIndex], scores

    def _chooseBestLabel(self, scores, options, articleText, sourceSentence=""):
        """Break ties using source-sentence presence, article position, and token length."""
        bestScore = max(scores.values())
        tiedLabels = [
            label for label, score in scores.items()
            if abs(score - bestScore) <= self.tieTolerance
        ]

        if len(tiedLabels) == 1:
            return tiedLabels[0]

        articleLower = articleText.lower()
        sourceLower = sourceSentence.lower()

        def getTieBreakValue(label):
            option = options[OPTION_LABELS.index(label)]
            optionLower = option.lower()
            firstPosition = articleLower.find(optionLower)
            if firstPosition == -1:
                firstPosition = 10 ** 9
            return (
                optionLower in sourceLower,
                -firstPosition,
                -abs(len(option.split()) - 2),
                -OPTION_LABELS.index(label),
            )

        return max(tiedLabels, key=getTieBreakValue)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# External code that imports LogisticAnswerPredictor by name continues to work.
LogisticAnswerPredictor = EnsembleAnswerPredictor


class DistractorRetriever:
    """
    Corpus-backed distractor candidate source built from human-written RACE
    options.  It replaces the old tiny manual fallback lists with a reusable
    local retrieval index.
    """

    def __init__(
        self,
        csvPath=DEFAULT_RACE_TRAIN_PATH,
        cachePath=DEFAULT_DISTRACTOR_INDEX_PATH,
        vectorizerPath=DEFAULT_MODEL_DIR / "tfidfVectorizer.joblib",
    ):
        self.csvPath = Path(csvPath)
        self.cachePath = Path(cachePath)
        self.vectorizerPath = Path(vectorizerPath)
        self.vectorizer = None
        self.matrix = None
        self.records = []
        self.categoryIndexes = {}
        self.loaded = False

    def retrieve(self, articleText, questionText, correctCandidate, limit=120):
        self._ensureLoaded()
        if self.matrix is None or not self.records:
            return []

        queryText = cleanText(
            " ".join([
                str(questionText),
                str(correctCandidate.text),
                str(correctCandidate.sentence),
            ])
        )
        queryVec = self.vectorizer.transform([queryText])
        correctCategory = getAnswerCategory(correctCandidate, questionText)
        preferredIndexes = self._candidateIndexesForCategory(correctCategory)
        topIndexes, scoreLookup = self._topIndexesForQuery(queryVec, preferredIndexes, max(limit * 3, limit))
        if len(topIndexes) < limit:
            globalIndexes, globalScores = self._topIndexesForQuery(queryVec, None, limit)
            scoreLookup.update(globalScores)
            topIndexes.extend(index for index in globalIndexes if index not in scoreLookup or index not in topIndexes)

        topIndexes = sorted(set(topIndexes), key=lambda index: scoreLookup.get(index, 0.0), reverse=True)
        retrieved = []
        seen = set()
        for index in topIndexes:
            record = self.records[int(index)]
            key = normaliseOptionKey(record.text)
            if key in seen:
                continue
            seen.add(key)
            retrieved.append({
                "text": record.text,
                "source": "race",
                "retrievalScore": float(scoreLookup.get(index, 0.0)),
                "sourceQuestion": record.source_question,
                "sourceCorrectAnswer": record.source_correct_answer,
                "sourceArticleSnippet": record.source_article_snippet,
                "optionCategory": record.option_category,
                "answerCategory": record.answer_category,
            })
            if len(retrieved) >= limit:
                break
        return retrieved

    def _candidateIndexesForCategory(self, correctCategory):
        exact = list(self.categoryIndexes.get(correctCategory, []))
        if correctCategory == "educationDegree":
            return exact
        if correctCategory in {"term", "definition", "statement"}:
            indexes = []
            for category in ("term", "definition", "statement"):
                indexes.extend(self.categoryIndexes.get(category, []))
            return indexes
        if correctCategory == "properNoun":
            indexes = list(exact)
            indexes.extend(self.categoryIndexes.get("term", []))
            return indexes
        return exact

    def _topIndexesForQuery(self, queryVec, indexes, limit):
        if indexes:
            rowIndexes = np.array(indexes, dtype=np.int64)
            matrix = self.matrix[rowIndexes]
            scores = np.asarray(matrix.dot(queryVec.T).todense()).ravel()
            if scores.size == 0:
                return [], {}
            candidateCount = min(limit, scores.size)
            localIndexes = np.argpartition(scores, -candidateCount)[-candidateCount:]
            globalIndexes = [int(rowIndexes[int(localIndex)]) for localIndex in localIndexes]
            scoreLookup = {
                int(rowIndexes[int(localIndex)]): float(scores[int(localIndex)])
                for localIndex in localIndexes
            }
            return sorted(globalIndexes, key=lambda index: scoreLookup[index], reverse=True), scoreLookup

        scores = np.asarray(self.matrix.dot(queryVec.T).todense()).ravel()
        if scores.size == 0:
            return [], {}
        candidateCount = min(limit, scores.size)
        topIndexes = np.argpartition(scores, -candidateCount)[-candidateCount:]
        scoreLookup = {int(index): float(scores[int(index)]) for index in topIndexes}
        return sorted([int(index) for index in topIndexes], key=lambda index: scoreLookup[index], reverse=True), scoreLookup

    def _ensureLoaded(self):
        if self.loaded:
            return
        if self._loadCache():
            self.loaded = True
            return
        self._buildIndex()
        self.loaded = True

    def _loadCache(self):
        if not self.cachePath.exists():
            return False
        try:
            payload = joblib.load(self.cachePath, mmap_mode="r")
        except Exception:
            return False

        if payload.get("version") != DISTRACTOR_INDEX_VERSION:
            return False
        if payload.get("csvMtime") != self._mtime(self.csvPath):
            return False
        if payload.get("vectorizerMtime") != self._mtime(self.vectorizerPath):
            return False

        self.vectorizer = payload.get("vectorizer")
        self.matrix = payload.get("matrix")
        self.records = payload.get("records", [])
        self.categoryIndexes = payload.get("categoryIndexes", {})
        return self.vectorizer is not None and self.matrix is not None

    def _buildIndex(self):
        if not self.csvPath.exists():
            self.records = []
            self.vectorizer = self._loadVectorizer()
            self.matrix = None
            return

        import pandas as pd

        dataFrame = pd.read_csv(self.csvPath)
        self.vectorizer = self._loadVectorizer()
        records = []
        docs = []

        for _, row in dataFrame.iterrows():
            answerLabel = str(row.get("answer", "")).strip()
            if answerLabel not in OPTION_LABELS:
                continue
            articleText = str(row.get("article", ""))
            questionText = str(row.get("question", ""))
            sourceCorrect = str(row.get(answerLabel, ""))
            answerCategory = inferOptionCategory(sourceCorrect, articleText, questionText)

            for optionLabel in OPTION_LABELS:
                if optionLabel == answerLabel:
                    continue
                optionText = normalizeCandidate(str(row.get(optionLabel, "")))
                if not optionLooksUsable(optionText, sourceCorrect):
                    continue
                optionCategory = inferOptionCategory(optionText, articleText, questionText)
                record = DistractorRecord(
                    text=optionText,
                    source_question=questionText,
                    source_correct_answer=sourceCorrect,
                    source_article_snippet=cleanQuestionEnd(articleText[:500]),
                    option_word_count=len(textTokens(optionText)),
                    answer_category=answerCategory,
                    option_category=optionCategory,
                    tokens=tuple(textTokens(optionText)),
                )
                records.append(record)
                docs.append(cleanText(f"{optionText} {questionText} {sourceCorrect}"))

        if self.vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=60000,
                ngram_range=(1, 2),
                min_df=2,
                stop_words="english",
            )
            self.matrix = self.vectorizer.fit_transform(docs) if docs else None
        else:
            self.matrix = self.vectorizer.transform(docs) if docs else None

        self.records = records
        self.categoryIndexes = self._buildCategoryIndexes(records)
        self._saveCache()

    def _buildCategoryIndexes(self, records):
        categoryIndexes = {}
        for index, record in enumerate(records):
            categoryIndexes.setdefault(record.option_category, []).append(index)
        return categoryIndexes

    def _loadVectorizer(self):
        if not self.vectorizerPath.exists():
            return None
        try:
            return joblib.load(self.vectorizerPath)
        except Exception:
            return None

    def _saveCache(self):
        if self.vectorizer is None or self.matrix is None:
            return
        try:
            self.cachePath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "version": DISTRACTOR_INDEX_VERSION,
                    "csvMtime": self._mtime(self.csvPath),
                    "vectorizerMtime": self._mtime(self.vectorizerPath),
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                    "records": self.records,
                    "categoryIndexes": self.categoryIndexes,
                },
                self.cachePath,
                compress=0,
            )
        except Exception:
            pass

    def _mtime(self, path):
        path = Path(path)
        return path.stat().st_mtime if path.exists() else None


class ModelBDistractorHintGenerator:
    def __init__(self, modelDir=DEFAULT_MODEL_B_DIR, embedder=None, answerPredictor=None, retriever=None):
        self.modelDir = Path(modelDir)
        self.embedder = embedder
        self.answerPredictor = answerPredictor
        self.retriever = retriever or DistractorRetriever(cachePath=self.modelDir / "raceDistractorIndex.joblib")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rankModel = joblib.load(self.modelDir / "rfDistractorRanker.joblib")

    @classmethod
    def fromZip(cls, zipPath=DEFAULT_MODEL_B_ZIP_PATH, modelDir=DEFAULT_MODEL_B_DIR, embedder=None, answerPredictor=None):
        extractedDir = ensureModelBFiles(zipPath, modelDir)
        return cls(extractedDir, embedder, answerPredictor=answerPredictor)

    def warmUpRetriever(self):
        """Load the corpus-backed distractor index before the first quiz request."""
        self.retriever._ensureLoaded()

    def buildOptions(self, articleText, correctCandidate, allCandidates, questionText="", answerPredictor=None):
        return self.buildOptionsWithDiagnostics(
            articleText,
            correctCandidate,
            allCandidates,
            questionText=questionText,
            answerPredictor=answerPredictor,
        )["options"]

    def buildOptionsWithDiagnostics(
        self,
        articleText,
        correctCandidate,
        allCandidates,
        questionText="",
        answerPredictor=None,
    ):
        """
        Build four MCQ options using a staged ladder:
        same-passage strict candidates, strict RACE retrieval, relaxed RACE
        retrieval, then last-resort corpus retrieval.  The last stage still uses
        human-written RACE options instead of manual word lists.
        """
        predictor = answerPredictor or self.answerPredictor
        correctKey = normaliseOptionKey(correctCandidate.text)
        selected = []
        selectedKeys = {correctKey}
        considered = 0
        diagnostics = {
            "optionQuality": "last_resort",
            "stageCounts": {
                "strict": 0,
                "retrieved": 0,
                "relaxed": 0,
                "last_resort": 0,
            },
            "retrievalCandidatesConsidered": 0,
            "fallbackUsed": False,
            "selectedDistractors": [],
        }

        stageEntries = [
            (
                "strict",
                self._samePassageEntries(articleText, correctCandidate, allCandidates, questionText),
                False,
            ),
        ]

        retrieved = self._retrievedEntries(articleText, questionText, correctCandidate)
        diagnostics["retrievalCandidatesConsidered"] = len(retrieved)
        stageEntries.extend([
            ("retrieved", [dict(entry, quality="retrieved") for entry in retrieved if self._passesStageFilters(entry, articleText, correctCandidate, questionText, strict=True)], False),
            ("relaxed", [dict(entry, quality="relaxed") for entry in retrieved if self._passesStageFilters(entry, articleText, correctCandidate, questionText, relaxed=True)], True),
            ("last_resort", [dict(entry, quality="last_resort") for entry in retrieved if self._passesStageFilters(entry, articleText, correctCandidate, questionText, lastResort=True)], True),
        ])

        for stageName, entries, relaxed in stageEntries:
            rankedEntries = self._rankDistractorEntries(
                entries,
                articleText,
                questionText,
                correctCandidate,
                relaxed=relaxed,
                answerPredictor=predictor,
            )
            considered += len(rankedEntries)
            beforeCount = len(selected)
            self._appendDiverseEntries(selected, selectedKeys, rankedEntries, correctCandidate, maxDistractors=3)
            diagnostics["stageCounts"][stageName] = len(selected) - beforeCount
            if len(selected) >= 3:
                break

        if len(selected) < 3 and retrieved:
            rawFallbacks = [
                dict(entry, quality="last_resort")
                for entry in retrieved
                if optionLooksUsable(entry.get("text", ""), correctCandidate.text)
            ]
            rawFallbacks.sort(key=lambda item: float(item.get("retrievalScore", 0.0)), reverse=True)
            beforeCount = len(selected)
            self._appendDiverseEntries(selected, selectedKeys, rawFallbacks, correctCandidate, maxDistractors=3)
            diagnostics["stageCounts"]["last_resort"] += len(selected) - beforeCount

        if len(selected) < 3:
            articleFallbacks = [
                {
                    "text": candidate.text,
                    "source": "article",
                    "quality": "last_resort",
                    "retrievalScore": 0.0,
                    "score": float(getattr(candidate, "score", 0.0) or 0.0),
                }
                for candidate in allCandidates
                if optionLooksUsable(candidate.text, correctCandidate.text)
            ]
            articleFallbacks.sort(key=lambda item: item["score"], reverse=True)
            beforeCount = len(selected)
            self._appendDiverseEntries(selected, selectedKeys, articleFallbacks, correctCandidate, maxDistractors=3)
            diagnostics["stageCounts"]["last_resort"] += len(selected) - beforeCount

        diagnostics["candidatesConsidered"] = considered
        if selected:
            diagnostics["optionQuality"] = self._overallQuality(selected)
            diagnostics["fallbackUsed"] = diagnostics["optionQuality"] not in {"strict", "retrieved"}
            diagnostics["selectedDistractors"] = [
                {
                    "text": item["text"],
                    "source": item.get("source", ""),
                    "quality": item.get("quality", "last_resort"),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "retrievalScore": round(float(item.get("retrievalScore", 0.0)), 4),
                    "modelBScore": round(float(item.get("modelBScore", 0.0)), 4),
                }
                for item in selected
            ]

        options = [correctCandidate.text] + [item["text"] for item in selected[:3]]
        if len(options) < 4:
            # The corpus path should normally fill all three distractors.  If the
            # corpus is missing, keep using unique same-passage compatible options
            # rather than inventing manual placeholders.
            for fallbackOption in buildAnswerOptions(correctCandidate, allCandidates):
                key = normaliseOptionKey(fallbackOption)
                if key not in selectedKeys and key != correctKey:
                    options.append(fallbackOption)
                    selectedKeys.add(key)
                if len(options) == 4:
                    break

        return {
            "options": options[:4],
            "distractors": selected[:3],
            "diagnostics": diagnostics,
        }

    def buildHint(self, correctCandidate, articleText=""):
        """
        Return three graduated hints for the given answer candidate.

        If articleText is provided, all three levels are generated.
        If articleText is absent, all three slots fall back to the
        masked-sentence hint (backward compatible).
        """
        if articleText:
            return buildGraduatedHints(correctCandidate, articleText)
        # Backward-compat fallback: repeat the masked hint at all three levels
        maskedHint = buildHint(correctCandidate)
        return [maskedHint, maskedHint, maskedHint]

    def rankDistractors(self, articleText, correctCandidate, allCandidates, questionText="", answerPredictor=None):
        return self.buildOptionsWithDiagnostics(
            articleText,
            correctCandidate,
            allCandidates,
            questionText=questionText,
            answerPredictor=answerPredictor,
        )["distractors"]

    def _samePassageEntries(self, articleText, correctCandidate, allCandidates, questionText):
        entries = []
        for candidate in allCandidates:
            if not isCompatibleDistractor(correctCandidate, candidate, questionText=questionText):
                continue
            if not optionLooksUsable(candidate.text, correctCandidate.text):
                continue
            entries.append({
                "text": candidate.text,
                "source": "article",
                "quality": "strict",
                "retrievalScore": 1.0,
                "optionCategory": getAnswerCategory(candidate, questionText),
                "answerCategory": getAnswerCategory(correctCandidate, questionText),
            })
        return entries

    def _retrievedEntries(self, articleText, questionText, correctCandidate):
        try:
            retrieved = self.retriever.retrieve(
                articleText,
                questionText,
                correctCandidate,
                limit=180,
            )
        except Exception:
            return []

        entries = []
        for item in retrieved:
            entry = dict(item)
            entry.setdefault("source", "race")
            entry.setdefault("quality", "retrieved")
            entries.append(entry)
        return entries

    def _passesStageFilters(
        self,
        entry,
        articleText,
        correctCandidate,
        questionText,
        strict=False,
        relaxed=False,
        lastResort=False,
    ):
        text = entry.get("text", "")
        if not optionLooksUsable(text, correctCandidate.text):
            return False
        if containsTokenPhrase(articleText, text):
            return False

        correctCategory = getAnswerCategory(correctCandidate, questionText)
        candidateCategory = entry.get("optionCategory") or inferOptionCategory(text, "", questionText)
        if lastResort:
            return True
        if strict:
            return categoryCompatibilityScore(correctCategory, candidateCategory) >= 1.0
        if relaxed:
            return categoryCompatibilityScore(correctCategory, candidateCategory, relaxed=True) > 0.0
        return False

    def _rankDistractorEntries(
        self,
        entries,
        articleText,
        questionText,
        correctCandidate,
        relaxed=False,
        answerPredictor=None,
    ):
        if not entries:
            return []

        correctCategory = getAnswerCategory(correctCandidate, questionText)
        modelBScores = self._scoreCandidateBatch(
            articleText,
            correctCandidate,
            [entry["text"] for entry in entries],
        )
        prelim = []
        for entry, modelBScore in zip(entries, modelBScores):
            text = entry["text"]
            candidateCategory = entry.get("optionCategory") or inferOptionCategory(text, "", questionText)
            retrievalScore = float(entry.get("retrievalScore", 0.0))
            typeScore = categoryCompatibilityScore(correctCategory, candidateCategory, relaxed=relaxed)
            lengthScore = lengthCompatibilityScore(correctCandidate.text, text)
            sourceScore = 0.12 if entry.get("source") == "article" else 0.0
            score = (
                0.42 * retrievalScore
                + 0.24 * modelBScore
                + 0.18 * typeScore
                + 0.12 * lengthScore
                + sourceScore
            )
            ranked = dict(entry)
            ranked.update({
                "score": score,
                "modelBScore": modelBScore,
                "typeScore": typeScore,
                "lengthScore": lengthScore,
                "optionCategory": candidateCategory,
            })
            prelim.append(ranked)

        prelim.sort(key=lambda item: item["score"], reverse=True)
        verifierEntries = prelim[:24]
        verifierMargins = self._verifierMarginsBatch(
            articleText,
            questionText,
            correctCandidate,
            [entry["text"] for entry in verifierEntries],
            answerPredictor,
        )
        for entry, verifierMargin in zip(verifierEntries, verifierMargins):
            entry["verifierMargin"] = verifierMargin
            entry["score"] += 0.16 * verifierMargin
        prelim.sort(key=lambda item: item["score"], reverse=True)
        return prelim

    def _verifierMarginsBatch(self, articleText, questionText, correctCandidate, candidateTexts, answerPredictor):
        if answerPredictor is None or not questionText or not candidateTexts:
            return [0.0 for _ in candidateTexts]
        try:
            options = [correctCandidate.text] + list(candidateTexts)
            features = [
                answerPredictor.getOptionFeatures(articleText, questionText, optionText)
                for optionText in options
            ]
            probabilities = answerPredictor._softVoteProba(features)
            correctScore = float(probabilities[0])
            return [
                max(-1.0, min(1.0, correctScore - float(score)))
                for score in probabilities[1:]
            ]
        except Exception:
            return [
                self._verifierMargin(
                    articleText,
                    questionText,
                    correctCandidate,
                    candidateText,
                    answerPredictor,
                )
                for candidateText in candidateTexts
            ]

    def _verifierMargin(self, articleText, questionText, correctCandidate, candidateText, answerPredictor):
        if answerPredictor is None or not questionText:
            return 0.0
        try:
            scores = answerPredictor.scoreOptions(
                articleText,
                questionText,
                [correctCandidate.text, candidateText],
            )
            return max(-1.0, min(1.0, float(scores.get("A", 0.0)) - float(scores.get("B", 0.0))))
        except Exception:
            return 0.0

    def _scoreCandidateBatch(self, articleText, correctCandidate, candidateTexts):
        if not candidateTexts:
            return []
        try:
            features = [
                self.getDistractorFeatures(correctCandidate.text, candidateText, articleText)
                for candidateText in candidateTexts
            ]
            return [float(score) for score in self.rankModel.predict_proba(features)[:, 1]]
        except Exception:
            return [0.0 for _ in candidateTexts]

    def _appendDiverseEntries(self, selected, selectedKeys, rankedEntries, correctCandidate, maxDistractors=3):
        for entry in rankedEntries:
            if len(selected) >= maxDistractors:
                return
            text = entry["text"]
            key = normaliseOptionKey(text)
            if not key or key in selectedKeys:
                continue
            if hasCorrectAnswerLeak(text, correctCandidate.text):
                continue
            if self._tooSimilarToSelected(text, selected):
                continue
            selected.append(entry)
            selectedKeys.add(key)

    def _tooSimilarToSelected(self, text, selected):
        words = set(textTokens(text))
        if not words:
            return True
        for item in selected:
            otherWords = set(textTokens(item["text"]))
            if not otherWords:
                continue
            overlap = len(words & otherWords) / max(min(len(words), len(otherWords)), 1)
            if overlap > 0.82:
                return True
        return False

    def _overallQuality(self, selected):
        qualityRank = {"strict": 0, "retrieved": 1, "relaxed": 2, "last_resort": 3}
        worst = max(selected, key=lambda item: qualityRank.get(item.get("quality", "last_resort"), 3))
        return worst.get("quality", "last_resort")

    def getDistractorFeatures(self, correctAnswer, candidateText, articleText):
        editDist = editDistance(correctAnswer.lower(), candidateText.lower())
        lengthDiff = abs(len(correctAnswer) - len(candidateText))
        inArticle = 1 if candidateText.lower() in articleText.lower() else 0
        cosineSimilarity = self.getCosineSimilarity(correctAnswer, candidateText)
        return [editDist, lengthDiff, inArticle, cosineSimilarity]

    def getCosineSimilarity(self, leftText, rightText):
        if self.embedder is None:
            return self.getLexicalSimilarity(leftText, rightText)

        leftEmbedding = self.embedder.encode([leftText], show_progress_bar=False)[0]
        rightEmbedding = self.embedder.encode([rightText], show_progress_bar=False)[0]
        leftNorm = np.linalg.norm(leftEmbedding)
        rightNorm = np.linalg.norm(rightEmbedding)
        if leftNorm == 0 or rightNorm == 0:
            return 0.0
        return float(np.dot(leftEmbedding, rightEmbedding) / (leftNorm * rightNorm))

    def getLexicalSimilarity(self, leftText, rightText):
        leftWords = set(cleanText(leftText).split())
        rightWords = set(cleanText(rightText).split())
        if not leftWords or not rightWords:
            return 0.0
        return len(leftWords.intersection(rightWords)) / math.sqrt(len(leftWords) * len(rightWords))


class QuestionRankerPredictor:
    """
    Wraps the trained Random Forest question ranker (rfQuestionRanker.joblib).

    At inference time it:
      1. Calls generateQuestionCandidates() to build all rule-based (question, answer)
         variants across the top answer candidates.
      2. Scores each variant with the RF classifier using the 39-dimensional feature
         vector from buildQuestionRankerFeatureVector().
      3. Returns the (QuestionCandidate, score) pair with the highest predicted
         probability of being the best question.

    Falls back gracefully: if the ranker is unavailable, callers should use the
    rule-based buildTemplateQuestion() instead.
    """

    def __init__(
        self,
        modelDir=DEFAULT_QUESTION_RANKER_MODEL_DIR,
    ):
        self.modelDir = Path(modelDir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rfModel = joblib.load(self.modelDir / "rfQuestionRanker.joblib")
        print(f"[QuestionRankerPredictor] loaded rfQuestionRanker from {self.modelDir}")

    @classmethod
    def fromZip(
        cls,
        zipPath=DEFAULT_QUESTION_RANKER_ZIP_PATH,
        modelDir=DEFAULT_QUESTION_RANKER_MODEL_DIR,
    ):
        """
        Load the ranker, extracting from the zip if the file is not on disk.
        Returns None silently if the zip is also absent (backward compatible).
        """
        try:
            extractedDir = ensureQuestionRankerFiles(zipPath, modelDir)
            return cls(extractedDir)
        except FileNotFoundError:
            print("[QuestionRankerPredictor] zip not found — ranker disabled, falling back to rule-based QG")
            return None
        except Exception as exc:
            print(f"[QuestionRankerPredictor] load error ({exc}) — falling back to rule-based QG")
            return None

    def selectBestQuestion(
        self,
        articleText,
        answerCandidates,
        answerPredictor=None,
        maxAnswerCandidates=8,
    ):
        """Return the highest ranked (question_str, answer_candidate) pair."""
        ranked = self.rankQuestionCandidates(
            articleText,
            answerCandidates,
            answerPredictor=answerPredictor,
            maxAnswerCandidates=maxAnswerCandidates,
        )
        if not ranked:
            return buildTemplateQuestion(answerCandidates[0]), answerCandidates[0]
        return ranked[0]["question"], ranked[0]["answerCandidate"]

    def rankQuestionCandidates(
        self,
        articleText,
        answerCandidates,
        answerPredictor=None,
        maxAnswerCandidates=8,
    ):
        """
        Generate and score all valid rule-based question candidates.  Returned
        rows are sorted best-first and include the matching AnswerCandidate.
        """
        # Import here to avoid circular dependency — question_ranker imports nothing from here
        from question_ranker import (
            generateQuestionCandidates,
            buildQuestionRankerFeatureVector,
            isObviouslyBadQuestion,
        )

        questionCandidates = generateQuestionCandidates(
            articleText,
            answerCandidates,
            maxAnswerCandidates=maxAnswerCandidates,
        )

        if not questionCandidates:
            return [{
                "question": buildTemplateQuestion(answerCandidates[0]),
                "answerCandidate": answerCandidates[0],
                "questionScore": 0.0,
            }]

        # Build feature matrix
        featureVectors = []
        validCandidates = []
        for qc in questionCandidates:
            if isObviouslyBadQuestion(qc.question):
                continue
            featureVec = buildQuestionRankerFeatureVector(
                articleText,
                qc,
                options=None,          # no options at this stage — Model-A features = 0
                answerPredictor=answerPredictor,
            )
            featureVectors.append(featureVec)
            validCandidates.append(qc)

        if not validCandidates:
            return [{
                "question": buildTemplateQuestion(answerCandidates[0]),
                "answerCandidate": answerCandidates[0],
                "questionScore": 0.0,
            }]

        import numpy as np
        xMatrix = np.array(featureVectors, dtype=np.float32)
        scores = self.rfModel.predict_proba(xMatrix)[:, 1]

        # Template-type bias: the RF's top features are all answer-level, so
        # generic_wh ("What is being described in the passage?") tends to win
        # over specific suffix_wh questions by a thin margin. Penalise generic
        # and cloze templates so suffix_wh wins whenever it produces a valid Q.
        # generic_wh would need to outscore suffix_wh by 12.5× to still win.
        _TEMPLATE_BIAS = {
            "definition": 1.20,   # prefix 'What is X known as?' — very reliable when it fires
            "where_wh":   1.10,   # 'Where did X happen?' — specific and rarely wrong
            "suffix_wh":  1.00,
            "cloze":      0.55,
            "generic_wh": 0.08,
        }
        biasArray = np.array(
            [_TEMPLATE_BIAS.get(qc.template_type, 1.0) for qc in validCandidates],
            dtype=np.float32,
        )
        scores = scores * biasArray

        rankedIndexes = np.argsort(scores)[::-1]
        rankedRows = []
        for index in rankedIndexes:
            qc = validCandidates[int(index)]
            answerCandidate = self._findAnswerCandidateForQuestionCandidate(qc, answerCandidates)
            rankedRows.append({
                "question": qc.question,
                "answerCandidate": answerCandidate,
                "questionScore": float(scores[int(index)]),
                "templateType": qc.template_type,
            })
        return rankedRows

    def _findAnswerCandidateForQuestionCandidate(self, questionCandidate, answerCandidates):
        answerLower = questionCandidate.answer_text.lower().strip()
        for candidate in answerCandidates:
            if candidate.text.lower().strip() == answerLower:
                return candidate
        return AnswerCandidate(
            text=questionCandidate.answer_text,
            sentence=questionCandidate.source_sentence,
            kind=questionCandidate.answer_kind,
            score=questionCandidate.answer_score,
        )


class TemplateQuizGenerator:
    def __init__(self, answerPredictor=None, modelBGenerator=None, questionRanker=None):
        self.answerPredictor = answerPredictor
        self.modelBGenerator = modelBGenerator
        self.questionRanker  = questionRanker

    def buildQuiz(self, articleText, answerText=None):
        candidates = findAnswerCandidates(articleText)
        if not candidates:
            raise ValueError("No answer candidates could be found in the article.")

        rankedSelections = []
        if answerText is not None:
            correctCandidate = self.chooseCandidate(candidates, answerText)
            rankedSelections = [{
                "question": buildTemplateQuestion(correctCandidate),
                "answerCandidate": correctCandidate,
                "questionScore": 0.0,
                "templateType": "manual",
            }]
        elif self.questionRanker is not None:
            rankedSelections = self.questionRanker.rankQuestionCandidates(
                articleText,
                candidates,
                answerPredictor=self.answerPredictor,
            )
        else:
            correctCandidate = self.chooseCandidate(candidates, None)
            rankedSelections = [{
                "question": buildTemplateQuestion(correctCandidate),
                "answerCandidate": correctCandidate,
                "questionScore": 0.0,
                "templateType": "fallback",
            }]

        if not rankedSelections:
            correctCandidate = self.chooseCandidate(candidates, answerText)
            rankedSelections = [{
                "question": buildTemplateQuestion(correctCandidate),
                "answerCandidate": correctCandidate,
                "questionScore": 0.0,
                "templateType": "fallback",
            }]

        bundle = self._chooseQuizBundle(articleText, candidates, rankedSelections)
        question = bundle["question"]
        correctCandidate = bundle["correctCandidate"]
        hints = bundle["hints"]
        options = bundle["options"]
        optionDiagnostics = bundle["optionDiagnostics"]

        options, correctLabel = shuffleOptions(options, articleText, correctCandidate.text)

        prediction = None
        if self.answerPredictor is not None:
            label, answer, scores = self.answerPredictor.predictAnswer(
                articleText,
                question,
                options,
                sourceSentence=correctCandidate.sentence,
            )
            prediction = {
                "label": label,
                "answer": answer,
                "scores": scores,
            }

        return {
            "question": question,
            "correctAnswer": correctCandidate.text,
            "correctLabel": correctLabel,
            "hints": hints,                    # list of 3 strings: general → specific
            "hint": hints[2],                  # backward-compat: level-3 masked hint
            "options": dict(zip(OPTION_LABELS, options)),
            "prediction": prediction,
            "optionQuality": optionDiagnostics.get("optionQuality", "unknown"),
            "optionDiagnostics": optionDiagnostics,
        }

    def _chooseQuizBundle(self, articleText, candidates, rankedSelections):
        bestBundle = None
        for selection in rankedSelections[:6]:
            question = selection["question"]
            correctCandidate = selection["answerCandidate"]
            if self.modelBGenerator is None:
                hints = buildGraduatedHints(correctCandidate, articleText)
                options = buildAnswerOptions(correctCandidate, candidates)
                optionDiagnostics = {
                    "optionQuality": "strict" if len(options) >= 4 else "last_resort",
                    "stageCounts": {"strict": max(0, len(options) - 1)},
                    "fallbackUsed": len(options) < 4,
                    "selectedDistractors": [{"text": option, "quality": "strict"} for option in options[1:]],
                }
            else:
                hints = self.modelBGenerator.buildHint(correctCandidate, articleText)
                optionBundle = self.modelBGenerator.buildOptionsWithDiagnostics(
                    articleText,
                    correctCandidate,
                    candidates,
                    questionText=question,
                    answerPredictor=self.answerPredictor,
                )
                options = optionBundle["options"]
                optionDiagnostics = optionBundle["diagnostics"]

            if len(options) < 4:
                continue

            verifierMatches = self._optionsKeepVerifierOnCorrect(
                articleText,
                question,
                correctCandidate,
                options,
            )
            bundle = {
                "question": question,
                "correctCandidate": correctCandidate,
                "hints": hints,
                "options": options,
                "optionDiagnostics": optionDiagnostics,
                "questionScore": float(selection.get("questionScore", 0.0)),
                "verifierMatches": verifierMatches,
            }
            bundle["selectionScore"] = self._bundleSelectionScore(bundle)

            if (
                bundle["optionDiagnostics"].get("optionQuality") in {"strict", "retrieved"}
                and bundle.get("verifierMatches")
            ):
                return bundle

            if bestBundle is None or bundle["selectionScore"] > bestBundle["selectionScore"]:
                bestBundle = bundle

        if bestBundle is None:
            selection = rankedSelections[0]
            correctCandidate = selection["answerCandidate"]
            question = selection["question"]
            hints = buildGraduatedHints(correctCandidate, articleText)
            options = buildAnswerOptions(correctCandidate, candidates)
            if len(options) < 4:
                raise ValueError("Could not build four answer options from the available article and corpus candidates.")
            bestBundle = {
                "question": question,
                "correctCandidate": correctCandidate,
                "hints": hints,
                "options": options,
                "optionDiagnostics": {
                    "optionQuality": "last_resort",
                    "fallbackUsed": True,
                    "selectedDistractors": [{"text": option, "quality": "last_resort"} for option in options[1:]],
                },
            }
        return bestBundle

    def _optionsKeepVerifierOnCorrect(self, articleText, question, correctCandidate, options):
        if self.answerPredictor is None:
            return True
        try:
            _, answer, _ = self.answerPredictor.predictAnswer(
                articleText,
                question,
                options,
                sourceSentence=correctCandidate.sentence,
            )
            return answer.lower() == correctCandidate.text.lower()
        except Exception:
            return False

    def _bundleSelectionScore(self, bundle):
        qualityScore = {
            "strict": 4.0,
            "retrieved": 3.0,
            "relaxed": 2.0,
            "last_resort": 1.0,
        }.get(bundle["optionDiagnostics"].get("optionQuality"), 0.5)
        verifierBonus = 0.35 if bundle.get("verifierMatches") else 0.0
        return qualityScore + verifierBonus + min(float(bundle.get("questionScore", 0.0)), 1.0) * 0.05

    def chooseCandidate(self, candidates, answerText=None):
        if answerText is None:
            return candidates[0]

        for candidate in candidates:
            if candidate.text.lower() == answerText.lower():
                return candidate

        return AnswerCandidate(
            text=answerText,
            sentence=candidates[0].sentence,
            kind="manual",
            score=10.0,
        )
