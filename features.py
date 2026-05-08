"""
features.py
-----------
Shared feature engineering for Model A.

Imported by BOTH the training pipeline (src/model_a_train.py) and the
inference predictor (template_quiz_pipeline.py) to guarantee that the
feature vector is identical between training and inference.

Feature vector (12 dimensions)
-------------------------------
BASE features (0-9):
  0  tfidf_cosine_sim        — document-level TF-IDF cosine sim (article vs query)
  1  unigram_overlap_ratio   — (question∪option tokens) ∩ article / (question∪option)
  2  idf_weighted_score      — mean IDF of option tokens that appear in article
  3  best_sentence_overlap   — max unigram overlap across all article sentences ★
  4  bigram_overlap_ratio    — bigram version of feature 1
  5  exact_match_flag        — 1 if option text appears verbatim in article
  6  option_token_length     — number of whitespace-delimited tokens in option
  7  option_char_length      — number of characters in option
  8  first_position          — normalised first-occurrence position (−1 if absent)
  9  is_number               — 1 if option is primarily numeric / date-like

CLUSTER features (10-11)  — appended after the unsupervised K-Means step:
  10 kmeans_cluster_label    — hard cluster assignment (0 or 1)
  11 kmeans_centroid_dist    — distance to nearest centroid (continuous)
"""

import math
import re
import string

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "tfidf_cosine_sim",
    "unigram_overlap_ratio",
    "idf_weighted_score",
    "best_sentence_overlap",
    "bigram_overlap_ratio",
    "exact_match_flag",
    "option_token_length",
    "option_char_length",
    "first_position",
    "is_number",
    "kmeans_cluster_label",
    "kmeans_centroid_dist",
]

BASE_FEATURE_COUNT = 10   # features 0-9 (built from text alone)
FEATURE_COUNT = len(FEATURE_NAMES)   # 12 total

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NUMBER_RE = re.compile(r"^\d[\d\s,.\-/]*$")
_DATE_RE = re.compile(
    r"\b(\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text helpers (module-private)
# ---------------------------------------------------------------------------

def _clean(text):
    return str(text).lower().translate(_PUNCT_TABLE)


def _tokenize(text):
    return _clean(text).split()


def _bigrams(tokens):
    if len(tokens) < 2:
        return set()
    return set(zip(tokens, tokens[1:]))


def _splitSentences(cleanArticle):
    return [s.strip() for s in re.split(r"[.!?]+", cleanArticle) if s.strip()]


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def computeTfidfCosineSim(cleanArticle, cleanQuery, tfidfVec):
    """Feature 0: document-level TF-IDF cosine similarity."""
    if tfidfVec is None:
        return computeLexicalCosineSim(cleanArticle, cleanQuery)
    articleVec = tfidfVec.transform([cleanArticle])
    queryVec = tfidfVec.transform([cleanQuery])
    numerator = float(articleVec.multiply(queryVec).sum())
    denominator = math.sqrt(
        float(articleVec.multiply(articleVec).sum())
        * float(queryVec.multiply(queryVec).sum())
    )
    return numerator / denominator if denominator > 0 else 0.0


def computeLexicalCosineSim(cleanArticle, cleanQuery):
    """Fallback Jaccard-style similarity when TF-IDF vectorizer is absent."""
    aWords = set(cleanArticle.split())
    qWords = set(cleanQuery.split())
    if not aWords or not qWords:
        return 0.0
    return len(aWords & qWords) / math.sqrt(len(aWords) * len(qWords))


def computeUnigramOverlapRatio(cleanQuestion, cleanOption, cleanArticle):
    """Feature 1: fraction of (question ∪ option) tokens present in article."""
    queryTokens = set(cleanQuestion.split()) | set(cleanOption.split())
    articleTokens = set(cleanArticle.split())
    return len(queryTokens & articleTokens) / len(queryTokens) if queryTokens else 0.0


def computeIdfWeightedScore(cleanOption, cleanArticle, idfDict):
    """Feature 2: mean IDF weight of option tokens that appear in the article."""
    optionTokens = set(cleanOption.split())
    articleTokens = set(cleanArticle.split())
    shared = optionTokens & articleTokens
    if not shared:
        return 0.0
    return sum(idfDict.get(t, 0.0) for t in shared) / len(shared)


def computeBestSentenceOverlap(cleanOption, cleanArticle):
    """
    Feature 3: max unigram overlap between the option and any single article sentence.

    This is the most discriminative traditional-ML feature for MCQ: the correct
    answer typically aligns strongly with one specific sentence in the article,
    while distractors do not.
    """
    optionTokens = set(cleanOption.split())
    if not optionTokens:
        return 0.0
    bestScore = 0.0
    for sentence in _splitSentences(cleanArticle):
        sentenceTokens = set(sentence.split())
        if sentenceTokens:
            score = len(optionTokens & sentenceTokens) / len(optionTokens)
            if score > bestScore:
                bestScore = score
    return bestScore


def computeBigramOverlapRatio(cleanQuestion, cleanOption, cleanArticle):
    """Feature 4: bigram version of the unigram overlap ratio."""
    queryBigrams = _bigrams(_tokenize(cleanQuestion)) | _bigrams(_tokenize(cleanOption))
    articleBigrams = _bigrams(_tokenize(cleanArticle))
    return len(queryBigrams & articleBigrams) / len(queryBigrams) if queryBigrams else 0.0


def computeExactMatchFlag(cleanOption, cleanArticle):
    """Feature 5: 1 if the option text appears verbatim in the article."""
    return 1.0 if cleanOption and cleanOption in cleanArticle else 0.0


def computeFirstPosition(cleanOption, cleanArticle):
    """Feature 8: normalised first-occurrence position; −1 if absent."""
    pos = cleanArticle.find(cleanOption)
    if pos == -1 or len(cleanArticle) == 0:
        return -1.0
    return pos / len(cleanArticle)


def computeIsNumber(cleanOption):
    """Feature 9: 1 if the option is primarily numeric or date-like."""
    stripped = cleanOption.strip()
    if _NUMBER_RE.match(stripped):
        return 1.0
    if _DATE_RE.search(stripped):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Composite: build the full BASE feature vector for one triple
# ---------------------------------------------------------------------------

def buildBaseFeatureVector(cleanArticle, cleanQuestion, cleanOption, idfDict, tfidfVec):
    """
    Construct the 10-dimensional BASE feature vector for one
    (article, question, option) triple.

    Parameters
    ----------
    cleanArticle  : lowercased, punctuation-stripped article text
    cleanQuestion : lowercased, punctuation-stripped question text
    cleanOption   : lowercased, punctuation-stripped option text
    idfDict       : {token: idf_weight} from the fitted TfidfVectorizer
    tfidfVec      : fitted TfidfVectorizer (or None for lexical fallback)

    Returns
    -------
    list of 10 floats
    """
    queryText = f"{cleanQuestion} {cleanOption}".strip()

    return [
        computeTfidfCosineSim(cleanArticle, queryText, tfidfVec),   # 0
        computeUnigramOverlapRatio(cleanQuestion, cleanOption, cleanArticle),  # 1
        computeIdfWeightedScore(cleanOption, cleanArticle, idfDict),  # 2
        computeBestSentenceOverlap(cleanOption, cleanArticle),        # 3
        computeBigramOverlapRatio(cleanQuestion, cleanOption, cleanArticle),   # 4
        computeExactMatchFlag(cleanOption, cleanArticle),             # 5
        float(len(cleanOption.split())),                              # 6
        float(len(cleanOption)),                                      # 7
        computeFirstPosition(cleanOption, cleanArticle),              # 8
        computeIsNumber(cleanOption),                                 # 9
    ]


# ---------------------------------------------------------------------------
# Batch builder (used in training for speed)
# ---------------------------------------------------------------------------

def buildBaseFeatureMatrixBatch(dataFrame, idfDict, tfidfVec):
    """
    Build the (n, 10) BASE feature matrix for an exploded-options DataFrame.

    Uses vectorised TF-IDF transforms for features 0-2 and a row-wise loop
    only for the sentence-level feature (feature 3), which requires per-row
    sentence splitting.

    Parameters
    ----------
    dataFrame : DataFrame with columns cleanArticle, cleanQuestion, cleanOption
    idfDict   : {token: idf_weight}
    tfidfVec  : fitted TfidfVectorizer

    Returns
    -------
    np.ndarray of shape (n_rows, BASE_FEATURE_COUNT), dtype float32
    """
    n = len(dataFrame)
    queryTexts = (
        dataFrame["cleanQuestion"].fillna("") + " " + dataFrame["cleanOption"].fillna("")
    ).tolist()
    articleTexts = dataFrame["cleanArticle"].fillna("").tolist()
    optionTexts = dataFrame["cleanOption"].fillna("").tolist()
    questionTexts = dataFrame["cleanQuestion"].fillna("").tolist()

    # --- Batch TF-IDF cosine similarity (feature 0) ---
    if tfidfVec is not None:
        articleVecs = tfidfVec.transform(articleTexts)
        queryVecs = tfidfVec.transform(queryTexts)
        from sklearn.metrics.pairwise import paired_cosine_distances
        tfidfSims = (1.0 - paired_cosine_distances(articleVecs, queryVecs)).astype(np.float32)
    else:
        tfidfSims = np.array(
            [computeLexicalCosineSim(a, q) for a, q in zip(articleTexts, queryTexts)],
            dtype=np.float32,
        )

    # --- Allocate output matrix ---
    matrix = np.zeros((n, BASE_FEATURE_COUNT), dtype=np.float32)
    matrix[:, 0] = tfidfSims

    # --- Row-wise features ---
    for i, (art, qst, opt) in enumerate(zip(articleTexts, questionTexts, optionTexts)):
        matrix[i, 1] = computeUnigramOverlapRatio(qst, opt, art)
        matrix[i, 2] = computeIdfWeightedScore(opt, art, idfDict)
        matrix[i, 3] = computeBestSentenceOverlap(opt, art)
        matrix[i, 4] = computeBigramOverlapRatio(qst, opt, art)
        matrix[i, 5] = computeExactMatchFlag(opt, art)
        matrix[i, 6] = float(len(opt.split()))
        matrix[i, 7] = float(len(opt))
        matrix[i, 8] = computeFirstPosition(opt, art)
        matrix[i, 9] = computeIsNumber(opt)

    return matrix


# ---------------------------------------------------------------------------
# Cluster feature helper (used by both training and inference)
# ---------------------------------------------------------------------------

def buildClusterFeaturesIfAvailable(optionTfidfMatrix, kmeansModel, svdModel):
    """
    Produce 2 K-Means cluster features for each option.

    Returns np.ndarray of shape (n_rows, 2): [cluster_label, centroid_dist].
    If kmeansModel or svdModel is None, returns zeros.
    """
    if kmeansModel is None or svdModel is None:
        n = optionTfidfMatrix.shape[0]
        return np.zeros((n, 2), dtype=np.float32)
    reducedVecs = svdModel.transform(optionTfidfMatrix)
    clusterLabels = kmeansModel.predict(reducedVecs).reshape(-1, 1).astype(np.float32)
    centroidDists = kmeansModel.transform(reducedVecs).min(axis=1).reshape(-1, 1).astype(np.float32)
    return np.hstack([clusterLabels, centroidDists])
