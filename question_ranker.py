"""
question_ranker.py
------------------
Reusable rule-based question-candidate generation and feature extraction for
the traditional ML question ranker.

The ranker does not generate language. It scores questions produced by
templates, using only features available at inference time.
"""

from dataclasses import dataclass
import math
import re
import string


OPTION_LABELS = ["A", "B", "C", "D"]
WH_WORDS = ["who", "what", "when", "where", "why", "how", "which"]
BAD_TRAILING_WORDS = {
    "a",
    "an",
    "and",
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
BAD_SECOND_WORDS = BAD_TRAILING_WORDS | {"now", "then", "there"}
VERB_WORDS = {
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


QUESTION_RANKER_FEATURE_NAMES = [
    "question_word_count",
    "question_char_count",
    "starts_with_wh",
    "is_cloze",
    "has_blank",
    "contains_answer_text",
    "has_verb_like_token",
    "second_word_bad",
    "trailing_word_bad",
    "double_punctuation",
    "wh_who",
    "wh_what",
    "wh_when",
    "wh_where",
    "wh_why",
    "wh_how",
    "wh_which",
    "answer_kind_date",
    "answer_kind_person",
    "answer_kind_acronym",
    "answer_kind_proper_noun",
    "answer_kind_keyword",
    "wh_matches_answer_kind",
    "answer_candidate_score",
    "answer_word_count",
    "source_sentence_index",
    "source_sentence_position",
    "source_sentence_word_count",
    "answer_position_in_sentence",
    "words_before_answer",
    "words_after_answer",
    "question_source_unigram_overlap",
    "question_article_unigram_overlap",
    "question_source_bigram_overlap",
    "model_a_correct_score",
    "model_a_best_other_score",
    "model_a_margin",
    "model_a_correct_rank",
    "model_a_score_spread",
]


@dataclass
class QuestionCandidate:
    question: str
    answer_text: str
    answer_kind: str
    answer_score: float
    source_sentence: str
    source_sentence_index: int
    template_type: str
    wh_word: str


def cleanText(text):
    text = str(text).lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenise(text):
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def bigrams(tokens):
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def splitSentences(articleText):
    sentences = re.split(r"(?<=[.!?])\s+", str(articleText).strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def cleanQuestionEnd(text):
    return re.sub(r"\s+", " ", str(text)).strip(" ,.;:")


def normaliseQuestion(text):
    text = cleanQuestionEnd(str(text).strip())
    text = re.sub(r"\?+", "?", text)
    return text if text.endswith("?") else f"{text}?"


def lowerFirstCharacter(text):
    text = str(text)
    if not text:
        return text
    if len(text) > 1 and text[0].isupper() and not text[1].islower():
        return text
    return text[0].lower() + text[1:]


def hasVerbLikeToken(text):
    tokens = tokenise(text)
    if any(token in VERB_WORDS for token in tokens):
        return True
    return any(token.endswith(("ed", "ing", "es")) for token in tokens if len(token) > 4)


def firstWhWord(question):
    tokens = tokenise(question)
    if not tokens:
        return ""
    return tokens[0] if tokens[0] in WH_WORDS else ""


def inferAnswerKind(answerText, sentence=""):
    answerText = str(answerText).strip()
    sentence = str(sentence)
    if re.fullmatch(r"(?:\d{4}|\d+\s*BC|[A-Z][a-z]+\s+\d{4})", answerText):
        return "date"
    if re.fullmatch(r"[A-Z]{2,}", answerText):
        return "acronym"
    if re.search(r"\b(named|soldier|scholar|king|queen|scientist|inventor|author|writer)\b", sentence, re.IGNORECASE):
        return "person"
    if re.fullmatch(r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?){0,4}", answerText):
        return "properNoun"
    return "keyword"


def expectedWhForKind(answerKind):
    if answerKind == "date":
        return "when"
    if answerKind == "person":
        return "who"
    return "what"


def findSentenceIndex(targetSentence, sentences):
    target = str(targetSentence).strip().lower()
    for index, sentence in enumerate(sentences):
        if sentence.strip().lower() == target:
            return index
    targetTokens = set(tokenise(target))
    bestIndex, bestOverlap = 0, -1
    for index, sentence in enumerate(sentences):
        overlap = len(targetTokens & set(tokenise(sentence)))
        if overlap > bestOverlap:
            bestIndex, bestOverlap = index, overlap
    return bestIndex


def findBestSourceSentence(articleText, answerText):
    sentences = splitSentences(articleText)
    if not sentences:
        return "", 0

    answerLower = str(answerText).lower()
    for index, sentence in enumerate(sentences):
        if answerLower and answerLower in sentence.lower():
            return sentence, index

    answerTokens = set(tokenise(answerText))
    bestIndex, bestScore = 0, -1.0
    for index, sentence in enumerate(sentences):
        sentenceTokens = set(tokenise(sentence))
        score = len(answerTokens & sentenceTokens) / max(len(answerTokens), 1)
        if score > bestScore:
            bestIndex, bestScore = index, score
    return sentences[bestIndex], bestIndex


def answerPositionFeatures(sentence, answerText):
    sentenceLower = str(sentence).lower()
    answerLower = str(answerText).lower()
    start = sentenceLower.find(answerLower)
    sentenceWords = tokenise(sentence)
    if start == -1:
        return -1.0, 0.0, float(len(sentenceWords))

    beforeText = sentence[:start]
    afterText = sentence[start + len(answerText):]
    beforeWords = tokenise(beforeText)
    afterWords = tokenise(afterText)
    return (
        start / max(len(sentence), 1),
        float(len(beforeWords)),
        float(len(afterWords)),
    )


def buildClozeQuestion(sentence, answerText):
    masked = re.sub(re.escape(str(answerText)), "_____", cleanQuestionEnd(sentence), count=1, flags=re.IGNORECASE)
    return normaliseQuestion(f"Which answer best completes this sentence: {masked}")


def buildWhQuestionFromSuffix(answerText, answerKind, sentence):
    sentence = cleanQuestionEnd(sentence)
    answerStart = sentence.lower().find(str(answerText).lower())
    if answerStart == -1:
        return None
    afterAnswer = sentence[answerStart + len(str(answerText)):].strip(" ,")
    if not afterAnswer:
        return None
    # Require the suffix to be long enough to form a meaningful question
    afterTokens = re.findall(r"[A-Za-z']+", afterAnswer)
    if len(afterTokens) < 3:
        return None
    # Reject if the suffix starts with a capitalised word — this usually means
    # the answer boundary was cut inside a multi-word proper noun
    if afterTokens and afterTokens[0][0].isupper():
        return None
    # Reject suffixes starting with subordinating conjunctions / relative pronouns
    # — these produce ungrammatical questions like "What because it generates…"
    _BAD_SUFFIX_STARTS = {
        "because", "since", "although", "though", "unless", "until",
        "while", "as", "after", "before", "when", "whenever",
        "which", "that", "who", "whose", "where", "whereby",
        "and", "but", "or", "so", "yet",
    }
    if afterTokens and afterTokens[0].lower() in _BAD_SUFFIX_STARTS:
        return None
    whWord = expectedWhForKind(answerKind).capitalize()
    return normaliseQuestion(f"{whWord} {lowerFirstCharacter(cleanQuestionEnd(afterAnswer))}")


def buildGenericQuestion(answerKind, sentence):
    if answerKind == "date":
        return "When did the event described in the passage take place?"
    if answerKind == "person":
        return "Who is being described in the passage?"
    return "What is being described in the passage?"


def buildDefinitionQuestion(answerText, sentence, articleText=None):
    """
    For 'known as / called' patterns where the answer is the LABEL at the end
    of the clause.  Builds a prefix-based question by extracting the subject.

    'The Mitochondria is famously known as the powerhouse of the cell'
    -> 'What is the Mitochondria famously known as?'

    When the subject is a pronoun ("It is known as…"), searches the full
    articleText for the nearest proper noun referent.
    """
    sentenceClean = cleanQuestionEnd(sentence)
    triggers = ["known as", "called", "referred to as", "named after"]
    for trigger in triggers:
        idx = sentenceClean.lower().find(trigger)
        if idx == -1:
            continue
        before = sentenceClean[:idx].strip(" ,")
        # Strip leading copula / adverbs to isolate the real subject
        before = re.sub(r"^(it is|it was|they are|they were|this is|this was|he is|she is)\s+", "", before, flags=re.IGNORECASE)
        _LONE_ADVERBS = {
            "famously", "commonly", "also", "often", "widely", "generally",
            "typically", "sometimes", "officially", "simply", "popularly",
            "formally", "colloquially", "informally", "known",
        }
        # Strip individual leading adverbs (handles 'famously' with no trailing space)
        words_in_before = before.strip().split()
        while words_in_before and words_in_before[0].lower() in _LONE_ADVERBS:
            words_in_before.pop(0)
        before = " ".join(words_in_before).strip()
        # If the subject collapsed to empty, a bare pronoun, or only adverbs, resolve it
        _bare_pronouns = {"it", "he", "she", "they", "this", "that", "these", "those"}
        if not before or before.lower() in _bare_pronouns:
            # Require 4+ chars total to avoid matching 'The', 'In', 'Of', etc.
            _PROP = r"\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*\b"
            # 1. Try the current sentence
            proper = re.search(_PROP, sentenceClean)
            # 2. Fall back to full article (finds the nearest proper noun)
            if proper is None and articleText:
                proper = re.search(_PROP, str(articleText))
            if proper:
                before = proper.group()
            else:
                continue  # cannot resolve referent — skip
        if len(before.split()) < 1:
            continue
        question = f"What is {lowerFirstCharacter(before)} {trigger}?"
        return normaliseQuestion(question)
    return None


def buildWhereQuestion(answerText, sentence):
    """
    For location-phrase answers.  Builds 'Where was/did X [action]?' with
    simple subject-auxiliary inversion.

    'Pierre-Francois Bouchard discovered a slab in the city of Rosetta'
    -> 'Where did Pierre-Francois Bouchard discover a slab?'
    """
    sentenceClean = cleanQuestionEnd(sentence)
    locStart = sentenceClean.lower().find(answerText.lower())
    if locStart == -1:
        return None
    beforeLoc = sentenceClean[:locStart].strip(" ,")
    if not beforeLoc or len(beforeLoc.split()) < 3:
        return None
    # Strip trailing preposition that led into the location phrase
    beforeLoc = re.sub(r"\s+(?:in|at|near|from|within)\s*$", "", beforeLoc, flags=re.IGNORECASE).strip()
    if not beforeLoc:
        return None
    # Apply subject-auxiliary inversion for grammatical 'Where' questions:
    # 'He was awarded…' -> 'Where was he awarded?'
    # 'Charles Darwin was born…' -> 'Where was Charles Darwin born?'
    _AUXILIARIES = ("was", "is", "were", "are", "has", "have", "had",
                    "will", "would", "could", "should", "did", "does", "do")
    inverted = None
    for aux in _AUXILIARIES:
        pattern = rf"^(.*?)\s+({aux})\s+(.*)"
        m = re.match(pattern, beforeLoc, re.IGNORECASE)
        if m:
            subject, auxiliary, rest = m.group(1), m.group(2), m.group(3)
            inverted = f"Where {auxiliary} {lowerFirstCharacter(subject)} {rest}"
            break
    if inverted is None:
        # No auxiliary found — produce 'Where [beforeLoc]?' as a simple fallback
        inverted = f"Where {lowerFirstCharacter(beforeLoc)}"
    return normaliseQuestion(inverted)


def generateQuestionCandidates(articleText, answerCandidates, maxAnswerCandidates=8):
    """
    Generate multiple rule-based question candidates for the ranker to score.

    answerCandidates can be any object with text, kind, score, and sentence
    attributes, including template_quiz_pipeline.AnswerCandidate.
    """
    sentences = splitSentences(articleText)
    generated = []
    seen = set()

    for answerRank, answerCandidate in enumerate(answerCandidates[:maxAnswerCandidates]):
        answerText = str(answerCandidate.text)
        sourceSentence = str(answerCandidate.sentence)
        if not sourceSentence:
            sourceSentence, sourceIndex = findBestSourceSentence(articleText, answerText)
        else:
            sourceIndex = findSentenceIndex(sourceSentence, sentences)

        answerKind = str(getattr(answerCandidate, "kind", "") or inferAnswerKind(answerText, sourceSentence))
        answerScore = float(getattr(answerCandidate, "score", 0.0) or 0.0)

        templates = []

        # --- Definition pattern (highest priority for 'definition' kind) ---
        if answerKind == "definition":
            defQ = buildDefinitionQuestion(answerText, sourceSentence, articleText=articleText)
            if defQ:
                templates.append(("definition", defQ))

        # --- Location pattern for 'Where' questions ---
        if answerKind == "location":
            whereQ = buildWhereQuestion(answerText, sourceSentence)
            if whereQ:
                templates.append(("where_wh", whereQ))

        # --- Standard suffix / generic / cloze ---
        whQuestion = buildWhQuestionFromSuffix(answerText, answerKind, sourceSentence)
        if whQuestion:
            templates.append(("suffix_wh", whQuestion))

        templates.append(("generic_wh", buildGenericQuestion(answerKind, sourceSentence)))
        templates.append(("cloze", buildClozeQuestion(sourceSentence, answerText)))

        for templateType, question in templates:
            question = normaliseQuestion(question)
            key = (question.lower(), answerText.lower())
            if key in seen:
                continue
            seen.add(key)
            generated.append(
                QuestionCandidate(
                    question=question,
                    answer_text=answerText,
                    answer_kind=answerKind,
                    answer_score=answerScore - answerRank * 0.05,
                    source_sentence=sourceSentence,
                    source_sentence_index=sourceIndex,
                    template_type=templateType,
                    wh_word=firstWhWord(question),
                )
            )

    return generated


def overlapRatio(leftText, rightText):
    left = set(tokenise(leftText))
    right = set(tokenise(rightText))
    return len(left & right) / len(left) if left else 0.0


def bigramOverlapRatio(leftText, rightText):
    left = bigrams(tokenise(leftText))
    right = bigrams(tokenise(rightText))
    return len(left & right) / len(left) if left else 0.0


def computeModelAFeatures(articleText, question, options, correctAnswer, answerPredictor):
    if answerPredictor is None or not options:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        scores = answerPredictor.scoreOptions(articleText, question, options)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    correctIndex = None
    correctLower = str(correctAnswer).lower()
    for index, option in enumerate(options[:4]):
        if str(option).lower() == correctLower:
            correctIndex = index
            break
    if correctIndex is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    correctLabel = OPTION_LABELS[correctIndex]
    correctScore = float(scores.get(correctLabel, 0.0))
    otherScores = [
        float(score) for label, score in scores.items()
        if label != correctLabel
    ]
    bestOther = max(otherScores) if otherScores else 0.0
    allScores = [float(scores.get(label, 0.0)) for label in OPTION_LABELS if label in scores]
    sortedScores = sorted(allScores, reverse=True)
    correctRank = 1 + sum(score > correctScore for score in allScores)
    scoreSpread = max(allScores) - min(allScores) if allScores else 0.0
    return correctScore, bestOther, correctScore - bestOther, float(correctRank), scoreSpread


def buildQuestionRankerFeatureVector(articleText, candidate, options=None, answerPredictor=None):
    question = candidate.question
    questionTokens = tokenise(question)
    sourceTokens = tokenise(candidate.source_sentence)
    articleTokens = tokenise(articleText)
    whWord = firstWhWord(question)
    secondWord = questionTokens[1] if len(questionTokens) > 1 else ""
    trailingWord = questionTokens[-1] if questionTokens else ""
    answerLower = str(candidate.answer_text).lower()
    questionLower = question.lower()
    answerPosition, beforeCount, afterCount = answerPositionFeatures(
        candidate.source_sentence,
        candidate.answer_text,
    )
    sentenceCount = max(len(splitSentences(articleText)), 1)
    expectedWh = expectedWhForKind(candidate.answer_kind)

    modelAValues = computeModelAFeatures(
        articleText,
        question,
        options or [],
        candidate.answer_text,
        answerPredictor,
    )

    values = [
        float(len(questionTokens)),
        float(len(question)),
        float(bool(whWord)),
        float(candidate.template_type == "cloze"),
        float("_____" in question),
        float(bool(answerLower and answerLower in questionLower)),
        float(hasVerbLikeToken(" ".join(questionTokens[1:]))),
        float(secondWord in BAD_SECOND_WORDS),
        float(trailingWord in BAD_TRAILING_WORDS),
        float(bool(re.search(r"([?!])\1+", question))),
    ]

    values.extend(float(whWord == item) for item in WH_WORDS)
    values.extend(
        [
            float(candidate.answer_kind == "date"),
            float(candidate.answer_kind == "person"),
            float(candidate.answer_kind == "acronym"),
            float(candidate.answer_kind == "properNoun"),
            float(candidate.answer_kind == "keyword"),
            float(whWord == expectedWh),
            float(candidate.answer_score),
            float(len(tokenise(candidate.answer_text))),
            float(candidate.source_sentence_index),
            float(candidate.source_sentence_index / sentenceCount),
            float(len(sourceTokens)),
            float(answerPosition),
            float(beforeCount),
            float(afterCount),
            overlapRatio(question, candidate.source_sentence),
            overlapRatio(question, articleText),
            bigramOverlapRatio(question, candidate.source_sentence),
        ]
    )
    values.extend(modelAValues)
    return values


def lexicalF1(leftText, rightText):
    left = set(tokenise(leftText))
    right = set(tokenise(rightText))
    if not left or not right:
        return 0.0
    precision = len(left & right) / len(left)
    recall = len(left & right) / len(right)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def goldQuestionSimilarity(candidateQuestion, goldQuestion):
    unigram = lexicalF1(candidateQuestion, goldQuestion)
    candBigrams = bigrams(tokenise(candidateQuestion))
    goldBigrams = bigrams(tokenise(goldQuestion))
    bigram = len(candBigrams & goldBigrams) / len(candBigrams) if candBigrams else 0.0
    whBonus = 1.0 if firstWhWord(candidateQuestion) and firstWhWord(candidateQuestion) == firstWhWord(goldQuestion) else 0.0
    return 0.65 * unigram + 0.25 * bigram + 0.10 * whBonus


def isObviouslyBadQuestion(question):
    tokens = tokenise(question)
    if len(tokens) < 5:
        return True
    if len(tokens) > 1 and tokens[1] in BAD_SECOND_WORDS:
        return True
    if tokens and tokens[-1] in BAD_TRAILING_WORDS:
        return True
    if re.search(r"\b(what|who|where|when|why|how)\s+(from|of|in|on|to|for)\b", question, re.IGNORECASE):
        return True
    return False

