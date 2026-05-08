from pathlib import Path
import random
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from template_quiz_pipeline import (  # noqa: E402
    EnsembleAnswerPredictor,
    ModelBDistractorHintGenerator,
    QuestionRankerPredictor,
    TemplateQuizGenerator,
    findAnswerCandidates,
)

# Alias kept for any external code that still uses the old name
LogisticAnswerPredictor = EnsembleAnswerPredictor


DEFAULT_SAMPLE_ARTICLE = """
The James Webb Space Telescope was launched in December 2021. It orbits the Sun
1.5 million kilometers away from Earth. Scientists use the telescope to observe
distant galaxies and study how stars and planets form.
""".strip()


ROSETTA_SAMPLE_ARTICLE = """
In July 1799, during Napoleon's Egyptian campaign, a French soldier named
Pierre-Francois Bouchard discovered a large granite slab in the city of Rosetta.
This stone was inscribed with a decree issued in 196 BC on behalf of King
Ptolemy V. Because the inscriptions included Ancient Greek alongside Egyptian
Hieroglyphs, it became the key to deciphering the lost language of the pharaohs.
For centuries, the meaning of hieroglyphs had been forgotten, but the Rosetta
Stone provided the translation needed to unlock them. Eventually, a brilliant
scholar named Jean-Francois Champollion decoded the script in 1822, opening a
window into three thousand years of Egyptian history.
""".strip()


class ReadingComprehensionPipeline:
    """
    End-to-end inference pipeline combining Model A (verifier ensemble)
    and Model B (distractor / hint generator).
    """

    def __init__(self):
        self.answerPredictor  = EnsembleAnswerPredictor.fromZip()
        # Model B uses lexical similarity at inference time (no neural embedder)
        self.modelBGenerator  = ModelBDistractorHintGenerator.fromZip(
            embedder=None,
            answerPredictor=self.answerPredictor,
        )
        self.modelBGenerator.warmUpRetriever()
        # RF question ranker — picks the best (question, answer) pair from all candidates
        self.questionRanker   = QuestionRankerPredictor.fromZip()
        self.generator = TemplateQuizGenerator(
            answerPredictor=self.answerPredictor,
            modelBGenerator=self.modelBGenerator,
            questionRanker=self.questionRanker,
        )

    def generateQuiz(self, articleText, answerText=None):
        startTime = time.perf_counter()
        quizResult = self.generator.buildQuiz(articleText, answerText=answerText)
        quizResult["latencySec"] = round(time.perf_counter() - startTime, 3)
        quizResult["modelInfo"] = self.getModelInfo()
        return quizResult

    def checkAnswer(self, quizResult, selectedLabel):
        selectedOption = quizResult["options"].get(selectedLabel)
        isCorrect = selectedLabel == quizResult["correctLabel"]
        return {
            "isCorrect": isCorrect,
            "selectedLabel": selectedLabel,
            "selectedOption": selectedOption,
            "correctLabel": quizResult["correctLabel"],
            "correctAnswer": quizResult["correctAnswer"],
        }

    def getModelInfo(self):
        rankerStatus = "loaded" if self.questionRanker is not None else "unavailable (fallback: rule-based)"
        return {
            "modelA": (
                "RF Question Ranker (rfQuestionRanker · ROC-AUC 0.972 · top-1 match 66.9%) "
                "selects best (question, answer) pair from rule-generated candidates; "
                "Soft-vote ensemble verifier (LR + Calibrated SVM + Random Forest, "
                "8 TF-IDF features + K-Means cluster features) ranks answer options"
            ),
            "questionRanker": rankerStatus,
            "modelABaseline": "T5-small fine-tuned on RACE (neural baseline, comparison only)",
            "modelB": (
                "Random Forest distractor ranker "
                "(edit distance, length diff, in-article flag, lexical cosine sim) "
                "+ extractive hint masking"
            ),
            "unsupervised": "KMeans(k=2) on TruncatedSVD(50) of option TF-IDF vectors",
            "ensemble": "Soft vote: mean(LR_proba, SVM_proba, RF_proba)",
            "tieBreaker": "Soft-vote ties prefer the option found in the source sentence",
        }


def loadPipeline():
    return ReadingComprehensionPipeline()


def generateQuiz(articleText, answerText=None):
    return loadPipeline().generateQuiz(articleText, answerText=answerText)


def getCandidatePreview(articleText, limit=8):
    return [
        {
            "text": candidate.text,
            "kind": candidate.kind,
            "score": round(candidate.score, 3),
            "sentence": candidate.sentence,
        }
        for candidate in findAnswerCandidates(articleText)[:limit]
    ]


def loadRandomRaceArticle(dataPath=PROJECT_ROOT / "data" / "raw" / "train.csv"):
    if not Path(dataPath).exists():
        return DEFAULT_SAMPLE_ARTICLE

    import pandas as pd

    dataFrame = pd.read_csv(dataPath)
    if dataFrame.empty or "article" not in dataFrame.columns:
        return DEFAULT_SAMPLE_ARTICLE

    index = random.randint(0, len(dataFrame) - 1)
    return str(dataFrame.iloc[index]["article"])


def generateT5Question(articleText: str, answerText: str) -> dict:
    """
    Generate a question using the fine-tuned T5-small model (neural baseline).

    Lazy-loads transformers only when called so the traditional pipeline stays
    dependency-free. Returns a dict with 'question' and 'latencySec', or
    {'error': reason} if the model is unavailable.

    The T5 was fine-tuned on Wh- questions from RACE (20k rows, 5 epochs).
    Input format:  "generate question: context: <article> answer: <answer>"
    """
    modelPath = PROJECT_ROOT / "models" / "model_a" / "neural" / "model_a_final"
    if not modelPath.exists():
        return {
            "question": None,
            "error": (
                f"T5 model not found at {modelPath}. "
                "Run: python3 src/model_a_train.py --task neural --train-csv data/raw/train.csv"
            ),
        }

    try:
        # Lazy imports — keep traditional pipeline dependency-free
        import string as _string
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        t0 = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(str(modelPath))
        model     = T5ForConditionalGeneration.from_pretrained(str(modelPath))

        # Pre-process exactly as in the neural training path.
        def _clean(text):
            return str(text).lower().translate(str.maketrans("", "", _string.punctuation))

        prompt = (
            "generate question: context: "
            + _clean(articleText)
            + " answer: "
            + _clean(answerText)
        )

        inputs     = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids  = inputs["input_ids"]
        attn_mask  = inputs.get("attention_mask", None)

        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=64,
            num_beams=4,
        )
        if attn_mask is not None:
            generate_kwargs["attention_mask"] = attn_mask

        outputs  = model.generate(**generate_kwargs)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)

        latency = round(time.perf_counter() - t0, 2)
        return {"question": question.capitalize(), "latencySec": latency, "error": None}

    except Exception as exc:
        return {"question": None, "error": f"{type(exc).__name__}: {exc}"}
