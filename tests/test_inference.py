from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from template_quiz_pipeline import (  # noqa: E402
    AnswerCandidate,
    ModelBDistractorHintGenerator,
    buildTemplateQuestion,
    containsTokenPhrase,
    findAnswerCandidates,
    shuffleOptions,
)
from src.inference import loadPipeline  # noqa: E402
from question_ranker import (  # noqa: E402
    QUESTION_RANKER_FEATURE_NAMES,
    buildQuestionRankerFeatureVector,
    generateQuestionCandidates,
)


def testFindAnswerCandidatesGetsNamedEntity():
    articleText = "The James Webb Space Telescope was launched in December 2021."
    candidates = findAnswerCandidates(articleText)
    candidateTexts = [candidate.text for candidate in candidates]
    assert "James Webb Space Telescope" in candidateTexts


def testShuffleOptionsPreservesCorrectLabel():
    options = ["Mitochondria", "Nucleus", "Ribosome", "Chloroplast"]
    shuffledOptions, correctLabel = shuffleOptions(options, "biology passage", "Mitochondria")
    assert "Mitochondria" in shuffledOptions
    assert shuffledOptions[["A", "B", "C", "D"].index(correctLabel)] == "Mitochondria"


def testBrokenWhQuestionFallsBackToCloze():
    candidate = AnswerCandidate(
        text="two years",
        sentence="The visa is valid for two years from now.",
        kind="keyword",
        score=1.0,
    )

    question = buildTemplateQuestion(candidate)

    assert question != "What from now?"
    assert question.startswith("Which answer best completes this sentence:")
    assert "_____" in question


def testReadableWhQuestionIsPreserved():
    candidate = AnswerCandidate(
        text="Mitochondria",
        sentence="The Mitochondria is a double-membrane-bound organelle found in most eukaryotic organisms.",
        kind="properNoun",
        score=1.0,
    )

    question = buildTemplateQuestion(candidate)

    assert question == "What is a double-membrane-bound organelle found in most eukaryotic organisms?"


def testReadablePersonQuestionIsPreserved():
    candidate = AnswerCandidate(
        text="Pierre-Francois Bouchard",
        sentence="A French soldier named Pierre-Francois Bouchard discovered a large granite slab in the city of Rosetta.",
        kind="person",
        score=1.0,
    )

    question = buildTemplateQuestion(candidate)

    assert question == "Who discovered a large granite slab in the city of Rosetta?"


def testQuestionRankerFeatureVectorMatchesFeatureNames():
    articleText = "The James Webb Space Telescope was launched in December 2021."
    answerCandidate = AnswerCandidate(
        text="James Webb Space Telescope",
        sentence=articleText,
        kind="properNoun",
        score=1.0,
    )
    questionCandidate = generateQuestionCandidates(articleText, [answerCandidate], maxAnswerCandidates=1)[0]

    features = buildQuestionRankerFeatureVector(
        articleText,
        questionCandidate,
        options=[
            "James Webb Space Telescope",
            "Hubble Space Telescope",
            "Kepler Space Telescope",
            "Spitzer Space Telescope",
        ],
    )

    assert len(features) == len(QUESTION_RANKER_FEATURE_NAMES)


def testTokenPhraseMatchingDoesNotUseSubstrings():
    assert containsTokenPhrase("Students discussed RNA in biology class.", "RNA")
    assert not containsTokenPhrase("I have many fellow international classmates.", "RNA")


def testCorpusBackedDistractorsForMbaPassageAvoidOldBogusOptions():
    articleText = """
    The other day my aunt paid me a visit. She was overjoyed. "I got the highest mark in the mid-term examination!" She said. Don't be surprised! My aunt is indeed a student, exactly, a college student at the age of 45.
    "Compared with the late 70s," she says, "now college students have many doors." I was shocked when she first told me how she had had no choice in her major. Look at us today! So many doors are open to us! I believe there have never been such abundant opportunities for self-development as we have today. And my aunt told me that we should reach our goals by grasping all these opportunities.
    The first door is the opportunity to study different subjects that interest us. My aunt was happy to study management, but she could also attend lectures on ancient Chinese poetry and on Shakespearean drama. As for myself, I am an English major, but I may also go to lectures on history.
    The second door is the door to the outside world. Learning goes beyond classrooms and national boundaries. I have many fellow international classmates, and I am applying to an exchange program with a university abroad. As for my aunt, she is planning to get an MBA degree in the U.K.
    The third door is the door to life-long learning. Many of my aunt's contemporaries say she's amazingly up-to-date for a middle-aged woman. She simply responds, "Age doesn't matter. What matters is your attitude. I don't think I'm too old to learn." Yes, she is right. Since the government removed the age limit for college admissions, there are already some untraditional students, sitting with us in the same classrooms. Like them, my aunt is old but young in spirit with incredible energy and determination.
    The doors open to us also pose challenges. For instance, we are faced with the challenge of a balanced learning, the challenge of preserving our fine tradition while learning from the West, and the challenge of learning continuously while carrying heavy responsibilities to our work and family. So, each door is a test of our courage, ability and judgment, but with the support of my teachers, parents, friends and my aunt, I believe I can meet the challenge head on.
    """.strip()

    quiz = loadPipeline().generateQuiz(articleText)
    optionTexts = set(quiz["options"].values())

    assert len(optionTexts) == 4
    assert quiz["correctAnswer"] == "MBA"
    assert {"West", "CPU", "DNA"}.isdisjoint(optionTexts)
    assert quiz["optionQuality"] in {"strict", "retrieved", "relaxed", "last_resort"}
    assert quiz["prediction"]["answer"] == "MBA"


def testIncompatibleSamePassageCandidateIsNotEmergencyFiller():
    articleText = (
        "As for my aunt, she is planning to get an MBA degree in the U.K. "
        "The family also discussed preserving tradition while learning from the West."
    )
    correct = AnswerCandidate(
        text="MBA",
        sentence="As for my aunt, she is planning to get an MBA degree in the U.K.",
        kind="acronym",
        score=1.0,
    )
    west = AnswerCandidate(
        text="West",
        sentence="The family also discussed preserving tradition while learning from the West.",
        kind="location",
        score=1.0,
    )

    generator = ModelBDistractorHintGenerator.fromZip()
    bundle = generator.buildOptionsWithDiagnostics(
        articleText,
        correct,
        [correct, west],
        questionText="What degree in the U.K?",
    )

    assert len(bundle["options"]) == 4
    assert "West" not in bundle["options"]
