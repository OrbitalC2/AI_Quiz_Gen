import numpy as np

from src.model_a_train import softVoteScores


class StaticProbabilityModel:
    """Minimal classifier double returning one fixed positive probability."""

    def __init__(self, positiveProbability):
        self.positiveProbability = positiveProbability

    def predict_proba(self, features):
        positive = np.full(len(features), self.positiveProbability)
        return np.column_stack((1.0 - positive, positive))


def testSoftVoteUsesLrSvmAndRfButExcludesNaiveBayes():
    models = {
        "lr": StaticProbabilityModel(0.2),
        "svm": StaticProbabilityModel(0.4),
        "nb": StaticProbabilityModel(0.99),
        "rf": StaticProbabilityModel(0.6),
    }

    scores = softVoteScores(models, np.zeros((2, 1)))

    np.testing.assert_allclose(scores, [0.4, 0.4])
