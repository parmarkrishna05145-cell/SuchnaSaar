"""Candidate models for SuchnaSaar, and the honest reason each one exists.

The project deliberately compares four representations of the same problem
rather than jumping straight to a large model:

    baseline    TF-IDF over words and character n-grams. Learns legal
                vocabulary without any hand-written rules.
    hybrid      the same TF-IDF union, plus the engineered signals from
                features.py. Tests whether explicit modality cues add
                anything a bag of n-grams cannot see.
    latent      a compressed latent-semantic representation built with
                TF-IDF + TruncatedSVD. A cheap stand-in for a neural
                sentence encoder that still runs offline on a free tier.
    ensemble    the mean of the hybrid and latent probabilities.

Two notes on honesty, both carried into the report:

1.  The `latent` model is **not** a transformer. It is TF-IDF followed by
    truncated SVD, i.e. classical latent semantic analysis. It is used here
    because it is reproducible offline, has no model download and no GPU
    requirement, which is the constraint the programme sets. A multilingual
    encoder is listed in the roadmap as the next step, not claimed here.

2.  Every model is deliberately small and interpretable. On a corpus of a
    few thousand sentences a large model would overfit and would also hide
    the reasoning from the citizen, which defeats the purpose of the tool.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

import features

RANDOM_STATE = 42


class SignalExtractor(BaseEstimator, TransformerMixin):
    """Wrap features.transform as a scikit-learn transformer."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        return np.asarray(features.transform(X), dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(features.names(), dtype=object)


def _word():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        lowercase=True,
        strip_accents="unicode",
    )


def _char():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        lowercase=True,
    )


def _clf(C: float = 4.0):
    # A multinomial solver is required here: the task has three classes
    # (reference / action_required / deadline), and liblinear is
    # one-vs-rest only, which the report does not want to depend on.
    return LogisticRegression(
        C=C,
        max_iter=4000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


def build_baseline() -> Pipeline:
    """TF-IDF (word + char) -> logistic regression. No hand-written rules."""
    return Pipeline(
        [
            ("features", FeatureUnion([("word", _word()), ("char", _char())])),
            ("clf", _clf()),
        ]
    )


def build_hybrid() -> Pipeline:
    """TF-IDF union + engineered modality signals -> logistic regression."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", _word()),
                        ("char", _char()),
                        ("signals", Pipeline([("extract", SignalExtractor()), ("scale", MaxAbsScaler())])),
                    ]
                ),
            ),
            ("clf", _clf()),
        ]
    )


def build_latent() -> Pipeline:
    """TF-IDF -> truncated SVD (latent semantic analysis) -> logistic regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
            ("svd", TruncatedSVD(n_components=120, random_state=RANDOM_STATE)),
            ("clf", _clf(C=2.0)),
        ]
    )


class EnsembleModel(BaseEstimator):
    """Mean of two models' probabilities.

    Averaging a sparse-feature model with a dense one is a cheap way to
    reduce variance, and it is easy to explain to a reviewer: the two
    models make different kinds of mistakes, so the average is steadier
    than either alone. It is not a stacking meta-learner, and the report
    does not claim it is.
    """

    def __init__(self, a=None, b=None, weights=(0.6, 0.4)):
        self.a = a
        self.b = b
        self.weights = weights

    def fit(self, X, y):  # noqa: N803 - sklearn API
        self.a.fit(X, y)
        self.b.fit(X, y)
        # Both members see the same label set, so either one's classes_ is the
        # label vocabulary. It is stored here so predict() can hand back the
        # original string labels rather than an index.
        self.classes_ = np.asarray(self.a.classes_)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        pa = self.a.predict_proba(X)
        pb = self.b.predict_proba(X)
        return self.weights[0] * pa + self.weights[1] * pb

    def predict(self, X):  # noqa: N803 - sklearn API
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def build_ensemble() -> EnsembleModel:
    return EnsembleModel(build_hybrid(), build_latent())


def build_all() -> dict:
    """Return every candidate, keyed by the name used in metrics.json."""
    return {
        "baseline": build_baseline(),
        "hybrid": build_hybrid(),
        "latent": build_latent(),
        "ensemble": build_ensemble(),
    }


MODEL_DESCRIPTIONS = {
    "baseline": "TF-IDF(word 1-2 + char_wb 3-5) + LogisticRegression",
    "hybrid": "TF-IDF + engineered notice signals + LogisticRegression",
    "latent": "TF-IDF(char_wb 3-5) + TruncatedSVD(120) + LogisticRegression",
    "ensemble": "0.6 x hybrid + 0.4 x latent",
}
