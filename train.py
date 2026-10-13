"""Train, evaluate and persist the SuchnaSaar models.

Running this script produces two artefacts and prints a report:

    model.joblib    the chosen model, ready for the Flask app
    metrics.json    every number quoted in the concept note and the deck

The evaluation has three parts, and all three are reported even when they
are unflattering:

1.  5-fold stratified cross-validation over the whole corpus. This is the
    headline number and it is the one most likely to be optimistic, because
    the corpus was written by one person in one style.
2.  A stratified held-out probe carved out of the corpus. Slightly more
    honest than (1), still contaminated by the same writing style.
3.  A hard, out-of-sample probe of deliberately ambiguous sentences that
    the model has never seen. This is the number that matters most, and it
    is the number that is expected to be the lowest.

The script also reports the per-class confusion matrix and the weights of
the engineered signals, because a classifier that cannot be explained to a
citizen is of no use in this project.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import dataset
import features
import models

OUT_MODEL = "model.joblib"
OUT_METRICS = "metrics.json"


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def _confusion(y_true, y_pred) -> dict:
    labels = dataset.data.LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = {}
    for i, actual in enumerate(labels):
        out[actual] = {pred: int(cm[i][j]) for j, pred in enumerate(labels)}
    return out


def cross_validate(candidates: dict, texts, labels) -> dict:
    """Run 5-fold stratified CV and return per-model metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=models.RANDOM_STATE)
    results = {}
    for name, model in candidates.items():
        t0 = time.time()
        pred = cross_val_predict(model, texts, labels, cv=skf)
        m = _metrics(labels, pred)
        m["confusion"] = _confusion(labels, pred)
        m["seconds"] = round(time.time() - t0, 1)
        results[name] = m
        print(f"  {name:10s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} ({m['seconds']}s)")
    return results


def evaluate_on(model, train_texts, train_labels, probe_texts, probe_labels) -> dict:
    model.fit(train_texts, train_labels)
    pred = model.predict(probe_texts)
    m = _metrics(probe_labels, pred)
    m["confusion"] = _confusion(probe_labels, pred)
    return m


def signal_importance(fitted_hybrid, top: int = 30) -> list:
    """Read the engineered-signal weights out of the fitted hybrid pipeline.

    The hybrid pipeline is a FeatureUnion of three blocks. The third block
    carries the engineered signals, scaled by MaxAbsScaler, so the
    coefficients that follow those columns are directly comparable to each
    other and can be ranked.
    """
    try:
        union = fitted_hybrid.named_steps["features"]
        names = list(union.transformer_list)
        # The signal block is the one whose transformer has get_feature_names_out
        # returning the engineered names. Locate it by name to stay explicit.
        sig_block = dict(names)["signals"]
        sig_names = list(sig_block.named_steps["extract"].get_feature_names_out())
        offset = 0
        for block_name, block in names:
            if block_name == "signals":
                break
            offset += len(block.get_feature_names_out())
        clf = fitted_hybrid.named_steps["clf"]
        coefs = clf.coef_  # shape (n_classes, n_features)
        desc = features.descriptions()
        rows = []
        # The coefficient rows follow clf.classes_, which scikit-learn sorts
        # alphabetically -- it is NOT the declaration order in data.LABELS.
        # Reading the index from the fitted estimator is the only safe way to
        # attribute a weight to the right class.
        class_index = {c: i for i, c in enumerate(clf.classes_)}
        idx_action = class_index["action_required"]
        idx_deadline = class_index["deadline"]
        for k, nm in enumerate(sig_names):
            col = offset + k
            rows.append(
                {
                    "signal": nm,
                    "description": desc.get(nm, nm),
                    "action_weight": round(float(coefs[idx_action][col]), 3),
                    "deadline_weight": round(float(coefs[idx_deadline][col]), 3),
                    "max_abs_weight": round(float(max(abs(coefs[idx_action][col]), abs(coefs[idx_deadline][col]))), 3),
                }
            )
        rows.sort(key=lambda r: -r["max_abs_weight"])
        return rows[:top]
    except Exception as exc:  # pragma: no cover - reported, not hidden
        return [{"error": f"signal importance unavailable: {exc}"}]


def top_text_features(fitted_hybrid, per_class: int = 10) -> dict:
    """Return the highest-weighted n-grams for each class, as a sanity check."""
    try:
        union = fitted_hybrid.named_steps["features"]
        blocks = dict(union.transformer_list)
        word = blocks["word"]
        word_names = list(word.get_feature_names_out())
        clf = fitted_hybrid.named_steps["clf"]
        class_index = {c: i for i, c in enumerate(clf.classes_)}
        out = {}
        for label in clf.classes_:
            coefs = clf.coef_[class_index[label]][: len(word_names)]
            order = np.argsort(coefs)[::-1][:per_class]
            out[label] = [
                {"term": word_names[j], "weight": round(float(coefs[j]), 3)} for j in order
            ]
        return out
    except Exception as exc:  # pragma: no cover
        return {"error": f"top text features unavailable: {exc}"}


def main() -> None:
    texts, labels = dataset.corpus_frame()
    counts = dataset.label_counts(labels)
    print("=" * 68)
    print("SuchnaSaar -- training and evaluation")
    print("=" * 68)
    print(f"corpus: {len(texts)} sentences")
    for k in dataset.data.LABELS:
        print(f"  {k:16s} {counts.get(k, 0)}")

    candidates = models.build_all()

    print("\n[1] 5-fold stratified cross-validation")
    cv = cross_validate(candidates, texts, labels)

    tr_t, tr_l, pr_t, pr_l = dataset.split_probe()
    print(f"\n[2] held-out probe from the corpus (n={len(pr_t)})")
    probe = {}
    for name, model in models.build_all().items():
        probe[name] = evaluate_on(model, tr_t, tr_l, pr_t, pr_l)
        print(f"  {name:10s} acc={probe[name]['accuracy']:.4f} macroF1={probe[name]['macro_f1']:.4f}")

    hb_t, hb_l = dataset.hard_probe_frame()
    print(f"\n[3] hard out-of-sample probe (n={len(hb_t)})")
    hard = {}
    for name, model in models.build_all().items():
        hard[name] = evaluate_on(model, texts, labels, hb_t, hb_l)
        print(f"  {name:10s} acc={hard[name]['accuracy']:.4f} macroF1={hard[name]['macro_f1']:.4f}")

    # Pick the winner on the hard probe, not on the flattering CV number.
    # Ties are broken on macro-F1 from cross-validation.
    chosen = max(
        hard.keys(),
        key=lambda n: (hard[n]["macro_f1"], cv[n]["macro_f1"]),
    )
    print(f"\nchosen model: {chosen}  ({models.MODEL_DESCRIPTIONS[chosen]})")

    final = models.build_all()[chosen]
    final.fit(texts, labels)
    joblib.dump(final, OUT_MODEL)

    # Per-class detail for the chosen model, on the hard probe and in CV.
    hard_pred = final.predict(hb_t)
    report = classification_report(hb_l, hard_pred, labels=dataset.data.LABELS, zero_division=0)

    metrics = {
        "project": "SuchnaSaar",
        "task": "Sentence-level triage of Indian government notices",
        "classes": dataset.data.LABELS,
        "corpus": {
            "total": len(texts),
            "per_class": counts,
            "probe_size": len(pr_t),
            "hard_probe_size": len(hb_t),
            "note": (
                "Sentences were written for this project in the register of Indian official "
                "drafting. They imitate the style; they are not quoted from any gazette."
            ),
        },
        "models": models.MODEL_DESCRIPTIONS,
        "chosen_model": chosen,
        "chosen_because": "highest macro-F1 on the hard out-of-sample probe",
        "cross_validation": {"folds": 5, "results": cv},
        "corpus_probe": {"n": len(pr_t), "results": probe},
        "hard_probe": {"n": len(hb_t), "results": hard, "report": report},
        "engineered_signals": {
            "count": features.N_ENGINEERED,
            "names": features.names(),
            "importance": signal_importance(final) if chosen == "hybrid" else signal_importance(models.build_hybrid().fit(texts, labels)),
        },
        "top_text_features": top_text_features(models.build_hybrid().fit(texts, labels)),
        "environment": {
            "note": "All dependencies are free and open source; no GPU and no paid API is used.",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    with open(OUT_METRICS, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"\nwrote {OUT_MODEL} and {OUT_METRICS}")
    print("\nper-class report for the chosen model on the hard probe:")
    print(report)

    majority = max(counts.values()) / sum(counts.values())
    print(f"majority-class baseline on the corpus: {majority:.4f}")
    print(f"chosen model on hard probe:            {hard[chosen]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
