"""Dataset assembly, splitting and the hard probe set.

The corpus in data.py is a single flat list of (sentence, role) pairs. This
module turns it into the pieces the evaluation needs:

    corpus_frame()   the full labelled corpus as three parallel lists
    split_probe()    a stratified held-out probe carved out of the corpus
    HARD_PROBE       a small, deliberately awkward set of sentences that are
                     not in the corpus at all, used as an out-of-sample check

Why a separate hard probe? Because a random held-out slice of a corpus you
wrote yourself will always flatter the model: the same drafting habits leak
into both sides of the split. The hard probe is written to be ambiguous --
short sentences with no verbs, sentences that mention a date but do not set
a deadline, sentences that ask for a document but only in a background
clause. If the model holds up on those, the result means something.
"""

from __future__ import annotations

import random

import data

RANDOM_STATE = 42


def corpus_frame():
    """Return (texts, labels) for the whole corpus."""
    texts = [s for s, _ in data.SENTENCES]
    labels = [l for _, l in data.SENTENCES]
    return texts, labels


def label_counts(labels) -> dict:
    out: dict[str, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out


def split_probe(test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Stratified split into (train_texts, train_labels, probe_texts, probe_labels).

    Stratified by hand rather than with train_test_split so that the exact
    composition of the probe is visible in the report. With three classes
    and a small corpus, hand-stratifying keeps every class represented in
    the probe at the same proportion as the corpus.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = {l: [] for l in data.LABELS}
    for text, label in data.SENTENCES:
        by_label[label].append(text)

    train_texts, train_labels, probe_texts, probe_labels = [], [], [], []
    for label, items in by_label.items():
        items = sorted(items)
        rng.shuffle(items)
        n_probe = max(1, round(len(items) * test_size))
        probe_items = items[:n_probe]
        train_items = items[n_probe:]
        train_texts.extend(train_items)
        train_labels.extend([label] * len(train_items))
        probe_texts.extend(probe_items)
        probe_labels.extend([label] * len(probe_items))

    return train_texts, train_labels, probe_texts, probe_labels


# --------------------------------------------------------------------------
# Hard probe: out-of-sample, deliberately ambiguous, never seen in training.
# --------------------------------------------------------------------------
HARD_PROBE: list[tuple[str, str]] = [
    # These set a real deadline but bury it in a subordinate clause.
    ("Failure to respond within the stipulated period will attract action under the rules.", "deadline"),
    ("The compliance window, which opened on 1 April, will close on 30 April.", "deadline"),
    ("No extension beyond 15 March will be granted under any circumstances.", "deadline"),
    ("The period allowed for preferring an appeal against this order is 30 days.", "deadline"),
    ("Payment received after the 10th of the month shall be treated as late.", "deadline"),
    # These are instructions phrased without the word "shall".
    ("Kindly upload the cancelled cheque leaf on the portal.", "action_required"),
    ("It is necessary for the applicant to produce the original testimonials.", "action_required"),
    ("The pending certificate may be collected from the counter after production of identity proof.", "action_required"),
    ("The details furnished must match the entries in the service record.", "action_required"),
    ("Before proceeding, ensure that the mobile number is registered in your name.", "action_required"),
    # These mention dates or documents but require nothing of the reader.
    ("The scheme was notified on 1 April and has been in operation since then.", "reference"),
    ("The report of the Committee was placed before the House on 15 March.", "reference"),
    ("A copy of this order is available in the records section for reference.", "reference"),
    ("The earlier circular dated 10 January stands withdrawn.", "reference"),
    ("The Department has received 4,200 applications in the current year.", "reference"),
    # These sound formal and legal but carry no obligation.
    ("The above facts are stated for the information of the Hon'ble Court.", "reference"),
    ("This office has been directed to forward the papers to the appellate authority.", "reference"),
    ("The matter stands adjourned to the next date of hearing.", "reference"),
    ("The provision was inserted by way of an amendment in the year 2019.", "reference"),
    ("The scheme is implemented through the district administration.", "reference"),
    # These are genuinely ambiguous and the model is expected to be uncertain.
    ("The applicant may, if so advised, submit a representation before the authority.", "action_required"),
    ("All establishments are hereby informed of the revised rate of contribution.", "reference"),
    ("The last date is approaching and pending applications should be completed.", "deadline"),
    ("The verification of documents will be taken up on 12 January.", "deadline"),
    ("Candidates are advised to retain the hall ticket for future use.", "action_required"),
]


def hard_probe_frame():
    texts = [t for t, _ in HARD_PROBE]
    labels = [l for _, l in HARD_PROBE]
    return texts, labels


if __name__ == "__main__":
    texts, labels = corpus_frame()
    print("corpus:", len(texts))
    for k, v in sorted(label_counts(labels).items()):
        print(f"  {k:16s} {v}")
    tr_t, tr_l, pr_t, pr_l = split_probe()
    print("train:", len(tr_t), "probe:", len(pr_t))
    print("hard probe:", len(HARD_PROBE))
