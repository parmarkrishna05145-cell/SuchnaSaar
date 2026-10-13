"""The agentic layer: turning a scored notice into a plan a citizen can act on.

The classifier in models.py can tell you that a sentence is an instruction
or a deadline. It cannot tell you what to do about it. This module does that
job, and it does it with rules rather than with another model, for three
reasons:

*   A rule can be read. If the agent tells someone to visit a portal, the
    line of code that made that decision is visible in this file.
*   A rule can be corrected by a non-programmer. When the wording of a
    notice changes, the fix is one line, not a retraining run.
*   A rule cannot hallucinate a helpline number. Every number and every
    portal in this file is written down in data.py with the official domain
    it was verified against. Nothing is generated.

The agent produces, for one pasted notice:

    verdict        a short summary of what the notice is asking for
    role_counts    how many sentences fell into each role
    actions        a numbered checklist, each item quoting the sentence
    deadlines      the dates and periods found, in order of appearance
    scheme         the scheme the notice appears to concern, if any
    routes         the portals and helplines worth visiting
    safety         whether the notice looks like a genuine official document
    caveat         a plain statement of what the tool cannot do
"""

from __future__ import annotations

import re

import data
import dataset
import features

# --------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------
_ABBREV = [
    "no.", "nos.", "rs.", "sr.", "shri", "smt.", "dr.", "viz.", "etc.",
    "i.e.", "e.g.", "w.e.f.", "u/s", "sec.", "fig.", "vol.", "hon.",
    "govt.", "dept.", "ex.", "guj.",
]

_SENT_SPLIT = re.compile(r"(?:(?<=[.;:])|(?<=:-))\s+(?=[A-Z0-9])")


def _protect_abbrevs(text: str) -> str:
    out = text
    for a in _ABBREV:
        out = re.sub(
            re.escape(a),
            lambda m: m.group(0).replace(".", "\u2024"),
            out,
            flags=re.I,
        )
    # Initials followed by a name or another initial (e.g. "V. B. KUMAR")
    out = re.sub(
        r"\b([A-Z])\.(?=\s+[A-Za-z])",
        lambda m: m.group(0).replace(".", "\u2024"),
        out,
    )
    # Short abbreviations before a number (e.g. "Guj. 10", "Para. 4")
    out = re.sub(
        r"\b([A-Za-z]{1,5})\.(?=\s*\d+)",
        lambda m: m.group(0).replace(".", "\u2024"),
        out,
    )
    return out


def segment(text: str) -> list[str]:
    """Split a notice into sentences without breaking on common abbreviations."""
    if not text or not text.strip():
        return []
    protected = _protect_abbrevs(text.strip())
    parts = _SENT_SPLIT.split(protected)
    cleaned = []
    for p in parts:
        p = p.replace("\u2024", ".").strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) >= 8:
            cleaned.append(p)
    return cleaned


# --------------------------------------------------------------------------
# Date and period extraction
# --------------------------------------------------------------------------
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

_DATE_PATTERNS = [
    # 31 December 2026 / 31 December
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s*\d{{0,4}}\b", re.I),
    # December 31, 2026
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}},?\s*\d{{0,4}}\b", re.I),
    # 31/12/2026, 31-12-2026, 31.12.2026
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"),
    # 2026-12-31
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # within 30 days / a period of 21 days / up to 6 months
    re.compile(r"\b(?:within|up to|up to a period of|a period of|period of|not later than|latest by)\s+\d+\s+(?:working\s+)?(?:days?|weeks?|months?|years?|hours?)\b", re.I),
    # 30 days from the date of ...
    re.compile(r"\b\d+\s+(?:working\s+)?(?:days?|weeks?|months?|years?)\s+from the date of\b", re.I),
]

_TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:hours|hrs|a\.m\.|p\.m\.)?", re.I)


# A bare "4 may" inside "paragraph 4 may be read as" is a modal verb, not a
# date. Month names that are also ordinary English words (may, march) are the
# usual culprit, so a match is discarded when the words immediately before it
# belong to a cross-reference rather than a date.
_REF_PREFIX = re.compile(
    r"(?:paragraph|para|clause|section|sub-?section|rule|regulation|schedule|"
    r"annexure|form|no\.|item|sl\.?|sr\.?)\s*$",
    re.I,
)


def extract_deadlines(text: str) -> list[dict]:
    """Return the date-like expressions in a notice, in order, de-duplicated."""
    seen = set()
    found = []
    body = text or ""
    for rx in _DATE_PATTERNS:
        for m in rx.finditer(body):
            phrase = m.group(0).strip()
            prefix = body[max(0, m.start() - 22): m.start()]
            if _REF_PREFIX.search(prefix):
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({"phrase": phrase, "position": m.start()})
    for m in _TIME_RE.finditer(text or ""):
        phrase = m.group(0).strip()
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        found.append({"phrase": phrase, "position": m.start()})
    found.sort(key=lambda d: d["position"])
    for d in found:
        d.pop("position", None)
    return found


# --------------------------------------------------------------------------
# Safety screen: is this plausibly an official notice, or a lookalike?
# --------------------------------------------------------------------------
_BAIT_PATTERNS = [
    (re.compile(r"\b(?:otp|one[- ]time password|cvv|pin)\b", re.I), "Asks for an OTP, PIN or CVV"),
    (re.compile(r"(?:bit\.ly|tinyurl|t\.co|rb\.gy|is\.gd|shorturl)", re.I), "Uses a shortened link that hides its destination"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "Links to a raw IP address instead of a domain"),
    (re.compile(r"\.(?:top|xyz|icu|online|club|work|click|buzz)\b", re.I), "Links to a cheap disposable domain"),
    (re.compile(r"\b(?:upi|gpay|phonepe|paytm)\b.{0,40}\b(?:refund|verify|release)\b", re.I), "Asks for a UPI payment to release something"),
    (re.compile(r"\b(?:pay|remit|transfer)\b.{0,30}\b(?:rs\.?|₹)\s?\d", re.I), "Asks for a payment to a person rather than a challan"),
    (re.compile(r"\b(?:whatsapp|telegram)\b.{0,30}\b(?:send|share|contact|join)\b", re.I), "Pushes the conversation to WhatsApp or Telegram"),
    (re.compile(r"\b(?:blocked|block|deactivat\w+|suspend\w+)\b.{0,25}\b(?:today|immediately|within 24 hours)\b", re.I), "Threatens immediate blocking unless you act"),
]

_GENUINE_PATTERNS = [
    (re.compile(r"\.(?:gov\.in|nic\.in)\b", re.I), "Links to an official .gov.in or .nic.in domain"),
    (re.compile(r"\b(?:official gazette|gazette of india|(?:[a-z]+\s+)?government gazette)\b", re.I), "Refers to the Official Gazette"),
    (re.compile(r"\b(?:ministry of|department of|government of(?:\s+[a-z]+)?|[a-z\s\-]+department)\b", re.I), "Names a government ministry or department"),
    (re.compile(r"\b(?:file no\.?|f\.?no\.?|no\.\s*[a-z0-9_/]+)\b", re.I), "Carries a file or reference number"),
    (re.compile(r"\b(?:hereby notified|notice is hereby given|is hereby directed|hereby appoints|hereby ordered|hereby declares?)\b", re.I), "Uses standard official notification phrasing"),
    (re.compile(r"\b(?:official website|official portal|portal of the department)\b", re.I), "Points to an official website"),
]


def safety_screen(text: str) -> dict:
    """Return a plain-language authenticity screen for a pasted notice.

    This is deliberately conservative and deliberately non-alarmist. It
    flags only concrete, checkable cues, and it says out loud that a clean
    screen is not proof of authenticity.
    """
    bait, genuine = [], []
    for rx, desc in _BAIT_PATTERNS:
        if rx.search(text or ""):
            bait.append(desc)
    for rx, desc in _GENUINE_PATTERNS:
        if rx.search(text or ""):
            genuine.append(desc)

    if bait and not genuine:
        level = "caution"
        headline = "This notice shows several cues that are common in fake notices and none of the cues of a genuine one."
    elif bait and genuine:
        level = "mixed"
        headline = "This notice shows both official cues and a few cues worth checking before you act."
    elif genuine:
        level = "consistent"
        headline = "This notice shows the cues of a genuine official document. That is not a guarantee."
    else:
        level = "unclear"
        headline = "There are no strong cues either way. Check the notice against the official portal before acting."

    return {
        "level": level,
        "headline": headline,
        "genuine_cues": genuine,
        "caution_cues": bait,
        "advice": (
            "A notice that asks for money, an OTP, a UPI transfer, or an app install is not a "
            "normal government notice. Verify the file number and the issuer on the department's "
            "official website, or on the eGazette, before you respond."
        ),
    }


# --------------------------------------------------------------------------
# Routing: which portal or helpline actually helps with this notice
# --------------------------------------------------------------------------
_ROUTE_RULES = [
    (re.compile(r"\b(?:income tax|return of income|assessment year|pan|tds|e-?filing)\b", re.I),
     "Income Tax e-Filing", "https://www.incometax.gov.in/"),
    (re.compile(r"\b(?:aadhaar|uidai|e-?kyc|biometric)\b", re.I),
     "UIDAI / My Aadhaar", "https://uidai.gov.in/"),
    (re.compile(r"\b(?:provident fund|epfo|epf|uan|pension)\b", re.I),
     "EPFO Member Portal", "https://unifiedportal-mem.epfindia.gov.in/memberinterface/"),
    (re.compile(r"\b(?:driving licen[cs]e|learner|vehicle registration|rc book|parivahan)\b", re.I),
     "Parivahan Sewa", "https://parivahan.gov.in/"),
    (re.compile(r"\b(?:ration|food security|public distribution|fair price)\b", re.I),
     "Department of Food and Public Distribution", "https://dfpd.gov.in/"),
    (re.compile(r"\b(?:pm-?kisan|kisan samman|landholding farmer|income support)\b", re.I),
     "PM-KISAN", "https://pmkisan.gov.in/"),
    (re.compile(r"\b(?:gst|goods and services tax)\b", re.I),
     "GST Portal", "https://www.gst.gov.in/"),
    (re.compile(r"\b(?:passport)\b", re.I),
     "Passport Seva", "https://www.passportindia.gov.in/"),
    (re.compile(r"\b(?:municipal|property tax|water bill|electricity|discom)\b", re.I),
     "Your municipal or utility portal", ""),
    (re.compile(r"\b(?:electoral roll|voter|election)\b", re.I),
     "Election Commission of India (voters' services)", "https://voters.eci.gov.in/"),
    (re.compile(r"\b(?:objection|appeal|grievance|representation|hearing)\b", re.I),
     "CPGRAMS (grievance portal)", "https://pgportal.gov.in/"),
]


def routes_for(text: str) -> list[dict]:
    """Suggest concrete portals for the specific subject matter of a notice."""
    out = []
    for rx, name, url in _ROUTE_RULES:
        if rx.search(text or "") and not any(r["name"] == name for r in out):
            out.append({"name": name, "url": url, "why": "matches the subject of this notice"})
    return out


# --------------------------------------------------------------------------
# The agent itself
# --------------------------------------------------------------------------
CAVEAT = (
    "SuchnaSaar sorts the sentences of a notice and shows you what each one asks of you. "
    "It is not legal advice, it does not interpret the law, and it cannot confirm that a "
    "notice is genuine. For a binding reading, check the notice against the issuing "
    "department's official website, the eGazette, or consult the department directly."
)


def analyse(text: str, model, confidence_floor: float = 0.55) -> dict:
    """Run the full pipeline over one notice and return the citizen-facing result.

    `model` must be a fitted classifier exposing predict_proba and classes_.
    """
    sentences = segment(text or "")
    if not sentences:
        return {
            "verdict": "Nothing to analyse.",
            "sentences": [],
            "role_counts": {},
            "actions": [],
            "deadlines": [],
            "scheme": None,
            "routes": [],
            "safety": safety_screen(text or ""),
            "caveat": CAVEAT,
        }

    proba = model.predict_proba(sentences)
    classes = list(model.classes_)
    idx = {c: i for i, c in enumerate(classes)}

    rows = []
    for sent, p in zip(sentences, proba):
        p_ref = float(p[idx["reference"]])
        p_act = float(p[idx["action_required"]])
        p_dl = float(p[idx["deadline"]])
        # The deadline class is the strongest signal of all: a date that is
        # attached to an instruction is the thing a reader most needs to see.
        # A sentence is only called a deadline if the model is reasonably
        # sure, otherwise it is reported with its uncertainty intact.
        best = max([("reference", p_ref), ("action_required", p_act), ("deadline", p_dl)], key=lambda t: t[1])
        role, conf = best
        if conf < confidence_floor:
            role = "uncertain"
        rows.append(
            {
                "sentence": sent,
                "role": role,
                "confidence": round(conf, 3),
                "probabilities": {
                    "reference": round(p_ref, 3),
                    "action_required": round(p_act, 3),
                    "deadline": round(p_dl, 3),
                },
                "signals": features.explain(sent, limit=3),
            }
        )

    counts = {}
    for r in rows:
        counts[r["role"]] = counts.get(r["role"], 0) + 1

    actions = [
        {"n": i + 1, "sentence": r["sentence"], "confidence": r["confidence"]}
        for i, r in enumerate([r for r in rows if r["role"] in ("action_required", "deadline")])
    ]

    deadlines = extract_deadlines(text)
    scheme = data.scheme_for(text)
    routes = routes_for(text)

    n_act = counts.get("action_required", 0)
    n_dl = counts.get("deadline", 0)
    if n_act == 0 and n_dl == 0:
        verdict = "This notice appears to contain no instruction and no deadline. It reads as background information."
    elif n_dl and not n_act:
        verdict = f"This notice sets {n_dl} deadline{'s' if n_dl > 1 else ''} and contains no separate instruction."
    elif n_act and not n_dl:
        verdict = f"This notice asks you to do {n_act} thing{'s' if n_act > 1 else ''}, and sets no explicit deadline."
    else:
        verdict = (
            f"This notice asks you to do {n_act} thing{'s' if n_act > 1 else ''} "
            f"and sets {n_dl} deadline{'s' if n_dl > 1 else ''}."
        )

    return {
        "verdict": verdict,
        "sentences": rows,
        "role_counts": counts,
        "actions": actions,
        "deadlines": deadlines,
        "scheme": scheme,
        "routes": routes,
        "safety": safety_screen(text),
        "caveat": CAVEAT,
        "sentence_count": len(sentences),
        "engineered_signals_available": features.N_ENGINEERED,
    }


def demo() -> None:
    """Run the agent over every bundled notice and print a short summary."""
    import joblib

    model = joblib.load("model.joblib")
    for notice in data.NOTICES:
        result = analyse(notice["body"], model)
        print("-" * 66)
        print(notice["title"])
        print(" ", result["verdict"])
        print("  roles:", result["role_counts"])
        if result["scheme"]:
            print("  scheme:", result["scheme"]["name"], "->", result["scheme"]["portal"])
        if result["deadlines"]:
            print("  dates:", ", ".join(d["phrase"] for d in result["deadlines"][:6]))
        print("  safety:", result["safety"]["level"], "--", result["safety"]["headline"])


if __name__ == "__main__":
    demo()
