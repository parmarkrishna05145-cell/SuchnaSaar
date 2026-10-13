"""Engineered notice signals for SuchnaSaar.

A government notice cannot be scored on vocabulary alone. The same words
("shall", "may", "within") appear in a legally binding instruction and in a
paragraph of pure background. What separates the two is *modality* and
*document role*: an instruction tells a named actor to do something by a
date, while a preamble explains why the document exists at all.

The signals below try to capture that difference explicitly, so the model
does not have to rediscover legal drafting from n-grams alone. Every signal
is a plain, auditable rule -- if the model changes its mind about a notice,
a human can read this file and see exactly which cues were available.

The module exposes three things:

    N_ENGINEERED     the number of engineered signals
    transform(texts) the numeric matrix the model consumes
    explain(text)    a human-readable account of which signals fired

Nothing here stores or transmits the text it is given.
"""

from __future__ import annotations

import math
import re

# --------------------------------------------------------------------------
# Signal specifications
# --------------------------------------------------------------------------
# Each entry is (name, one-line description, compiled pattern). The pattern is
# searched case-insensitively; a hit contributes 1.0 to that column.

_PATTERNS = [
    # -- modality: does the sentence tell someone to do something? ----------
    (
        "sig_imperative",
        "Uses directive language (shall / must / is required to / are directed to)",
        r"\b(shall|must|is required to|are required to|are directed to|is directed to"
        r"|are advised to|is hereby directed|needs to|has to)\b",
    ),
    (
        "sig_deadline",
        "Sets a deadline or cut-off date",
        r"\b(on or before|last date|due date|cut-?off|within\s+\d+\s+(?:days|weeks|months|years)"
        r"|not later than|latest by|time limit|deadline|before the (?:due|last) date"
        r"|upto|up to \d|till \d)\b",
    ),
    (
        "sig_obligation_negative",
        "Imposes a prohibition ('no person shall ...')",
        r"\b(no person shall|no one shall|shall not be|is prohibited|are prohibited"
        r"|must not|shall be prohibited|shall not (?:be )?(?:allowed|permitted))\b",
    ),
    (
        "sig_negation_of_action",
        "Explicitly tells the reader that no action is needed",
        r"\b(no action is required|no further action|nothing further is required"
        r"|no reply is required|no response is necessary|may be ignored"
        r"|for record only|for information only)\b",
    ),
    # -- the concrete asks --------------------------------------------------
    (
        "sig_document_req",
        "Requires documents to be produced, uploaded or enclosed",
        r"\b(submit|furnish|attach|upload|produce|enclose|self[- ]attested"
        r"|along ?with|supporting document\w*|copy of|affidavit|declaration)\b",
    ),
    (
        "sig_application_action",
        "Asks the reader to apply, register, enrol or renew",
        r"\b(apply|application|register|registration|enrol\w*|renew\w*|re-?apply"
        r"|opt[- ]in|sign up|fill (?:up|in) the form)\b",
    ),
    (
        "sig_update_action",
        "Asks the reader to update, link, correct or verify a record",
        r"\b(update|link(?:ing)?|seed(?:ing)?|correct\w*|rectif\w*|verif\w*"
        r"|e-?kyc|kyc|re-?verification)\b",
    ),
    (
        "sig_payment",
        "Involves a fee, payment, deposit or challan",
        r"(₹|rs\.?\s?\d|inr\s?\d)|\b(fee|fees|payment|pay\b|remit|deposit|challan"
        r"|demand draft|penalty amount)\b",
    ),
    (
        "sig_penalty",
        "Carries a stated consequence for non-compliance",
        r"\b(penalt\w+|fine\w*|liable to|prosecut\w+|shall be cancelled"
        r"|shall be rejected|shall be withdrawn|forfeit\w*|action will be taken"
        r"|deemed to have been)\b",
    ),
    (
        "sig_certificate_action",
        "Requires a certificate, licence, permit or approval",
        r"\b(certificate|licence|license|permit|approval|clearance"
        r"|no objection certificate|noc|authoris\w+|sanction)\b",
    ),
    (
        "sig_eligibility",
        "States who is eligible or entitled to a benefit",
        r"\b(eligib\w+|entitled|beneficiar\w+|qualif\w+|who can apply"
        r"|not eligible|ineligible)\b",
    ),
    (
        "sig_objection",
        "Invites objections, claims, representations or appeals",
        r"\b(objection\w*|representation\w*|claim\w*|appeal\w*|grievance\w*"
        r"|comments? (?:are|may be) invited|invite\w* objections|hearing)\b",
    ),
    (
        "sig_inspection",
        "Provides for inspection, audit or physical verification",
        r"\b(inspect\w*|audit\w*|site visit|physical verification|survey"
        r"|enquir\w+|scrutiny)\b",
    ),
    (
        "sig_portal_link",
        "Points to a website or online portal",
        r"https?://|www\.|\b(portal|website|online\s+(?:portal|application)"
        r"|web\s*form|online mode)\b",
    ),
    (
        "sig_helpline",
        "Gives a helpline or a contact channel",
        r"\b(helpline|toll[- ]free|help ?line|call cent(?:re|er)"
        r"|contact\s+\d|1800[\s-]?\d|1915|1930|155260|14416|1967)\b",
    ),
    # -- document role: is this sentence doing legal work or background? ----
    (
        "sig_legal_boilerplate",
        "Legal drafting boilerplate rather than an instruction",
        r"\b(in exercise of the powers conferred|whereas|hereinafter|aforesaid"
        r"|notwithstanding|pursuant to|in supersession of|inter alia"
        r"|mutatis mutandis|ab initio|vide\b|in the matter of)\b",
    ),
    (
        "sig_preamble",
        "Announces the document rather than instructing anyone",
        r"\b(is hereby notified|hereby notified|it is hereby|notice is hereby given"
        r"|for information of|the following .{0,40}are hereby)\b",
    ),
    (
        "sig_effective_date",
        "States when a rule takes effect",
        r"\b(with effect from|shall come into force|shall take effect"
        r"|w\.?e\.?f\.?|from the date of (?:its )?publication)\b",
    ),
    (
        "sig_scope_objective",
        "States the objective or background of a scheme",
        r"\b(with the objective of|in order to|aims? to|purpose of this"
        r"|vision of|keeping in view|with a view to)\b",
    ),
    (
        "sig_historical",
        "Reports something that already happened, rather than instructing",
        r"\b(has been|have been|had been|was launched|were issued|has already"
        r"|was introduced|has approved|has sanctioned|earlier issued"
        r"|had approved|was notified)\b",
    ),
    (
        "sig_authority",
        "Names a government authority as the issuing body",
        r"\b(ministry of|department of|government of|authority|commission(?:er)?"
        r"|directorate|municipal|panchayat|collector|tehsildar|registrar|board)\b",
    ),
    (
        "sig_procedural",
        "Describes internal procedure (meetings, quorum, records)",
        r"\b(quorum|convened|meeting of the|minutes of|record of proceedings"
        r"|shall be placed before|laid on the table)\b",
    ),
    (
        "sig_extension",
        "Announces an extension of time",
        r"\b(extend\w*|extension|further period|extended till|extended up ?to"
        r"|time is extended)\b",
    ),
    (
        "sig_correction",
        "Is a corrigendum, amendment or errata",
        r"\b(corrigendum|corrigenda|errata|amendment|amended|stands modified"
        r"|read as|substitut\w+ for)\b",
    ),
    (
        "sig_exemption",
        "Deals with an exemption, waiver or relaxation",
        r"\b(exempt\w*|waiv\w*|relax\w*|exemption|shall not apply"
        r"|not applicable to|outside the purview)\b",
    ),
    (
        "sig_benefit_amount",
        "States an amount of money a citizen receives or pays",
        r"(₹\s?\d|rs\.?\s?\d)|\b(amount of|assistance of|subsidy"
        r"|financial assistance|income support|instal?l?ment|grant of)\b",
    ),
    (
        "sig_public_notice",
        "Addresses the general public as an audience",
        r"\b(public notice|general public|all concerned|all stakeholders"
        r"|for information of all|public at large|attention of|citz?ens are)\b",
    ),
    (
        "sig_form_reference",
        "Refers to a numbered form, annexure or schedule",
        r"\b(form\s*(?:no\.?|number)?\s*[a-z0-9\-]+|annexure|appendix"
        r"|schedule\s+[ivx\d]+|proforma)\b",
    ),
    (
        "sig_section_ref",
        "Cites a section, rule or clause number",
        r"\b(section\s+\d+|sub-?section|clause\s+\d+|rule\s+\d+"
        r"|paragraph\s+\d+|regulation\s+\d+)\b",
    ),
    (
        "sig_percentage",
        "Contains a percentage or a rate",
        r"\d+(?:\.\d+)?\s?(?:%|per ?cent)|\bpro ?rata\b",
    ),
]

_COMPILED = [(n, d, re.compile(p, re.I)) for n, d, p in _PATTERNS]

# Numeric (non-keyword) signals, appended after the keyword signals.
_NUMERIC_NAMES = [
    "n_urls",
    "has_gov_domain",
    "n_dates",
    "has_currency",
    "digit_ratio",
    "upper_ratio",
    "char_len_log",
    "word_count_log",
    "is_heading_like",
    "has_all_caps_word",
    "starts_with_number",
]

N_ENGINEERED = len(_COMPILED) + len(_NUMERIC_NAMES)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_GOV_DOMAIN_RE = re.compile(r"\.(?:gov\.in|nic\.in|gov|nic)\b", re.I)
_CURRENCY_RE = re.compile(r"(?:₹|rs\.?|inr)\s?\d", re.I)
_DATE_RE = re.compile(
    r"(?:\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*,?\s*\d{2,4}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}\s+(?:days|weeks|months|years)\b)",
    re.I,
)


def names() -> list[str]:
    """Return every engineered signal name, in matrix-column order."""
    return [n for n, _, _ in _COMPILED] + list(_NUMERIC_NAMES)


def descriptions() -> dict[str, str]:
    """Return a name -> plain-English description map for reporting."""
    d = {n: desc for n, desc, _ in _COMPILED}
    d.update(
        {
            "n_urls": "Number of links in the sentence",
            "has_gov_domain": "A link points to an official .gov.in / .nic.in domain",
            "n_dates": "Number of date-like or duration expressions",
            "has_currency": "Mentions a rupee amount",
            "digit_ratio": "Proportion of characters that are digits",
            "upper_ratio": "Proportion of characters that are uppercase",
            "char_len_log": "Sentence length (log characters)",
            "word_count_log": "Sentence length (log words)",
            "is_heading_like": "Short, unpunctuated line that reads as a heading",
            "has_all_caps_word": "Contains a fully capitalised word (form / acronym)",
            "starts_with_number": "Begins with a numbered-list marker",
        }
    )
    return d


def _row(text: str) -> list[float]:
    t = text or ""
    row = [1.0 if rx.search(t) else 0.0 for _, _, rx in _COMPILED]

    urls = _URL_RE.findall(t)
    words = t.split()
    n_chars = max(len(t), 1)
    n_words = max(len(words), 1)
    digits = sum(c.isdigit() for c in t)
    uppers = sum(c.isupper() for c in t)

    stripped = t.strip()
    heading_like = (
        len(stripped) < 90
        and not stripped.endswith((".", ";", ":"))
        and len(words) <= 12
        and bool(stripped)
    )

    row.extend(
        [
            float(len(urls)),
            1.0 if _GOV_DOMAIN_RE.search(t) else 0.0,
            float(len(_DATE_RE.findall(t))),
            1.0 if _CURRENCY_RE.search(t) else 0.0,
            digits / n_chars,
            uppers / n_chars,
            math.log1p(len(t)),
            math.log1p(n_words),
            1.0 if heading_like else 0.0,
            1.0 if any(w.isupper() and len(w) >= 3 and w.isalpha() for w in words) else 0.0,
            1.0 if re.match(r"^\s*(?:\d+[.)]|\([a-z0-9]+\)|[ivx]+[.)])\s", stripped, re.I) else 0.0,
        ]
    )
    return row


def transform(texts) -> list[list[float]]:
    """Turn an iterable of sentences into the engineered-signal matrix."""
    if isinstance(texts, str):
        texts = [texts]
    return [_row(t) for t in texts]


def explain(text: str, limit: int = 6) -> list[dict]:
    """Return the signals that fired, strongest first, with plain descriptions.

    Only signals with a non-zero value are returned, because a list of
    everything that did *not* fire teaches the reader nothing.
    """
    row = _row(text)
    desc = descriptions()
    fired = []
    for name, value in zip(names(), row):
        if value:
            fired.append(
                {
                    "signal": name,
                    "description": desc.get(name, name),
                    "value": round(float(value), 4),
                }
            )
    # Keyword signals are more legible to a non-technical reader than the
    # numeric ones, so show them first when both fire.
    fired.sort(key=lambda s: (not s["signal"].startswith("sig_"), -s["value"]))
    return fired[:limit]
