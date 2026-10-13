<div align="center">

# SuchnaSaar

### Read a government notice. See what it actually asks you to do.

SuchnaSaar is an AI-powered web app that helps citizens read Indian government notices. Paste the text of a gazette notification, circular, office memorandum or public notice to get a sentence-level breakdown of what is background, what is an instruction, and what carries a deadline — plus the dates, the scheme, and the official portal to go to.

</div>

---

## What SuchnaSaar does

- Splits a notice into sentences and labels each one **reference**, **action required** or **deadline**.
- Builds a numbered checklist, quoting each obligation verbatim — it never paraphrases a requirement.
- Extracts every date, period and time in order of appearance.
- Identifies the scheme the notice concerns and links to its official portal.
- Routes the subject matter to the correct official portal or helpline.
- Screens the text for the cues that separate a genuine notice from a lookalike.

SuchnaSaar is decision support, not legal advice. It sorts and quotes what a notice says; it does not interpret the law and cannot confirm that a notice is genuine.

## Try it

1. Start the app (below) and open [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Paste the text of a notice, or click one of the five bundled sample notices.
3. Read the verdict, the checklist, the dates, the routing and the authenticity screen.

## How it works

The web app is built with Flask. A sentence-level text classifier combines word and character n-grams with 41 engineered modality signals. The `/analyze` endpoint returns the role of every sentence together with a checklist, dates, scheme, routes and an authenticity screen; `/health` reports whether the model is loaded.

Submitted text is processed in memory to produce one response. The app writes it nowhere: no database, no log, no file, and no outbound network call.

## Run locally

You'll need Python and pip.

```bash
python -m venv .venv
```

Activate the virtual environment, then install the runtime dependencies and start the app:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux: use this activation command instead
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Reproduce the evaluation

Every number in the concept note and the deck comes out of one command:

```bash
python train.py     # writes model.joblib and metrics.json
python agent.py     # runs the agent over the five bundled notices
```

`train.py` reports three evaluations: 5-fold stratified cross-validation, a held-out probe carved out of the corpus, and a hard out-of-sample probe of deliberately ambiguous sentences. All three are reported even when they are unflattering.

## API

Analyse a notice with a JSON `POST` request:

```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Beneficiaries are directed to link their bank account with the Aadhaar number. The last date for the linking of the Aadhaar number with the bank account is 30 April."}'
```

The endpoint returns the verdict, a per-sentence breakdown with probabilities, the checklist, the dates found, the scheme, the routes and the authenticity screen.

- `GET /sample/<id>` returns one of the five bundled sample notices.
- `GET /health` reports the model status.

## Project structure

| File or folder | Purpose |
| --- | --- |
| `app.py` | Flask web app and API routes |
| `templates/index.html` | Web interface |
| `model.joblib` | Trained sentence-role classifier |
| `features.py` | 41 engineered notice signals |
| `agent.py` | Checklist, date extraction, scheme identification, routing, authenticity screen |
| `models.py` | Candidate models and why each exists |
| `data.py` | Corpus, sample notices, verified portals, helplines and scheme facts |
| `dataset.py` | Splitting, and the hard out-of-sample probe |
| `train.py` | Training and the three-part evaluation |
| `metrics.json` | Every number quoted in the report and the deck |
| `requirements.txt` | Runtime Python dependencies |

## Corpus

The corpus is 207 sentences written for this project in the register of Indian official drafting. **They imitate the style of gazette notifications, circulars and memoranda; they are not quoted from any gazette.** A corpus of imitated drafting has to be honest about what it is.

Every scheme fact, portal and helpline number in `data.py` is real and is recorded together with the official domain it was verified against. Nothing was invented; where a fact could not be verified against an official source it was left out rather than guessed.

## Help improve SuchnaSaar

Bug reports and suggestions are welcome. The most useful contribution is a real notice, redacted, together with the sentence you would have acted on.
