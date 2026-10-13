"""Smoke test: does the Flask app actually serve and analyse a notice?

Runs the real app in-process with Flask's test client, so no port is bound
and no server is left running. Verifies the four routes and checks that a
sample notice produces the structure the interface expects.
"""

from __future__ import annotations

import json
import sys

import app as application

client = application.app.test_client()
failures = []


def check(name, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


# ------------------------------------------------------------------ /health
r = client.get("/health")
check("GET /health returns 200", r.status_code == 200, str(r.status_code))
h = r.get_json()
check("model is loaded", h.get("model_error") is None, str(h.get("model_error")))
check("health reports no storage of text", h.get("stores_submitted_text") is False)
check("health reports no outbound calls", h.get("outbound_network_calls") is False)

# ------------------------------------------------------------------ /
r = client.get("/")
check("GET / returns 200", r.status_code == 200, str(r.status_code))
html = r.get_data(as_text=True)
check("index renders the tool name", "SuchnaSaar" in html)
check("index lists the sample notices", "N1" in html and "N5" in html)
check("index shows a real metric from metrics.json", "corpus" in html)
check("index lists verified portals", "pgportal.gov.in" in html)
check("index lists the consumer helpline", "1915" in html)

# --------------------------------------------------------------- /sample/N1
r = client.get("/sample/N1")
check("GET /sample/N1 returns 200", r.status_code == 200, str(r.status_code))
s1 = r.get_json()
check("sample N1 carries body text", len(s1.get("text", "")) > 300, f"{len(s1.get('text',''))} chars")
r = client.get("/sample/ZZ")
check("GET /sample/ZZ returns 404", r.status_code == 404, str(r.status_code))

# ---------------------------------------------------------------- /analyze
payload = {
    "text": (
        "The Scheme was launched to provide income support to landholding farmer families. "
        "Beneficiaries are directed to link their bank account with the Aadhaar number. "
        "The last date for the linking of the Aadhaar number with the bank account is 30 April. "
        "This is for information of all concerned."
    )
}
r = client.post("/analyze", json=payload)
check("POST /analyze returns 200", r.status_code == 200, str(r.status_code))
d = r.get_json()
for key in ("verdict", "sentences", "role_counts", "actions", "deadlines", "routes", "safety", "caveat"):
    check(f"response contains '{key}'", key in d)
check("verdict mentions the counts", "deadline" in d.get("verdict", "") or "ask" in d.get("verdict", ""), d.get("verdict", ""))
check("a deadline was extracted", any("30 April" in x["phrase"] for x in d.get("deadlines", [])),
      json.dumps(d.get("deadlines", [])))
check("the Aadhaar action appears in the checklist", any("Aadhaar" in a["sentence"] for a in d.get("actions", [])))
check("the reference line is not in the checklist", not any("for information of all concerned" in a["sentence"] for a in d.get("actions", [])))
check("every sentence carries three probabilities",
      all(set(x["probabilities"]) == {"reference", "action_required", "deadline"} for x in d.get("sentences", [])))
check("the caveat is present and non-empty", len(d.get("caveat", "")) > 60)
check("the scheme was identified", d.get("scheme") is not None, str(d.get("scheme", {}).get("name") if d.get("scheme") else None))

# ------------------------------------------------------- input validation
r = client.post("/analyze", json={"text": "too short"})
check("short input is rejected with 400", r.status_code == 400, str(r.status_code))
r = client.post("/analyze", data="not json", content_type="text/plain")
check("non-JSON body is rejected with 400", r.status_code == 400, str(r.status_code))

# ---------------------------------------------- every sample through the app
for n in application.data.NOTICES:
    r = client.post("/analyze", json={"text": n["body"]})
    ok = r.status_code == 200
    dd = r.get_json() if ok else {}
    check(f"sample {n['id']} analyses without error",
          ok and dd.get("sentence_count", 0) >= 8,
          f"{dd.get('sentence_count')} sentences, safety={dd.get('safety',{}).get('level')}")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("all smoke tests passed")
