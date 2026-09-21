"""Generate 500 synthetic production-support / security-triage tickets.

HOW GROUND TRUTH WORKS -- read this before quoting any accuracy number.

Labels are assigned FIRST and the ticket text is generated FROM the label. That
makes the labels internally consistent and free, but it also means "accuracy"
measures agreement with this generator's own label scheme, not with human
triage practice. A system that reverse-engineers these templates scores well
without being good at triage.

Latency, cost, and schema-conformance results do not depend on label quality at
all, so they survive this caveat intact. Accuracy and ECE do not -- treat them
as a smoke test of relative behaviour, not as evidence about production quality.

To harden the comparison, the generator deliberately injects the failure modes
that Jev's own documentation names as weaknesses
(https://docs.typesafe.ai/model-jaggedness/jev-1.13.md):

  * ambiguous department signal      (~15%)  -- two plausible owners
  * irrelevant log spew as padding   (~20%)  -- "accuracy falls as state grows"
  * negation and scoping words       (~12%)  -- "answers the question you wrote"
  * mixed / relative date formats    (~10%)  -- "reads dates as text"

Usage:
    python -m data.generate_dataset [--n 500] [--seed 7] [--out data/scenarios.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import Scenario, TriageDecision  # noqa: E402

# ---------------------------------------------------------------------------
# Surface fragments, keyed by department
# ---------------------------------------------------------------------------

OPENERS = [
    "Hi team,",
    "Hello support,",
    "Hey,",
    "Filing this on behalf of my team.",
    "Raising a ticket:",
    "",
]

SIGNOFFS = [
    "Thanks,\n{name}\n{role}, {company}",
    "Regards,\n{name} ({company})",
    "-- {name}, {role}",
    "Appreciate any help.\n{name}",
    "",
]

NAMES = ["Dana Whitfield", "Marcus Oyelaran", "Priya Raghunathan", "Tomas Lindqvist",
         "Aiko Tanaka", "Samir Haddad", "Grace Okonkwo", "Ben Feldman",
         "Larissa Costa", "Wei Zhang", "Noor Al-Rashid", "Jonas Petersen"]
ROLES = ["Platform Lead", "CTO", "Head of Ops", "SRE", "Billing Manager",
         "Engineering Manager", "Security Engineer", "Founder", "Data Lead"]
COMPANIES = ["Northwind Logistics", "Petal & Co", "Arclight Media", "Fenwick Health",
             "Kestrel Robotics", "BlueRidge Freight", "Tessellate Labs", "Orrery Fintech"]

DEPT_BODIES: dict[str, list[str]] = {
    "billing": [
        "We were charged {amount} twice on the same invoice ({invoice}). The "
        "second charge appears under the same subscription id.",
        "Our plan was downgraded at renewal but we were still invoiced at the "
        "previous tier. Invoice {invoice} shows {amount} instead of the "
        "quoted amount.",
        "The proration on invoice {invoice} does not match what the checkout "
        "page quoted. We are disputing {amount} of it.",
        "We cancelled seat licences last cycle and the credit never landed. "
        "Invoice {invoice} still bills for all {seats} seats.",
        "Payment method on file keeps getting declined even though the card is "
        "valid. Invoice {invoice} is now marked past due for {amount}.",
        "We need a refund for {amount} on invoice {invoice} -- the usage it "
        "bills for came from a test account we had already deleted.",
    ],
    "infrastructure_outage": [
        "The {region} API endpoint has been returning 503s since {when}. Our "
        "retry queue is backing up at roughly {rate} requests a minute.",
        "p99 latency on /v2/ingest went from 180ms to over {latency}ms in "
        "{region} starting {when}. Nothing changed on our side.",
        "Webhook delivery has stopped entirely for our account in {region}. "
        "Last successful callback was {when}.",
        "We are seeing intermittent connection resets from the {region} "
        "cluster -- about {pct}% of calls since {when}.",
        "Dashboard loads but every query times out after 30s. Started {when}, "
        "affecting all {seats} of our users in {region}.",
        "Batch export jobs have been failing with gateway timeouts since "
        "{when}. {rate} jobs are stuck in the queue.",
    ],
    "account_security": [
        "There are {count} successful logins to our admin account from an IP "
        "range in a country where we have no staff, starting {when}.",
        "One of our API keys was committed to a public repository. It has been "
        "live since {when} and we cannot rotate it from the console.",
        "MFA was disabled on the owner account without any of us doing it. The "
        "audit log shows the change at {when}.",
        "We received a password reset email nobody requested, and the audit "
        "log shows {count} reset attempts since {when}.",
        "A former contractor still has active sessions {when}. Their access "
        "should have been revoked when the contract ended.",
        "Our SSO metadata was modified and now authenticates against a domain "
        "we do not control. Noticed {when}.",
    ],
    "feature_request": [
        "Would it be possible to add a webhook for subscription downgrades? We "
        "currently poll every {rate} minutes to detect them.",
        "We would like SCIM provisioning so we can stop managing our {seats} "
        "seats by hand.",
        "Any plans to expose per-region latency in the dashboard? We build it "
        "ourselves from the export today.",
        "Could the export API support cursor pagination? At our volume the "
        "offset approach gets slow past {rate} rows.",
        "Requesting a sandbox environment that mirrors production config, so we "
        "can test upgrades before rolling to {seats} users.",
        "It would help if audit log retention were configurable beyond the "
        "current window -- we need {count} days for compliance.",
    ],
}

# Urgency cues, indexed by rubric level 0-4.
URGENCY_CUES: list[list[str]] = [
    [  # 0 -- no time pressure
        "No rush at all on this.",
        "Purely for the roadmap, whenever you get to it.",
        "Not blocking anything, just noting it.",
        "Low priority, next quarter is fine.",
    ],
    [  # 1 -- minor inconvenience
        "We have a workaround for now, so this is not blocking.",
        "Mildly annoying but we are coping.",
        "Not urgent -- we can keep going manually in the meantime.",
        "It slows us down a little but nobody is stuck.",
    ],
    [  # 2 -- one workflow blocked
        "This is blocking our nightly reconciliation workflow.",
        "One of our teams cannot complete their process until this is sorted.",
        "We are blocked on this specific flow, though the rest still works.",
        "Our onboarding pipeline is stalled because of it.",
    ],
    [  # 3 -- severe
        "This is affecting revenue and we need it resolved today.",
        "Core workflows are broken and customers are noticing.",
        "We are at risk of missing contractual SLAs on this.",
        "This is serious -- data integrity is in question.",
    ],
    [  # 4 -- critical
        "This is a full production outage. Please treat as P1.",
        "Active incident, all hands on our side. We need someone now.",
        "Critical -- we are losing data every minute this continues.",
        "Emergency. Our whole customer base is affected right now.",
    ],
]

NEGATIONS = [
    "To be clear, this is not a billing question.",
    "This is not a feature request -- the documented behaviour is not what we see.",
    "We are not asking for a refund, only an explanation.",
    "Note that this is not affecting production, only staging.",
    "This is not an outage on our side; we have ruled out our own network.",
]

DATE_PHRASES = [
    "03/04 (DD/MM)", "2026-09-14T22:10Z", "last Tuesday", "the 14th",
    "yesterday around 23:40 local", "Sept 14 or maybe the 15th",
    "two days ago", "09/14/2026",
]

WHENS = [
    "about 40 minutes ago", "since 02:15 UTC", "since yesterday evening",
    "for the last three days", "since the deploy on Friday", "since 2026-09-18",
    "this morning", "since roughly 14:00 local",
]

# Irrelevant padding -- the documented "large state with distractor" failure mode.
LOG_SPEW = [
    "INFO  [scheduler] tick 4471 completed in 12ms; queue depth 0",
    "DEBUG [cache] evicted 219 keys (lru), hit ratio 0.981",
    "INFO  [healthz] all 7 probes green; uptime 14d 6h",
    "DEBUG [pool] resized 8 -> 12 connections, idle 4",
    "INFO  [metrics] flushed 1284 series to collector in 38ms",
    "DEBUG [gc] young collection 3.1ms, heap 412MB/1024MB",
    "INFO  [config] reloaded from disk, 0 diffs applied",
    "DEBUG [tls] session resumed via ticket, cipher TLS_AES_128_GCM_SHA256",
    "INFO  [worker-3] leased 25 tasks, ack deadline 30s",
    "DEBUG [dns] resolved upstream in 4ms (cached)",
]

# Ambiguity pairings: (primary label, secondary signal to mention in passing).
AMBIGUOUS_PAIRS = [
    ("account_security", "billing",
     "We also noticed an unfamiliar charge on the last invoice, which is how "
     "we started looking."),
    ("billing", "account_security",
     "We did also see a login we did not recognise, though the charge is the "
     "main thing."),
    ("infrastructure_outage", "billing",
     "The API is also returning 402 on some calls, so there may be a plan "
     "limit involved."),
    ("infrastructure_outage", "feature_request",
     "Longer term it would be good to have a status webhook for this."),
    ("feature_request", "infrastructure_outage",
     "We hit a few timeouts last week that made us want this more."),
    ("account_security", "infrastructure_outage",
     "There were also some 503s around the same window."),
]

# Urgency priors per department (weights over levels 0-4).
URGENCY_PRIORS: dict[str, list[float]] = {
    "billing":               [0.10, 0.34, 0.34, 0.18, 0.04],
    "infrastructure_outage": [0.02, 0.10, 0.25, 0.38, 0.25],
    "account_security":      [0.03, 0.12, 0.25, 0.35, 0.25],
    "feature_request":       [0.45, 0.38, 0.14, 0.03, 0.00],
}

DEPARTMENTS = list(URGENCY_PRIORS)


def escalation_rule(department: str, urgency_level: int) -> bool:
    """Ground-truth escalation policy.

    Deterministic and stated openly so readers can judge it: escalate on
    severe-or-worse urgency, and one level earlier for security, where the cost
    of waiting is asymmetric.
    """
    if department == "account_security":
        return urgency_level >= 2
    return urgency_level >= 3


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        amount=f"${rng.choice([49, 120, 349, 1200, 4800, 12400]):,}.{rng.randrange(100):02d}",
        invoice=f"INV-{rng.randrange(10000, 99999)}",
        seats=rng.choice([12, 40, 85, 240, 1100]),
        region=rng.choice(["eu-west-1", "us-east-1", "ap-southeast-2", "us-west-2"]),
        when=rng.choice(WHENS),
        rate=rng.choice([40, 250, 900, 5000, 18000]),
        latency=rng.choice([900, 2400, 7800, 15000]),
        pct=rng.choice([3, 12, 35, 60]),
        count=rng.choice([2, 7, 19, 64, 300]),
        name=rng.choice(NAMES),
        role=rng.choice(ROLES),
        company=rng.choice(COMPANIES),
    )


def build_scenario(idx: int, rng: random.Random) -> Scenario:
    # --- 1. choose the label first -----------------------------------------
    ambiguous = rng.random() < 0.15
    if ambiguous:
        department, _secondary, aside = rng.choice(AMBIGUOUS_PAIRS)
    else:
        department = DEPARTMENTS[idx % len(DEPARTMENTS)]
        aside = None

    levels = [0, 1, 2, 3, 4]
    urgency_level = rng.choices(levels, weights=URGENCY_PRIORS[department], k=1)[0]
    escalate = escalation_rule(department, urgency_level)

    truth = TriageDecision(
        department=department,          # type: ignore[arg-type]
        urgency_level=urgency_level,    # type: ignore[arg-type]
        escalate_supervisor=escalate,
    )

    # --- 2. generate surface text from the label ---------------------------
    parts: list[str] = []
    opener = rng.choice(OPENERS)
    if opener:
        parts.append(opener)

    parts.append(_fill(rng.choice(DEPT_BODIES[department]), rng))
    parts.append(rng.choice(URGENCY_CUES[urgency_level]))

    if aside:
        parts.append(aside)

    has_negation = rng.random() < 0.12
    if has_negation:
        parts.append(rng.choice(NEGATIONS))

    has_mixed_dates = rng.random() < 0.10
    if has_mixed_dates:
        d1, d2 = rng.sample(DATE_PHRASES, 2)
        parts.append(f"First seen {d1}, and again {d2}.")

    distractor_chars = 0
    if rng.random() < 0.20:
        n_lines = rng.randrange(6, 18)
        spew = "\n".join(rng.choice(LOG_SPEW) for _ in range(n_lines))
        block = f"\nAttaching the tail of our application log in case it helps:\n{spew}\n"
        parts.append(block)
        distractor_chars = len(block)

    signoff = rng.choice(SIGNOFFS)
    if signoff:
        parts.append(_fill(signoff, rng))

    return Scenario(
        id=f"tkt-{idx:04d}",
        ticket="\n\n".join(parts).strip(),
        truth=truth,
        ambiguous=ambiguous,
        distractor_chars=distractor_chars,
        has_negation=has_negation,
        has_mixed_dates=has_mixed_dates,
    )


def generate(n: int, seed: int) -> list[Scenario]:
    rng = random.Random(seed)
    # Shuffle the department cycle so ordering carries no signal.
    order = list(range(n))
    rng.shuffle(order)
    return [build_scenario(i, rng) for i in order]


def write_dataset(scenarios: list[Scenario], out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        s.model_dump_json(exclude_none=False) + "\n"
        for s in sorted(scenarios, key=lambda s: s.id)
    )
    # newline="\n" is load-bearing, not style: without it Windows translates the
    # "\n" separators to "\r\n" on the way out, so the bytes on disk stop matching
    # the string we hash here -- and `dataset_sha256()`, which hashes the file,
    # would disagree with the digest recorded in the manifest. The dataset is
    # identified by a content hash, so it has to be byte-identical everywhere.
    out.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    manifest = {
        "path": out.name,
        "n": len(scenarios),
        "sha256": digest,
        "label_provenance": "labels assigned first; ticket text generated from label",
        "difficulty_mix": {
            "ambiguous": sum(s.ambiguous for s in scenarios),
            "with_distractors": sum(s.distractor_chars > 0 for s in scenarios),
            "with_negation": sum(s.has_negation for s in scenarios),
            "with_mixed_dates": sum(s.has_mixed_dates for s in scenarios),
        },
        "department_counts": {
            d: sum(s.truth.department == d for s in scenarios) for d in DEPARTMENTS
        },
        "urgency_counts": {
            str(lv): sum(s.truth.urgency_level == lv for s in scenarios) for lv in range(5)
        },
        "escalate_rate": round(
            sum(s.truth.escalate_supervisor for s in scenarios) / len(scenarios), 4
        ),
    }
    (out.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return digest


def load_dataset(path: Path) -> list[Scenario]:
    with path.open(encoding="utf-8") as fh:
        return [Scenario.model_validate_json(line) for line in fh if line.strip()]


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "scenarios.jsonl")
    args = ap.parse_args()

    scenarios = generate(args.n, args.seed)
    digest = write_dataset(scenarios, args.out)

    print(f"wrote {len(scenarios)} scenarios -> {args.out}")
    print(f"sha256 {digest}")
    print(f"manifest -> {args.out.parent / 'manifest.json'}")


if __name__ == "__main__":
    main()
