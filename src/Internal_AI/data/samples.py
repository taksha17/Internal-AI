"""Synthetic corpus generator used to exercise the data plane end to end.

The platform ships without company data by design (raw data never leaves the
boundary). These generators stand in for pilot sources so the pipeline,
tokenizer, training and evaluation paths can be run and tested anywhere.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PRODUCTS = [
    "invoice reconciliation",
    "shipment tracking",
    "customer onboarding",
    "warehouse audit",
    "payment retries",
    "returns processing",
    "license renewal",
    "inventory reconciliation",
]

INTERNAL_JARGON = [
    "ledgerline",
    "backorderqueue",
    "fulfilmenthub",
    "reconciliationdelta",
    "tenantquotas",
]

TERM_SYNONYMS: dict[str, str] = {
    "ledgerline": "internal ledger feed",
    "backorderqueue": "pending order backlog",
    "fulfilmenthub": "central fulfilment service",
    "reconciliationdelta": "mismatch between books",
    "tenantquotas": "per tenant usage limits",
    "invoice reconciliation": "invoice matching",
    "shipment tracking": "parcel delivery status",
    "customer onboarding": "new account setup",
    "warehouse audit": "stocktake review",
    "payment retries": "repeated charge attempts",
    "returns processing": "refund handling",
    "license renewal": "entitlement extension",
    "inventory reconciliation": "stock matching",
}

TICKET_TEMPLATES = [
    (
        "Customer reports that {product} failed for account {account}. The error returned was "
        "'ECONNRESET while calling ledgerline service'. Escalated to tier two. Reference {ticket}."
    ),
    (
        "Follow-up on {ticket}: batch job for {product} completed with 42 warnings. Requesting "
        "review of reconciliationdelta entries before month-end close."
    ),
    (
        "Request from finance: export {product} totals for {account} in CSV, grouped by region and "
        "fulfilmenthub, for the quarterly reconciliation report. Due Friday. Ticket {ticket}."
    ),
    (
        "Customer {account} cannot access the fulfilmenthub dashboard since the "
        "tenantquotas change. Screenshots attached in {ticket}."
    ),
    (
        "Backorderqueue depth for {product} exceeded threshold twice this week. Root cause appears "
        "to be stale ledgerline data from the upstream warehouse. See {ticket}."
    ),
]

DOC_TEMPLATES = [
    (
        "Runbook: {product} failures\n\n"
        "When {product} reports a reconciliationdelta mismatch, first check the ledgerline feed, "
        "then confirm tenantquotas for the account, then re-run the warehouse audit job. If the "
        "backorderqueue is deep, escalate to the platform on-call. Reference {ticket}."
    ),
    (
        "Architecture note: {product} pipeline\n\n"
        "The {product} pipeline ingests from the warehouse, normalizes into the fulfilmenthub, and "
        "writes ledgerline entries. It is scheduled per tenant with quotas applied per account. "
        "Any schema change requires a versioned migration and a data quality report."
    ),
    (
        "Customer support macro: {product}\n\n"
        "We have escalated your {product} request, reference {ticket}. Our team checks the "
        "ledgerline daily and will update the fulfilmenthub status. No action is needed from you "
        "at this time."
    ),
]

CODE_TEMPLATES = [
    (
        "def reconcile_ledgerline(account_id, feed):\n"
        "    delta = {{}}\n"
        "    for entry in feed.entries(account_id):\n"
        "        key = entry.region\n"
        "        delta[key] = delta.get(key, 0) + entry.amount\n"
        "    return {{k: v for k, v in delta.items() if v != 0}}\n"
    ),
    (
        "class BackorderQueue:\n"
        "    def __init__(self, tenant):\n"
        "        self.tenant = tenant\n"
        "        self.depth = 0\n"
        "\n"
        "    def push(self, order_id):\n"
        "        self.depth += 1\n"
        "        return self.depth\n"
    ),
]


OPENING_SENTENCES = [
    "Customer reports that {product} failed for account {account} on {region} infrastructure.",
    "The {team} team escalated {product} after the {region} batch job stalled on {account}.",
    "Finance asked for a {quarter} breakdown of {product} for account {account} across {region}.",
    "A {region} tenant hit the request ceiling during {product} and the queue backed up overnight.",
    "Support logged {ticket} because {product} showed a mismatch on account {account}.",
    "The {team} on-call noticed {product} reconciliation drifting in the {region} cluster.",
    "Weekly review flagged {product} as the slowest path for account {account} in {region}.",
    "Customer {account} asked why {product} shows a stale balance in the {region} dashboard.",
]

BODY_SENTENCES = [
    "The error was 'ECONNRESET while calling ledgerline service' and the retry budget was spent.",
    "Backorderqueue depth rose to {count} items before the upstream warehouse caught up.",
    "The fulfilmenthub still shows the previous tenantquotas for this tenant.",
    "Root cause appears to be stale ledgerline data from the upstream warehouse feed.",
    "A reconciliationdelta of {amount} remained open after the nightly close.",
    "The {team} team confirmed the tenantquotas change was applied to the wrong region.",
    "Two invoices were written to the wrong ledgerline bucket during the {quarter} migration.",
    "The customer asked for a credit note once the warehouse audit completed successfully.",
    "Payment retries for account {account} exhausted the daily allowance twice.",
    "The runbook for {product} does not yet cover the {region} failover path.",
    "Returns processing continued while shipment tracking was unavailable for {region}.",
    "Customer onboarding stalled at the licence renewal step for account {account}.",
    "The fulfilmenthub backlog was cleared after two warehouse audit cycles.",
    "Shipping labels were regenerated for {count} orders during the {quarter} close.",
    "An engineer checked the fulfilmenthub tenantquotas before escalating to the platform team.",
    "The {team} team rebuilt the backorderqueue index and reduced the wait to {count} seconds.",
    "Invoice reconciliation flagged {count} shipments whose labels no longer matched.",
    "A support macro was updated to explain the {region} outage to account {account}.",
]

CLOSING_SENTENCES = [
    "Reference {ticket}; please confirm the resolution with the {team} team.",
    "Screenshots and the ledgerline extract are attached to {ticket}.",
    "Escalated to tier two under {ticket} with a request for a written explanation.",
    "Follow the {product} runbook and re-run the warehouse audit before replying.",
    "The customer was told to expect an update within one business day. Case {ticket}.",
    "No customer data was copied outside the platform while handling {ticket}.",
]

TEAMS = ["platform", "support", "finance", "fulfilment", "data", "reliability"]
REGIONS = ["emea", "amer", "apac", "nordics", "anz", "latam"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


def generate_corpus(
    path: Path,
    records: int = 120,
    duplicate_ratio: float = 0.15,
    pii_ratio: float = 0.2,
    seed: int = 7,
) -> dict[str, object]:
    """Write a synthetic jsonl corpus with duplicates and PII.

    Documents are composed from varied sentence pools rather than one template,
    so the corpus holds genuinely distinct documents. A corpus of near
    identical templates collapses the embedding space and makes every
    downstream metric meaningless.
    """
    generator = random.Random(seed)
    rows: list[dict[str, object]] = []
    ticket_number = 1000
    for index in range(records):
        product = generator.choice(PRODUCTS)
        ticket_number += generator.randint(1, 7)
        slots = {
            "product": product,
            "account": f"acct-{generator.randint(1000, 9999)}",
            "ticket": f"TSK-{ticket_number}",
            "team": generator.choice(TEAMS),
            "region": generator.choice(REGIONS),
            "quarter": generator.choice(QUARTERS),
            "count": generator.randint(3, 480),
            "amount": f"{generator.randint(120, 98000)}.{generator.randint(10, 99)}",
        }
        sentences = [generator.choice(OPENING_SENTENCES).format(**slots)]
        body = generator.sample(BODY_SENTENCES, generator.randint(2, 5))
        if generator.random() < 0.6:
            body.append(generator.choice(BODY_SENTENCES).format(**slots))
        sentences.extend(sentence.format(**slots) for sentence in body)
        if generator.random() < 0.75:
            sentences.append(generator.choice(CLOSING_SENTENCES).format(**slots))
        generator.shuffle(sentences)
        text = " ".join(sentences)
        if generator.random() < pii_ratio:
            text += (
                f" Contact the requester directly at user{index}@example.com or "
                f"+1 415 555 {index % 10000:04d}."
            )
        if generator.random() < 0.1:
            text += (
                f" Internal key AKIAIOSFODNN7EXAMPLE was used by the batch job for {ticket_number}."
            )
        sensitivity = "internal"
        if generator.random() < 0.08:
            sensitivity = "customer_confidential"
        rows.append(
            {
                "id": f"rec-{index:04d}",
                "text": text,
                "sensitivity": sensitivity,
                "acl": ["support", "engineering"] if sensitivity == "internal" else ["support"],
            }
        )
    duplicates = int(records * duplicate_ratio)
    for index in range(duplicates):
        source = rows[generator.randrange(len(rows))]
        rows.append({**source, "id": f"dup-{index:04d}"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {"path": str(path), "records": len(rows), "duplicates_injected": duplicates}


def generate_eval_pairs(path: Path, records: int = 24, seed: int = 11) -> dict[str, object]:
    """Write a query-document relevance set for retrieval evaluation."""
    generator = random.Random(seed)
    items: list[dict[str, object]] = []
    for index in range(records):
        product = generator.choice(PRODUCTS)
        word = generator.choice(INTERNAL_JARGON)
        query = f"how do I fix a {word} mismatch in {product}"
        relevant = [f"rec-{index:04d}"]
        items.append({"query": query, "relevant_ids": relevant, "kind": "retrieval"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return {"path": str(path), "items": len(items)}
