"""Synthetic bank KYC forms for the form-filling agent.

Every value comes from a fixed pool and is marked fake, so a filled form can
never be mistaken for a real customer: names start with Test or Sample,
phones use the fictional 555-01 range, addresses are on Test Street, and id
numbers carry a SAMPLE prefix. The generator takes a seed and nothing else,
which keeps the corpus reproducible and keeps model output out of it entirely
— a template is deterministic and costs no model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from internal_ai.ids import content_hash

GENERATOR_VERSION = 2
DEFAULT_FORMS = 1200
FORMS_PER_CUSTOMER = 4

KYC_FIELDS: tuple[str, ...] = (
    "full_name",
    "date_of_birth",
    "nationality",
    "id_type",
    "id_number",
    "address",
    "phone",
    "email",
    "occupation",
    "employer",
    "annual_income_band",
    "account_type",
    "source_of_funds",
    "pep_declaration",
)

FIRST_NAMES = (
    "Testara",
    "Sampta",
    "Dummya",
    "Exampla",
    "Fictoria",
    "Mockesha",
    "Testwin",
    "Sampleen",
)
LAST_NAMES = (
    "Testerson",
    "Samplington",
    "Mockford",
    "Exampler",
    "Fictman",
    "Mockley",
    "Testworthy",
    "Samplefield",
)
NATIONALITIES = ("Testlandian", "Samplian", "Examplese", "Mocklandian")
ID_TYPES = ("SAMPLE-PASSPORT", "SAMPLE-DRIVING-LICENCE", "SAMPLE-NATIONAL-ID")
OCCUPATIONS = (
    "Test Engineer",
    "Sample Analyst",
    "Mock Auditor",
    "Dummy Clerk",
    "Mock Underwriter",
    "Sample Compliance Clerk",
)
EMPLOYERS = (
    "Test Street Bank",
    "Sample Ledger Ltd",
    "Mock Vault Inc",
    "Mock Trust Ltd",
    "Dummy Savings Ltd",
)
INCOME_BANDS = ("0-25k TEST", "25-50k TEST", "50-100k TEST", "100k+ TEST")
ACCOUNT_TYPES = ("TEST-CURRENT", "TEST-SAVINGS", "TEST-BUSINESS")
SOURCES_OF_FUNDS = ("test salary", "sample savings", "mock pension")
PEP_VALUES = ("not a PEP (test)",)


@dataclass(frozen=True, slots=True)
class KycForm:
    """One blank-then-filled KYC form; values are the ground truth."""

    form_id: str
    customer_id: str
    values: dict[str, str]

    def as_dict(self) -> dict[str, str | dict[str, str]]:
        return {
            "form_id": self.form_id,
            "customer_id": self.customer_id,
            "values": dict(self.values),
        }


def _customer_values(rng: random.Random, index: int) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return {
        "full_name": f"{first} {last}",
        "date_of_birth": (
            f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1950, 2000)}"
        ),
        "nationality": rng.choice(NATIONALITIES),
        "id_type": rng.choice(ID_TYPES),
        "id_number": f"SAMPLE-ID-{index:06d}",
        "address": f"{rng.randint(1, 999)} Test Street, Sampleton ST{index % 90 + 10}",
        "phone": f"+1-555-01{rng.randint(10, 99):02d}",
        "email": f"test.user{index:04d}@example.test",
        "occupation": rng.choice(OCCUPATIONS),
        "employer": rng.choice(EMPLOYERS),
        "annual_income_band": rng.choice(INCOME_BANDS),
        "account_type": rng.choice(ACCOUNT_TYPES),
        "source_of_funds": rng.choice(SOURCES_OF_FUNDS),
        "pep_declaration": rng.choice(PEP_VALUES),
    }


def generate_forms(count: int = DEFAULT_FORMS, seed: int = 7) -> list[KycForm]:
    """Build count synthetic forms, four per customer, from the seed alone."""
    rng = random.Random(seed)
    forms: list[KycForm] = []
    customer = 0
    while len(forms) < count:
        customer += 1
        customer_id = f"CUST-TEST-{customer:05d}"
        values = _customer_values(rng, customer)
        for _repeat in range(FORMS_PER_CUSTOMER):
            if len(forms) >= count:
                break
            forms.append(
                KycForm(
                    form_id=f"KYC-{len(forms) + 1:05d}",
                    customer_id=customer_id,
                    values=dict(values),
                )
            )
    return forms


PASSPORT_TEMPLATE = (
    "PASSPORT {id_type} {id_number} holder {full_name} "
    "born {date_of_birth} nationality {nationality}"
)
DOC_FAMILIES: tuple[tuple[str, ...], ...] = (
    (
        PASSPORT_TEMPLATE,
        "PASSPORT of {nationality} issue: holder {full_name}, born {date_of_birth}, "
        "{id_type} {id_number}",
        "IDENTITY PAPER {id_type} number {id_number} belongs to {full_name}, "
        "a {nationality} national born {date_of_birth}",
    ),
    (
        "UTILITY BILL {full_name} resides at {address} phone {phone}",
        "UTILITY BILL for the property at {address}: the account holder is {full_name}, "
        "who can be called on {phone}",
        "COUNCIL TAX NOTICE addressed to {full_name} at {address}, contact number {phone}",
    ),
    (
        "PAYSLIP {full_name} works as {occupation} for {employer} income band {annual_income_band}",
        "PAYSLIP from {employer}: {full_name} is employed there as a {occupation}, and the "
        "pay sits in the {annual_income_band} band",
        "EMPLOYMENT RECORD {employer} lists {full_name} as a {occupation} on the "
        "{annual_income_band} income band",
    ),
    (
        "BANK STATEMENT {email} source of funds {source_of_funds} account {account_type}",
        "BANK STATEMENT for the {account_type} account held by {email}, where the source "
        "of funds is stated as {source_of_funds}",
        "ACCOUNT SUMMARY {account_type} in the name of {email}, funded from {source_of_funds}",
    ),
    (
        "DECLARATION {full_name} declares {pep_declaration}",
        "SIGNED DECLARATION: {full_name} states that they are {pep_declaration}",
        "COMPLIANCE DECLARATION signed by {full_name}, recording {pep_declaration}",
    ),
    (
        "TAX RESIDENCY CERTIFICATE {full_name} of {nationality} residing at {address}",
        "TAX CERTIFICATE: {full_name}, a {nationality} national of {address}, "
        "id {id_type} {id_number}",
        "RESIDENCY RECORD {id_number} ({id_type}) certifies {full_name} resides at {address}",
    ),
)

DOC_TEMPLATES: tuple[str, ...] = tuple(template for family in DOC_FAMILIES for template in family)


def evidence_lines(
    form: KycForm, rng: random.Random, distractors: tuple[str, ...] = ()
) -> list[str]:
    """The customer's documents as checkable lines, shuffled with decoys.

    The agent's job is selection, not recall: the true lines name this form's
    values, the decoys name other customers', and only the line's content
    tells them apart.
    """
    lines = [rng.choice(family).format(**form.values) for family in DOC_FAMILIES]
    lines.extend(distractors)
    rng.shuffle(lines)
    return lines


@dataclass(frozen=True, slots=True)
class KycCorpus:
    forms: tuple[KycForm, ...]
    seed: int
    manifest: dict[str, str | int] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return content_hash([form.as_dict() for form in self.forms])


def build_corpus(count: int = DEFAULT_FORMS, seed: int = 7) -> KycCorpus:
    """Generate the corpus and record how, so a rerun can prove it matches."""
    forms = generate_forms(count=count, seed=seed)
    corpus = KycCorpus(forms=tuple(forms), seed=seed)
    manifest = {
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "forms": len(forms),
        "customers": len({form.customer_id for form in forms}),
        "content_hash": corpus.content_hash,
    }
    return KycCorpus(forms=corpus.forms, seed=seed, manifest=manifest)
