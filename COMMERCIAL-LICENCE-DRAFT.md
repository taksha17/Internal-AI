# Commercial Licence Agreement — DRAFT

> **This is a draft, not a licence.** Nothing in this file grants any right. It
> exists so that a customer's counsel and a lawyer have something concrete to
> mark up, and so that the founder's own reading of the deal is written down
> rather than assumed. No agreement is in force until the licensor signs it.
> Every `[bracketed]` value is a decision that has not been made.

Status: draft 0.1, 31 October 2026. Not reviewed by a lawyer. Not to be sent
to a customer as-is.

## Notes for the lawyer

The decisions this draft makes, so they can be argued with rather than
discovered later:

1. **The grant is on the customer's own hardware.** The licensor ships
   software and support; it does not run a hosted service. There is no
   multi-tenant SaaS here and none is planned.
2. **The customer keeps everything it makes.** Datasets, weights, adapters,
   evaluation reports, prompts and records are the customer's, with no claim
   by the licensor over them.
3. **The source is public and noncommercial.** `LICENSE` grants the world
   reading, study and noncommercial use. This agreement is the only way to use
   it in a business, and it is what the customer is paying for.
4. **The licensor's own company is licensed already** under `LICENSE`, so
   there is no internal use that needs a separate grant.
5. **No telemetry.** The platform has no outbound calls in its default
   configuration and sends no records anywhere. That is a design property, and
   the customer may verify it rather than trust it.
6. **Base models are a separate decision.** This agreement licenses the
   software. What the customer loads into it is licensed by whoever wrote the
   weights, and the platform refuses a base it has not been told about.
7. **The warranty position is deliberately narrow.** The licensor warrants the
   software's behaviour against its documentation and its own test suite. It
   does not warrant that a model trained with it will be good, useful, safe or
   fit for any purpose: model quality is a function of the customer's data,
   their decisions and their gates.

---

## 1. Parties

This agreement is between:

- **[LICENSOR], the copyright holder of the software** (the `Required Notice`
  line in `LICENSE`), of [address], and
- **[LICENSEE], the customer**, of [address].

"LICENSOR" and "LICENSEE" are each a "Party" and together the "Parties".

## 2. Definitions

- **Software**: the source code, object code, documentation and configuration
  files made available by the licensor under this repository, excluding Third
  Party Components.
- **Third Party Components**: the components listed in
  `THIRD_PARTY_NOTICES.md`, and any base model the licensee loads, each
  licensed by its own author.
- **Installation**: one deployed instance of the Software on hardware the
  licensee controls, whether or not that hardware is physical.
- **Customer Materials**: the licensee's documents, records, code, prompts,
  datasets, model weights, adapters, evaluation reports and every other
  artefact the licensee creates or processes with the Software.
- **Licensed Use**: using the Software on the licensee's own hardware, for the
  licensee's own internal business purposes, including training, evaluating
  and serving models built from the licensee's own data.

## 3. Grant

Subject to payment of the fees in clause 5 and to the rest of this agreement,
the licensor grants the licensee a non-exclusive, non-transferable,
non-sublicensable licence to install and run the Software on hardware the
licensee controls, for the Licensed Use, for the term in clause 4.

The licensee may permit its employees and its contractors to exercise the
rights above on its behalf, provided each is bound by obligations at least as
protective as this agreement. The licensee remains responsible for their use.

No right is granted to distribute, sublicense, resell, host for third parties,
or make the Software available to any person who is not an employee or
contractor of the licensee, except where clause 6 provides otherwise.

## 4. Term

- **Evaluation licence**: [30] days from the date of this agreement, for a
  proof of concept, free of charge, non-production. It converts to a
  production licence on payment of the fees, without a gap in entitlement.
- **Production licence**: [perpetual, or a term of [12] months renewable],
  starting on the first day of the month after the fees are paid, as set out
  in the order form.

This agreement survives expiry or termination in respect of clauses 8, 9, 10,
12, 13 and 15.

## 5. Fees

- Production: **[currency] [amount] per Installation per [year], payable
  [annually in advance]**, or as set out in the order form.
- Evaluation: no fee, on the terms of clause 4.
- [Support and managed deployment, if bought, are priced separately.]

Fees are exclusive of taxes and are payable within [30] days of invoice. The
licensor may adjust the fee for a renewal by giving [90] days' notice before
the renewal date.

## 6. Redistribution and service rights

[To be decided, per deal.]

- The licensee may redistribute the Software only as embedded in a product or
  appliance it distributes, and only where that product does not expose the
  Software itself as a service for others' models.
- The licensee may offer the Software as part of a hosted service only with
  the licensor's written consent and [a separate hosted licence].
- Where the licensee ships the Software to a customer of its own, the
  licensee is the licensor of the Software to that customer and the licensor
  [grants the licensee the right to sublicense it for that purpose] /
  [requires a pass-through of these terms].

## 7. The customer's obligations

The licensee will:

- use the Software only within the Licensed Use;
- keep its own credentials and access controls, and not share an account
  outside its organisation;
- not remove or obscure the `Required Notice` line, the copyright notices or
  the Third Party Notices;
- not present the Software as the licensor's product, and not use the
  licensor's name or marks except as permitted by `LICENSING.md`;
- maintain its own record of which Third Party Components and base models it
  loads, and satisfy their own licences;
- comply with the law applying to its use, including export, sanctions and
  data protection law.

## 8. Data, privacy and telemetry

The Software is designed to run inside the licensee's boundary. In its default
configuration it makes no outbound network calls and transmits no record,
prompt, dataset, weight or aggregate statistic to the licensor or to any third
party. The licensor has no telemetry, no tracking pixel and no analytics in the
Software.

The licensee is the controller of the Customer Materials, and the licensor
processes them only on the licensee's hardware and only as the Software is
written to do. The parties will enter a data processing agreement if the
licensee requires one.

## 9. Ownership

The licensor retains all rights in the Software and in the licensor's
improvements to it, including any configuration, patch or extension the
licensor supplies. The licensee retains all rights in the Customer Materials,
and the licensor claims no right over them, including any model, weight,
adapter, dataset or evaluation report the licensee produces.

Feedback the licensee sends may be used by the licensor without restriction or
obligation, but no Customer Material and no identifying information is used
for that purpose without written permission.

## 10. Third Party Components and base models

The Software includes Third Party Components under their own licences, listed
in `THIRD_PARTY_NOTICES.md`. The licensor grants no rights in them beyond
those licences, and no additional warranty in respect of them.

The Software will refuse to load a base model that it has not been told about
and that is not allowlisted, hash-pinned and licence-classed. A licensee that
wishes to load a base model the licensor has not allowlisted needs a written
exception from the licensor, and remains responsible for complying with that
base model's licence.

## 11. Warranties

The licensor warrants that, for [12] months from delivery, the Software will
perform materially in accordance with its documentation and its published test
suite, and that it contains no material virus or malicious code.

The licensor does not warrant that the Software will be uninterrupted or
error-free, that it will meet the licensee's availability or performance
requirements, or that any model the licensee trains with it will be accurate,
useful, safe, non-infringing or fit for any purpose. Model quality depends on
the licensee's data, its configuration and its own release decisions.

The licensor will, at its option, repair or replace the Software where it
fails to perform in accordance with its documentation.

## 12. Limitation of liability

Nothing in this agreement limits liability that cannot lawfully be limited.

Subject to that, and to clause 13:

- neither Party is liable for indirect or consequential loss, loss of profit,
  loss of business, loss of anticipated savings, or loss or corruption of
  data, however arising;
- each Party's total liability under this agreement is limited to the fees
  paid and payable by the licensee in the [12] months before the event giving
  rise to the claim;
- neither Party is liable for loss caused by the licensee's own training data,
  by its decision to promote a model, or by its use of the Software in
  breach of this agreement.

## 13. Indemnities and insurance

[The licensor indemnifies the licensee against third-party claims that the
Software, as supplied and used in accordance with this agreement, infringes
that third party's intellectual property rights, subject to the licensor's
right to modify or replace the Software at its own cost. The licensee
indemnifies the licensor against claims arising from the Customer Materials or
from the licensee's use of a base model.]

[Public liability and professional indemnity insurance: [amounts], with
[insurers].]

## 14. Suspension, termination and effect

The licensee may stop using the Software at any time. The licensor may
suspend the licence, on written notice, if the licensee uses the Software in
breach of clause 7 and does not fix it within [30] days, or if continued use
would breach the law.

Either Party may terminate for the other's material breach that is not fixed
within [30] days of notice, or immediately if the other becomes insolvent.

On termination the licensee stops using the Software, removes it from its
hardware, and deletes its copies, except that it may keep one archival copy
solely to demonstrate what it was running. Clauses 8, 9, 10, 12, 13 and 15
survive.

## 15. Confidentiality and publicity

Each Party keeps the other's confidential information confidential and uses it
only for this agreement.

Neither Party issues a press release naming the other without the other's
written consent, and the licensee may say that it uses the Software [in
marketing, subject to the licensor's reasonable approval of the wording].

## 16. Audit

On reasonable notice, the licensee may ask the licensor to confirm in writing
which base models the Software would load for it, and the licensor will
answer from its own records. The licensee keeps its own record of what it
loads, and may show it to the licensor on request where a base model's licence
requires it.

## 17. Governing law and disputes

This agreement is governed by the laws of [Delaware], without regard to
conflict of laws rules. The Parties submit to the exclusive jurisdiction of
the courts of [Delaware], and each may seek injunctive relief in any
jurisdiction to protect its confidential information or its intellectual
property.

## 18. Notices

Notices are in writing and delivered by hand, by tracked courier or by email
to the addresses in clause 1, to the attention of [name]. Email notices are
deemed received [24] hours after sending, provided no delivery failure is
received.

## 19. General

- This agreement is the whole agreement between the Parties on its subject,
  and replaces anything agreed before it, including the `LICENSE` file's terms
  so far as they apply to the Licensed Use.
- The `LICENSE` file continues to govern the Software as published for
  everyone else, and the licensor's company remains licensed under it.
- A change to this agreement is in writing and signed by both Parties.
- If a clause is not enforceable, the rest of the agreement still applies, and
  the Parties will replace it with something enforceable that keeps its
  intent.
- Nothing in this agreement creates a partnership, joint venture or
  employment relationship, or lets either Party bind the other.
- This agreement may be signed in counterparts, including electronically.

## 20. Signatures

For **[LICENSOR]**

| | |
| --- | --- |
| Name | [ ] |
| Title | [ ] |
| Date | [ ] |
| Signature | [ ] |

For **[LICENSEE]**

| | |
| --- | --- |
| Name | [ ] |
| Title | [ ] |
| Date | [ ] |
| Signature | [ ] |
