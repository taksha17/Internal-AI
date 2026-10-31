# Licensing

Two things are being licensed here, and they are different: the software, and
the models you build with it.

**The software** is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). It is not open source. Reading
it, changing it, running it for research, teaching, study, charity or a
noncommercial project is free. Using it in a business is not, except by the
licensor's own company, which is licensed already.

**The models** are yours. Weights you train, adapters you produce, datasets you
assemble and evaluation reports you write belong to you and your organisation:
nothing in this repository claims any right over them, and the platform never
sends them anywhere.

This page explains where the line falls. It is a plain-language summary written
by the people who wrote the code, not legal advice, and it does not replace the
licence text or an agreement signed by both sides.

## When you need a commercial licence

You need one if the software is used by or for a company, in the course of its
business, and you are not the licensor's own company (see below). That includes
the cases that surprise people:

- running it internally to build or serve a model your organisation uses;
- offering answers, search or evaluation as part of a product or service;
- embedding it in something you distribute, even if that thing is free;
- running it on a customer's machines as part of a paid engagement;
- using it in a pilot that will be judged on commercial terms.

You do not need one to read the code, to evaluate it on your own machine as an
individual, to teach with it, to study it, to reproduce its tests, or to
contribute to it.

## The licensor's own company

The licensor, any company the licensor controls, and their employees and
contractors acting for them do not need a separate agreement to use the
software commercially. The licence in the `LICENSE` file grants it to them:
perpetual, irrevocable and royalty-free, and it travels with the copyright, so
when the copyright is assigned to the company, the company holds it.

That exception is narrow on purpose. It covers the party that writes the
software and the company that develops and delivers it — running it to train a
model, deploying it for a customer, shipping it under a commercial agreement,
and having an agency write code against it. It does not cover a customer, a
partner, or anyone else: those still need a commercial licence before the
software runs inside their business.

## What the commercial agreement will contain

Not a price list — that is a conversation — but the shape is settled enough to
describe:

- an **evaluation licence**: time-boxed, non-production, for a proof of concept;
- a **production licence**: per installation or per node, including internal
  business use, with or without support;
- **support terms**: severity levels, response targets, upgrades and a
  long-term-support branch once there are paying customers;
- **redistribution and service rights**, if you intend to host it for others;
- the **warranty and liability** position, which for an on-premises product
  with no telemetry is narrower than a SaaS contract's.

Ask by opening an issue at <https://github.com/taksha17/Internal-AI>. There is no
sales form and no tracking pixel; the repository is the front door.

## What is not written yet

Two documents have to exist before the first customer signs anything, and
neither is written:

- the **commercial licence agreement** itself: the per-installation or per-node
  terms, the support levels, the liability position, and a signature block. The
  shape of it is described above; the words are not drafted.
- the **contributor licence agreement** for outside code. Without it, an
  outside contribution cannot be merged, because the chain of title to a
  contribution in a product the company sells is not established.

Both need a lawyer rather than an engineer. Until they exist, the arrangement
in this file is coherent and the gaps are named rather than papered over.

## Base models are licensed separately

Loading a pretrained base is a separate licensing decision from using this
software, and the platform is built so you cannot make it by accident:

| Base | Licence | Class | Loadable |
| --- | --- | --- | --- |
| `Qwen/Qwen3-4B` | Apache-2.0 | permissive | yes, when on disk and hash-verified |
| `Qwen/Qwen2.5-3B` | `qwen-research` | research-only | no: refused by the default allowlist |

The allowlist is `training/alpha/pin.py`, the class comes from the declared
licence string, an unreviewed licence gets the strictest class, and
`scratch_only` — the default install — loads no pretrained weights at all.
Adding a base means adding its repo id, revision, shard hashes and licence, and
a base whose class is not permissive additionally needs its own
`base_model_exception` gate. Third-party components and their licences are
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributions

Contributions are accepted on the terms in [CONTRIBUTING.md](CONTRIBUTING.md):
your contribution is licensed under the same noncommercial terms, and the
copyright holder may also license it commercially. That second half is what
makes dual licensing possible, and it is why an outside contribution should not
be merged until a contributor licence agreement is in place. Until it is, the
project accepts changes from the owner and from agents working under the owner's
direction.

## Brand

The name and the marks in `Logo/` are not licensed with the software. A fork may
say it is based on this project; it may not present itself as this project.

## Decisions that need a lawyer, not an engineer

Stated plainly so they do not quietly become engineering decisions:

1. The legal entity that holds the copyright — a person or a company — and the
   `Required Notice` line that names it. It is a person today. When the company
   exists, the copyright is assigned to it and the notice line changes to the
   company name: one line, one file. Nothing else moves, because the commercial
   licence granted to the licensor's company in `LICENSE` travels with the
   copyright.
2. The contributor licence agreement wording, and whether assignment or a broad
   licence grant is wanted.
3. Whether the commercial offer is a licence, a subscription, or a licence plus
   managed deployment.
4. The warranty, indemnity and liability position, and whether the evaluation
   licence needs its own terms.
5. Whether any part of the product should move to a permissive licence to widen
   adoption, and if so which part.

Until items 1 and 2 are settled, the current arrangement is coherent: the code
stays noncommercial for everyone but the licensor's own company, the models
stay the customer's, and no outside contribution is merged without a written
grant.
