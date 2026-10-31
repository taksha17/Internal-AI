# Third-party notices

This file lists what the project depends on and under what terms, for the
components shipped or required by `pyproject.toml`. Licence names are those the
upstream projects declare; the authoritative text is in each package, and this
list must be regenerated whenever a dependency changes.

## Runtime

| Component | Licence | Used for |
| --- | --- | --- |
| fastapi | MIT | JSON API and console |
| uvicorn | BSD-3-Clause | ASGI server |
| pydantic | MIT | request and domain validation |
| httpx | BSD-3-Clause | outbound calls when a lab enables them |
| numpy | BSD-3-Clause | numeric work outside the training loop |
| jinja2 | BSD-3-Clause | server-rendered console pages |
| python-multipart | Apache-2.0 | form uploads |
| pypdf | BSD-3-Clause | PDF text extraction |

## Development and optional extras

| Component | Licence | Extra |
| --- | --- | --- |
| pytest | MIT | `dev` |
| pytest-cov | MIT | `dev` |
| ruff | MIT | `dev` |
| torch | BSD-3-Clause | `train`, `alpha` |
| safetensors | Apache-2.0 | `alpha` |
| tokenizers | Apache-2.0 | `alpha` |

## Vendored

| Component | Version | Licence | Location |
| --- | --- | --- | --- |
| htmx | 2.0.4 | BSD-2-Clause | `src/internal_ai/console/static/htmx.min.js` |

htmx is vendored rather than loaded from a CDN so the console works on a network
with no outbound access. Its licence text travels with the file.

## Base models

Pretrained weights are not shipped with this repository. The two bases the alpha
track knows about are pinned by repo id, revision and shard hash in
`src/internal_ai/training/alpha/pin.py`.

| Base | Licence | Class | Loadable by default |
| --- | --- | --- | --- |
| `Qwen/Qwen3-4B` | Apache-2.0 | permissive | yes |
| `Qwen/Qwen2.5-3B` | `qwen-research` | research-only | no |

A base is only ever loaded when provenance mode has been opened through a
`policy_decision` gate on `P6`, and a base whose licence class is not
permissive additionally needs a `base_model_exception` gate. Adapters produced
from an allowed base remain subject to that base's terms, which the model card
records by repo id, revision and licence.

## What this project does not include

No model weights, no datasets, no customer records, and no telemetry. An install
with `INTERNAL_AI_OFFLINE=1` — the default — makes no outbound call at all.
