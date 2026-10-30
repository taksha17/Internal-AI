# 46 — The answer comes from the weights on this machine

Step 10 of `docs/yc-plan/RFS-READINESS.md`, the last one.

## What was wrong

`/v1/answer` returned 503 whenever `INTERNAL_AI_LOCAL_RUNNER` was unset. Every
answer therefore came from a process someone else had started: the platform
borrowed an inference server instead of being one, which is exactly the shape
the RFS says a company should not have to accept. The weights, the adapter and
the tokenizer were already on disk and pinned by sha256, and nothing read them.

## What changed

`training/alpha/local.py` generates from the pinned base with the adapter on the
layers it covers. It reuses the layer code the trainer already trusts —
`load_layer`, `layer_forward`, `rms_norm`, `linear` — and adds the two things a
generator needs that a training step does not: the final norm and a projection
onto the vocabulary, tied to the embedding when the base has no separate
`lm_head`.

`answer_adapter` chooses the path: a configured runner if the lab has one,
otherwise the weights here. The local path asks `require_open_weights` first,
because reading pretrained parameters is the act provenance mode governs. A
`scratch_only` install gets a 503 that names the policy and the gate, not a
generic failure, and a missing snapshot gets one that says which file is gone.

`LocalModelAdapter` implements the same `LLMAdapter.complete` protocol the runner
does, so `cited_answer` — and with it the citation check, the memorisation probe
and the guardrails — is unchanged and cannot tell which generator produced the
text.

## The honest limits of this generator

It is a reference implementation, not a serving engine:

- **It recomputes the prefix for every token.** There is no KV cache, so cost
  grows with the square of the answer length.
- **It streams every layer from disk on every token.** That is what the 4 GB
  recipe can do, and it is why this is usable for a smoke test and not for a
  production query mix.
- **It is greedy only.** `temperature` is refused rather than ignored, because
  an answer that changes between two identical questions is not auditable.

Those are stated here instead of being papered over with a benchmark that
measures the wrong thing. A KV cache and a resident-weight path are the next
change, not this one.

## The extra was incomplete

`[project.optional-dependencies].alpha` declared `torch` and `safetensors`, and
the tokenizer a base ships with comes from `tokenizers`, which was not declared.
`train_prefix` had the same latent failure in `_encode`: a lab that installed
only the alpha extra would have had a trainer that dies on its first sequence.
It is now a declared dependency, and the generator raises a named
`LocalGenerationUnavailable` naming the extra when it is absent, which the route
returns as a 503 rather than a stack trace.

## Tests

`tests/test_alpha_local.py` — a real one-layer snapshot on disk produces a
token, twice identically; adapter pairs load per layer and an empty adapter is
refused; a card that names no pinned base, or the right repo at the wrong
revision, is refused; generation refuses a temperature; the default install
refuses the local path in `scratch_only`; and a configured runner is still
preferred when one exists.
