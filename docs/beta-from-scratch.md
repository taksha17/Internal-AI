# Beta: a from-scratch language model

Beta is not built in the alpha work. It is the later track, and it exists
because the machines we have cannot train it yet.

## What beta is

Beta is a language model trained from random initialisation on a released
internal dataset. It does not start from Qwen, Llama, Gemma, or any other
published checkpoint. The from-scratch transformer in
`src/internal_ai/training/model.py` already has a causal mask for that
objective. A change is a new dataset version and a new run. Alpha adapters
are not training data for it.

## Why it waits

PRD section 10 sizes a from-scratch run at about 6 x parameters x tokens.
The planning figures there are about 1,000 GPU-hours for a 1B model and
about 100,000 GPU-hours for a 7B model. Alpha runs on the 4 GB GTX 1650 by
adapting a pinned open base with a rank-8 LoRA streamed from disk. That
recipe does not train a from-scratch model of this size. The 1650 is the
alpha machine, not the beta machine.

## What has to be true before beta starts

- A compute choice from PRD section 10, with enough GPU-hours for the
  target size.
- The same human gates as the rest of the platform: the requester does not
  approve their own gate.
- The from-scratch track still has no loader for a published checkpoint.
