# Guided flow follows the pick — design

Intent: the guided flow should build what the suggester recommends, and say
plainly what else that path needs.

Why: today `build.guided_register` hardcodes the purpose as "guided search over
this folder", and `guided_steps` ends at "Askable" whatever the data is. So the
card can rank a classifier first and the flow underneath will build a search
index and call the job done. The recommendation is already actionable for
nobody: the options are text.

## Goals

- The ranked options are actionable. Picking one registers the source with that
  intent, so the name and the purpose on the gate evidence say what is actually
  being built.
- The tracker's steps follow the chosen intent, and each step names the role it
  needs and links to where it is done.
- A step the signed-in person cannot take is shown as needing a role rather than
  hidden or pretended.
- The search path is unchanged for anyone who just wants search.

## Non-goals

- No one-button chain that runs dataset, tokenizer, encoder and classifier in
  sequence. It would spend budget and start training with nobody reading
  anything, and it would fail halfway for anyone without `ml_lead`.
- No new task state, capability, gate type, table or column.
- A task classifier stays outside promotion, as `latest_frozen_pass` requires.
- No change to who may do what. `roles.py` and `permissions.py` are untouched.

## Anti-goals

- Implying `decide` or `generate` is one click.
- Hiding the encoder prerequisite: a classifier here is a head over a trained
  encoder, and a novice who is not told that will think the tool is broken.
- Calling a search index finished when the card on the same page says the data
  suits a classifier better.

## What the code does today

- `build.guided_register` fixes the name from the folder, the purpose to "guided
  search over this folder" and the readers to the default group.
- `guided_steps` builds six states from gates and dataset flags: inspected,
  source approved, dataset built, split, released, askable.
- The choices and the honest role limits, all already in `permissions.py`:

  | Step | Action | Roles |
  | --- | --- | --- |
  | register, inspect | `register_source` | data_owner, ml_lead |
  | approve the source | gate decision | data_owner, compliance |
  | build dataset, split, tokenizer | `build_dataset` | data_owner, ml_lead |
  | request release, release | `release_dataset` | ml_lead, data_owner |
  | create eval set | `create_eval_set` | ml_lead, domain_expert |
  | freeze eval set | `freeze_eval_set` | ml_lead |
  | train, register, evaluate, promote | `train_model`, `register_model`, `evaluate_model`, `promote_model` | ml_lead |
  | ask | `ask` | anyone signed in |

- So a data owner can paste a folder, approve it, build, split and release a
  dataset, and then stops: everything from training onward is `ml_lead`.

## Approaches considered

- **A: intent-aware tracker (picked).** The pick becomes real and the steps
  follow it, prerequisites and role limits included.
- **B: one button runs the chain.** Rejected above.
- **C: copy only.** Smallest, and it leaves the same page recommending a
  classifier while building search.

## Decisions

- The intent is a parameter of the tracker view, not new stored state:
  `?guided=<source>&intent=<kind>`. The one thing that is recorded is the
  purpose text at registration, which already exists and now names the intent,
  so an approver reads what is really being built.
- Steps are read from facts that already exist: the source gate, the dataset,
  its tokenizer file, registered models trained on that dataset, and the console
  jobs for training, evaluation and promotion. No new columns, no new machine.
- Switching intent is allowed and is a view change, because the honest
  recommendation moves as the data grows. Nothing is rewritten by switching.
- Each step carries a label, whether it is done, a link, the role it needs, and
  whether this person holds it. The wall is named, not hidden.
- The last step of each intent is where that intent actually ends: search ends
  at askable; decide ends at evaluated, with the non-promotable note; generate
  ends at promoted behind the existing gate.

## Step chains

- **search:** inspected, approved, dataset built, split, released, askable.
- **decide:** inspected, approved, dataset built, split, released, tokenizer,
  encoder trained on that dataset, classifier trained, evaluated. The encoder is
  a training run of its own, and the chain says so.
- **generate:** inspected, approved, dataset built, split, released, tokenizer,
  model trained, registered, evaluated on a frozen set, promoted.

## What is left to the builder

- How the pick is offered on the card and whether the recommended option starts
  selected, so a novice still has a one-press path.
- Wording for a step that needs a role the person does not hold, and whether the
  tracker offers to switch intent.
- Exact hrefs: the dataset page for split, release and tokenizer; the Models page
  for training and evaluation; evals for the frozen set; approvals for gates.
- Tests: each intent produces its own chain; a data owner sees the `ml_lead` steps
  named as needing a role rather than missing; picking `decide` records a purpose
  that says classifier; the search chain is unchanged; switching intent changes
  nothing but the view.

## What landed

- `build.GUIDED_INTENTS` names the three aims for the form, and
  `build.GUIDED_PURPOSES` maps each to the sentence an approver reads. The
  intent is a parameter of `guided_register`, so the gate evidence says whether
  a classifier or a from-scratch model is being built. An unknown intent falls
  back to search rather than failing a novice who sent nothing.
- The guided form is a path field plus three aims, search selected, so the
  one-press path is unchanged. The ranked options on the scan panel are now
  buttons: each posts the folder and its own intent, and the option the scan
  ranks first is the primary one. That is where the pre-selected recommendation
  lives, because a recommendation needs a scan and the form comes before one.
- `guided_steps` returns the chain for the intent, defaulting to the scan's
  recommendation when none is asked for, and to search when nothing is ready.
  Each step carries the action it takes; the tracker shows the role it needs
  when the signed-in person does not hold it, and marks the number circle
  dashed. Every step is a view over a gate, a dataset flag, a tokenizer file, a
  job or a model that already exists.
- Switching intent is a link, so nothing is rewritten. A pending step this
  person can take links to where it is done.
- Tests: `tests/test_guided_search.py` covers the recorded purpose and the
  fallback, the decide chain and its named `ml_lead` wall for a data owner, the
  ml_lead view of the same chain, the unchanged search chain, the default
  following the recommendation, and the scan panel's option carrying its own
  path and intent into the guided route.

## Builder check (self-read)

- Answered already: every step's role comes from `permissions.py`, and the
  encoder prerequisite comes from the classifier path built earlier.
- My calls: intent lives in the URL and the purpose text, not a new column; the
  chain is shown rather than run; the classifier keeps its non-promotable note.
- Theirs: none. Nothing here changes what gets built beyond wording and layout.
