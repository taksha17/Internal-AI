# Novice search-first — implementation plan

From the approved design (`2026-10-27-novice-search-first-design.md`).
No new training logic, no governance change, no new model family.

## Where we hook in

- `services/build.py`: `inspect_source`, `register_source`, `build_dataset`,
  `split_dataset`, `request_release`, `release_dataset`, `ask`.
  Ask needs a released dataset; the searcher runs with `embedder=None`
  (keyword path), so a novice gets value before any training.
- `console/app.py` + `templates/sources.html`: the inspect worker and the
  register/build buttons already exist; the guided mode is a thin skin over them.
- `services/jobs.py`: dataset build already runs as a job with status polling.

## Build order

1. Guided entry on the Data page. New card: one path field, one button
   "Build guided search". POSTs to a new `/console/guided` route that calls
   `register_source` with auto-filled name (folder stem), purpose
   ("guided search over pasted folder") and default ACL, then redirects to a
   guided progress section. Same `REGISTER_SOURCE` permission, same refusal
   paths (outside roots, weights, secrets, non-internal).
2. Progress tracker. A guided section (on sources or dataset page) showing
   six states with existing links: inspected, source approved, built, split,
   released, askable. Each step button calls the existing service function;
   build runs through the existing job + `_job_status` poll. No new state
   machine; read state from gates, dataset stats and release flag.
3. Honest small-data copy. On the dataset and Ask pages show records/tokens
   plus one line: keyword search is answering; training is optional and only
   helps past a stated volume. BM25 stays visible next to any model result.
4. Optional tiny run stays manual. Keep Models page defaults
   (steps 60, cap 2.0); auto pair-mode only if trivial (tree/code -> title,
   else paraphrase). No auto-promotion; promotion keeps its gate.

## Defaults to lock in review

- Name from folder stem sanitised as today; ACL `["engineering"]`.
- Purpose string fixed for guided runs so the gate evidence reads clearly.
- Copy thresholds: below ~100 records / ~25k tokens say keyword search wins
  (STATUS.md pilot numbers); link to what volume would change that.

## Tests

- E2E novice path on the synthetic corpus: inspect -> register -> approve
  (second person) -> build job -> split -> release -> ask returns citations.
- Denials already covered stay green; add guided-route denials: outside
  roots, weight folder, secret folder, build before approval, ask before
  release, over-budget tiny run.
- `pytest`, `ruff check`, `ruff format --check` clean; commit + push per AGENTS.md.

## Out of scope for this cut

- Classifier / extractor auto-detect (option B), probe runs (option C),
  auto gate approval, new compute backend, SSO, public exposure.
