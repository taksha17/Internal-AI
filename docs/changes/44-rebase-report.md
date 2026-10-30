# 44 — A rebase is a recipe, and the decision is four columns

Step 9 of `docs/yc-plan/RFS-READINESS.md`.

## What was wrong

"The base model changes and you keep everything" is the strongest sentence in
the RFS and the repo had nothing behind it. There was one hardcoded pin, one
adapter, and one evaluation that compared an adapter to keyword search. Nothing
recorded what would have to be repeated, and nothing compared the four things a
person actually has to choose between: the adapter on the old base, the adapter
on the new base, the new base alone, and the API they would rent instead.

## What a recipe is

`build_recipe` writes down what a rebase repeats: the dataset and its content
hash, the base repo id, revision and licence class, the eval set ids, the seed
and sequence count, and the brief's metrics **as they stand at that moment**.

The bars are copied rather than referenced, for the same reason
`brief.progress_rows` reads recorded targets: editing the brief tomorrow must
not rewrite how today's report was judged. It refuses four things at the
door — an unreleased dataset, an eval set that can still move, a base whose
licence class this install may not load, and a brief with no metric to judge
by. Authoring it needs the new `write_recipe` capability, held by `ml_lead` and
`program_lead`; a `data_owner` is refused and that is tested.

## The four columns

`compare` takes the same metrics from all four candidates and judges the new
adapter three ways. It breaches when:

- a brief metric is missed, or cannot be judged at all;
- any column does not measure a bar the brief names — a gap is a breach, not a
  blank cell, because "the rented API does not report recall@10" is a fact the
  decision needs;
- the adapter does not beat the un-adapted new base — a tie counts as no gain,
  the same rule the base comparison in step 6 applies;
- the new adapter loses ground against the old one. A rebase that scores lower
  than what it replaced is a regression however good the absolute number is.

`propose` then records `rebase.promotion_proposed` or refuses with
`rebase.promotion_refused` naming the first three breaches. It is not a gate
and does not open one: the promotion gate, its approvers and the serving
pointer stay exactly where they were, so the old model keeps serving until a
person approves the new one.

## What is not here

The run itself. `train_alpha` already adapts a base as a governed job with a
`reproducibility.json`, and `evaluate_alpha` already refuses an adapter that
does not beat its base. A rebase is those two, twice, plus this report; wiring
them into one command needs a second pinned base on disk, which no test machine
has. Rather than invent a runner that cannot be exercised, this change stops at
the part that decides — and says so.

## Tests

`tests/test_rebase.py` — what the recipe records and reloads, the four
refusals plus the missing capability, the four-column table, and each of the
four breach kinds, with the refusal reaching the audit log.
