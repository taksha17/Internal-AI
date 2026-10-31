# The promotion margin

A candidate is promoted only when it beats something real by enough to be worth
believing. This page fixes the numbers, because "it looked better" is how a
platform ends up serving a model that is worse than the baseline it replaced.

## What is compared

Every promotion compares one candidate against three things, on the same frozen
evaluation set:

| Column | What it is | Why it is required |
| --- | --- | --- |
| Candidate | the model being promoted | the thing being argued for |
| Un-adapted base | the same base with no adapter, or the previous from-scratch model | proves the adaptation did anything at all |
| Keyword baseline | BM25 over the same corpus | proves the model earns its cost over the retrieval the company already had |
| Previous candidate | the model currently serving | catches a regression the first two columns would miss |

A tie is not a gain. A candidate that matches the un-adapted base is refused,
even when it is the best number on the page.

## The bar

1. **Metric.** `recall@10` on answerable questions is the primary number, with
   citation-correctness rate as a second. Both come from the eval report; a
   report that carries neither cannot promote anything.
2. **Set size.** At least 200 cases that a person has labelled, per category
   being promoted on. Fewer cases means the difference is not measurable, and
   the run is refused rather than reported with a caveat.
3. **Margin.** The candidate must lead **both** the un-adapted base and the
   keyword baseline by at least **+0.05 absolute** on `recall@10`.
4. **Confidence.** The paired difference must exceed **two standard errors**. A
   0.06 lead on 200 correlated questions is often noise; the arithmetic decides,
   not the person holding the gate.
5. **Regression.** Recall may not fall by more than **0.05 absolute** against
   the previous candidate, and the safety probe must pass. Both are already
   enforced in code.
6. **One look.** A frozen set is compared against one candidate once. A second
   look at the same set needs a new version behind a new `eval_set_freeze` gate,
   so the margin is not found by trying until it appears.

## What is enforced in code

`workflows.promote_model` refuses, in this order:

- an adapter produced by a backend that is not full-depth (`peft`), naming the
  backend it found;
- a promotion without an approved `model_promotion` gate;
- a promotion without a passing report on a set that is still frozen;
- a report judged by the person who built the model or the person approving it;
- a failed safety probe, or a recall drop above 0.05;
- a regression accepted by its own builder — the refusal names the metric, both
  figures, the model it fell behind and the questions that were lost;
- a lead under 0.05 over the un-adapted base or the keyword baseline, or a
  report that carries no such comparison at all;
- a lead that does not clear two standard errors of the paired difference, or a
  report that carries only a mean and no per-question scores.

`workflows.margin_refusal` holds point 3 and `workflows.confidence_refusal`
holds point 4, because the gate approves a decision rather than re-running the
evaluation: a report that never ran the comparison has not shown a lead, and an
approving human reading a number is not the same as the platform having checked
it. So a report now carries one score per question for the candidate and for
whatever it was compared against (`per_case`), and the standard error is
computed from those pairs rather than estimated from a mean.

The consequence is deliberate. A two-question eval set cannot promote anything,
because one flipped answer is half the score and the paired difference cannot
clear its own noise; the same applies to a short set that happens to show a
friendly number. The 200-case floor in point 2 is still the reviewer's question
rather than a coded refusal — the confidence rule refuses a set too small to
support its lead, which is the part the arithmetic can decide.

## What a refusal must say

A refusal names the comparison that failed and the figure that failed it, for
example `candidate 0.41 vs keyword 0.39: +0.02 is below the +0.05 margin`. A
refusal that only says "did not pass" is a bug in the refusal, not a detail.
