## What failed

Checklist D-09, D-15. The quality filter has a language allowlist and an alphabetic-ratio heuristic. Records do not store a language tag, and nobody has measured how often the tag is wrong. There is no toxicity filter and no written rule for borderline text.

## Change

- During `filter_quality`, set `record.meta["language"]` using the existing allowlist heuristic. The tag is `en`, `other`, or `unknown`. Persist it on the dataset row's meta.
- At release, sample 30 records with the PII sampler's seed. The release stays blocked until a person with `domain_expert` posts a review: each sampled id marked `correct` or `wrong`. More than 10 percent `wrong` fails the release. Store `language_review` on the dataset stats.
- Add a toxicity wordlist file in the repo, `src/internal_ai/data/toxicity_terms.txt`, reviewed in the same pull request, and a borderline policy sentence in the dataset release notes: borderline text is kept and flagged `meta.borderline = true`, not deleted. A record that matches the wordlist is flagged. Release records `toxicity_flagged` count. A dataset with flagged rows still releases if the policy sentence is stored. A dataset with flagged rows and no policy sentence does not.

## Proof

- A record of English prose is tagged `en`.
- A review with 4 wrong out of 30 fails release.
- A flagged row without the policy sentence fails release.

## Out of scope

Do not download a language-identification model or a toxicity classifier.
