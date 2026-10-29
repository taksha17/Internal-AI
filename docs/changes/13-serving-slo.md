## What failed

Checklist V-03, V-04, V-05, V-09, V-10. One live `POST /v1/answer` returned 200 in 4.3 seconds from `qwen2.5-3b-lumen`, with the citation kept on `Acmy_Website/next.config.ts:36`. Forty `/health` calls were all 200, p50 2.1 ms, p95 2.7 ms. Seventy authenticated `/v1/models` calls returned 58 times 200 and 12 times 429. The audit row stored latency, hit count, and model, not the question. No latency target exists, no cost per 1,000 requests, no peak load figure, no input or output check beyond that citation rewrite, and no retention rule for the log.

## Change

- Read `latency_ms` from the project brief. `POST /v1/answer` and `POST /v1/search` write `latency_ms` as they do now. A daily rollup, `internal-ai serve report`, prints p50 and p95 from the audit log and whether p95 is under the brief. It does not add a new database table: it reads `serving.request` events.
- Cost per 1,000 requests is `1000 * mean(cost_usd)` over those events. Print it in the same report.
- Load: a test sends 40 concurrent `/health` requests and 70 authenticated `/v1/models` requests and expects zero 500s and at least one 429 once the limit is passed. Do not load-test `/v1/search` against the 12,721-record corpus.
- Guardrail: an answer whose `citation_passed` is false is returned with HTTP 200 and `citation_passed: false`, which already replaces the text with the hit lines. Add a second check that refuses to return the raw question in the audit `details`. A test spies on the recorded details and fails if the question string is present.
- Retention: the report command drops serving events older than the brief's timeline only when a person passes `--apply`. Default is a dry run that prints the count.

## Proof

- The report prints p50, p95, and cost per 1,000 from fixture audit rows.
- The concurrency test sees a 429 and no 500.
- The audit spy does not see the question text.

## Out of scope

Do not add a second GPU pool or an autoscaler. Capacity separation stays a note on the brief's deployment field until there is a second machine.
