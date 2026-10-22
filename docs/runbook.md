# Runbook

Day-to-day operation of the platform.

## Every morning

```bash
export INTERNAL_AI_HOME=./var
.venv/bin/python -m internal_ai.cli report          # spend, blocked tasks, open gates, risks
.venv/bin/python -m internal_ai.cli board           # what is in flight
.venv/bin/python -m internal_ai.cli audit verify    # the chain must verify
```

If the audit chain does not verify, stop and investigate before dispatching any
work. A broken chain means the history is not trustworthy, and every gate
decision on top of it is in doubt.

## Clearing a gate

```bash
.venv/bin/python -m internal_ai.cli gates list --status pending
.venv/bin/python -m internal_ai.cli gates approve --gate-id <id> --by <approver_role> --note "<evidence>"
```

Rules to remember:

- The approver must be one of the roles the gate names. An approval from
  someone else is recorded but does not satisfy the gate.
- The requester cannot approve their own gate. This is enforced, not advisory.
- Agents cannot approve gates at all.

## Watching a training run

```bash
tail -f var/artifacts/runs/<run_id>/metrics.jsonl
```

Each line has step, loss, ema_loss, grad_norm, learning_rate,
tokens_per_second, tokens_seen and cost_usd.

Stop the run and investigate if:

- loss becomes non-finite: the run stops itself
- three consecutive loss spikes: the run stops itself
- contrastive loss collapses near zero late in training: the run warns, and a
  human decides. A collapsed loss usually means memorisation, and the model
  will look excellent on the training pairs while failing on new data.

## When something is wrong

```bash
# stop all dispatch immediately
.venv/bin/python -m internal_ai.cli budget kill --actor <your_name>

# inspect the history
.venv/bin/python -m internal_ai.cli audit list --action train.run_started --limit 20
.venv/bin/python -m internal_ai.cli audit list --actor ml_engineer --limit 50

# resume dispatch
.venv/bin/python -m internal_ai.cli budget kill --release --actor <your_name>
```

The kill switch is global and stops dispatch immediately. It does not stop a
running training job; cancel that job through the compute plane.

## Rollback of a served model

```bash
curl -X POST http://localhost:8081/v1/models/embedding-v1/rollback \
  -H "X-Forge-Key: <admin key>" \
  -d '{"model_id": "<previous model id>", "reason": "regression in pilot"}'
```

The pointer change takes effect immediately and is audited with the previous
model id.

## Releasing a dataset

```bash
.venv/bin/python -m internal_ai.cli data lineage --dataset-id <id>
```

The release gate needs a quality report, a PII audit sample and the lineage.
Confirm the per-step statistics, especially `pii_scrub.total_redactions` and
`deduplicate.duplicate_rate`, before approving.

## Adding a role agent

1. Add the `RoleSpec` in `roles.py` with mission, sub-agents, inputs, outputs,
   capabilities, gates and KPIs.
2. Add it to `reviewer_for` so no role reviews itself.
3. Give it the narrowest capability set that lets it work.
4. Add a test showing the role is denied the capabilities it must not hold.
5. Only then activate it in the planner.

## Adding a gate type

1. Add it to `GateType` in `domain.py`.
2. Add the required approver roles and the required evidence in `gates.py`.
3. Decide whether agents may request it. Requesting is allowed; approving is
   not.

## Cost review

Weekly, the program lead reviews:

```bash
.venv/bin/python -m internal_ai.cli budget status
.venv/bin/python -m internal_ai.cli report
```

Look for spend concentrated in one agent, runs that were approved but never
produced a report, and jobs that consumed budget without producing an
artifact.
