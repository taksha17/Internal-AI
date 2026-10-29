## What is left

The advice rows name small, 1B, and 7B, with hours and whether each size can run here. The train button does not read them. A from-scratch run can start on a size the advice marks waiting. The trainer that runs is the one that starts from random weights.

## Change

- The train button and `internal-ai train` take a size: `small`, `1b`, or `7b`.
- They read `compute_advice:{dataset_id}`. No advice stored waits with "record what this corpus needs".
- A size whose row is not runnable is refused, and the error includes that row's GPU-hours. A runnable size uses the existing from-scratch trainer on the released dataset.
- The model card records `size` and `planning_gpu_hours`. It does not set `track` to `alpha`.
- Promotion stays as it is: a frozen eval of questions a domain expert filed, a passing report, and an approved gate.

## Proof

- Advice that marks 7B waiting raises on a 7B train and does not create a model.
- Advice that marks small runnable lets the existing tiny run complete.
- That run's card says `small` and has no alpha track.
- A train with no advice stored raises before any step.

## Out of scope

Do not register `/media/taksha/New Volume2/TT Resumes`. Do not load the pinned Qwen base. Do not start a run on the 1650.
