# MiniMax-M3 on CaP-Bench++ LIBERO

This checkout's `minimax-libero` branch contains the CaP-X Agent0 adapter used
to validate MiniMax-M3 on the CaP-Bench++ LIBERO suites.

## Protocol

- MiniMax-M3 is the only model, at temperature `1.0`.
- RGB is sent directly to the model; no VDM, image differencing, temperature
  ensemble, or multi-model ensemble is enabled.
- Each trial uses the M4-style multi-turn Python execution loop. The request
  payload explicitly accumulates and resends the complete history, including
  the concrete BDDL goal, observations/RGB attachments, prior Python code,
  stdout/stderr, and simulator feedback.
- Success is determined only from simulator metrics (`reward` and
  `task_completed`) plus retained telemetry, primitive/code trace, and video.
  Model prose is not a success signal.

The two source commits are:

- `50a3cff`: preserve the concrete LIBERO goal and explicit multi-turn history.
- `5149270`: accept MiniMax direct-Python continuations while retaining the
  simulator success contract.

## Reproduction

The workspace-level runner scripts live in the sibling experiment workspace at
`LIBERO/tools/capbenchpp_20trial/`. From the workspace root, create a fresh
output directory and freeze the 60-entry manifest:

```bash
cap-x/.venv/bin/python LIBERO/tools/capbenchpp_20trial/build_manifest.py \
  --root LIBERO/experiments/<run-id>
cap-x/.venv/bin/python LIBERO/tools/capbenchpp_20trial/validate_manifest.py \
  --root LIBERO/experiments/<run-id>
```

Run all 60 entries serially, with 20 trials per entry:

```bash
cap-x/.venv/bin/python LIBERO/tools/capbenchpp_20trial/run_matrix.py \
  --mode full --root LIBERO/experiments/<run-id> \
  --max-infra-attempts 10 --trial-timeout-seconds 1800 \
  --attempt-grace-seconds 30
```

The runner enumerates every manifest entry and trial ID, writes an independent
evidence directory, checks the four persistent services before launch, waits
for recoverable service outages, and continues serially after normal policy
failures. It stops on an unrecoverable infrastructure budget, an uncaught
runner error, or external process termination. `progress.json`, `runner.log`,
metrics, prompts/responses, code, telemetry, and videos are the recovery and
audit record. A separate operator may inspect these files, but no second
runner should be started for the same output root.

## Smoke gate

Before a full matrix, the dedicated smoke selection uses two tasks for each
scene/condition: one retained historical-success task and one distinct task
with the highest paper score. The current 12 x 20 smoke completed `240/240`
valid trials with `49` simulator successes. Its detailed per-task comparison
is kept in the workspace experiment output, not mixed into a later full run.

## Evidence and incidents

Never delete failed attempts. External MiniMax TLS failures have no simulator
result and must be reported as API infrastructure incidents; they are not
policy failures. A local service error inside a trial that later produces valid
metrics remains annotated in that valid trial. The full result denominator is
the number of trials with valid simulator metrics, and any missing trials must
be explicitly replaced or reported.
