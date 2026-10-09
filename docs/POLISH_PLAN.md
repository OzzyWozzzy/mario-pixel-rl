# Project polish plan

Goal: a compact student ML project with reliable runs, readable code, reproducible
evaluation, and clear documentation of what has actually been tested.

## Order of work

1. Isolate fresh and smoke runs. Validate YAML and save run metadata.
2. Fix frame pooling and seeding. Handle terminal observations and checkpoint RND.
3. Share environment configuration between training and playback. Save evaluation JSON.
4. Add focused tests, lint/format configuration, and a basic GitHub checks workflow.
5. Rewrite the README and update the implementation guide. Verify the complete pipeline.

## Acceptance checks

- Unit tests exercise custom rewards, episode boundaries, config validation, run
  isolation, curiosity restoration, and fresh optimizer versus resume behavior.
- Baseline and curiosity smoke training save usable checkpoints and metadata.
- Curiosity state accompanies periodic, best, and final checkpoints.
- A saved model loads for evaluation and produces metrics and playable video.
- Generated training artifacts stay outside normal source control.
- Documentation distinguishes verification from measured learning performance.

## Local validation results

Completed the five implementation steps above. Validation on macOS Apple Silicon,
Python 3.13:

- Focused unit tests pass, including checkpoint restoration and model lifecycle.
- Baseline and full-game RND smoke runs each trained for 2,048 transitions.
- Periodic, best, and final RND checkpoints have matching curiosity state files.
- A resumed RND run continued from 2,048 to 2,176 transitions.
- Standalone evaluation wrote episode JSON and a 240×256 RGB video at 15 FPS.
- Ruff lint and formatting checks pass locally.

The GitHub workflow is configured but has not been run on GitHub. These checks
validate the workflow, not the policy's learning quality.

## Follow-up experiment

Run PPO and PPO + RND on stage 1-1 with identical base rewards, actions, rollout
settings, evaluation protocol, and transition budgets. Change only the curiosity
flag. Use independent training seeds as compute permits, retain individual curves,
and report completion, game score, wall-clock cost, and limitations. This experiment
is separate from code validation and requires full training runs.

## Comparison workflow

Added a matched experiment runner and report exporter. A two-seed pilot completed
four runs with 2,048 transitions each. Final evaluation used three episodes per
policy. No pilot policy completed stage 1-1, as expected for this short budget.
Reports include individual curves, final metrics, and measured training time.
The main experiment remains a separate run with the protocol in EXPERIMENTS.md.
