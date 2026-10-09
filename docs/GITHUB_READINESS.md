# GitHub readiness

The project is ready for an initial source commit after the local checks below.
This is a working visual RL project with historical gameplay and an experiment
workflow. A larger controlled benchmark remains future work.

## Verified locally

- Clean editable installation with development and plotting dependencies.
- Fifteen tests pass in the clean Python 3.13 environment.
- Ruff lint and formatting checks pass.
- Two subprocess workers train PPO with RND and save periodic, best, and final state.
- Saved models load for evaluation and video recording.
- The matched two-seed pilot runs and exports figures and metrics.
- Historical checkpoints are evaluated with deterministic and sampled actions.
- Documentation links and configuration paths resolve.
- Git ignore rules exclude runs, model weights, caches, and the virtual environment.

Testing found and fixed stage-clear flags lost inside action repeats, rollout
metrics reset too early, and an incorrect README configuration path.

## Commit contents

Include source, configs, tests, project metadata, the GitHub workflow, documentation,
and the small reports under results/. Keep the full historical recordings and
checkpoint ZIPs outside the source commit. Checkpoint hashes in the historical
report identify the evaluated models.

The source snapshot is prepared for GitHub publication. The workflow runs on pushes
and pull requests. Its status on GitHub is the final check for the hosted Linux runner.

## Research scope

Historical results show that trained policies reach stage 1-2. They are selected
checkpoints with incomplete training provenance, not a controlled RND comparison.
The 2,048-transition pilot is too short to support a learning advantage claim.
Use the protocol in EXPERIMENTS.md for the next measured comparison.
