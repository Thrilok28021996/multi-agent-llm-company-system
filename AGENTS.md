# Multi-Agent LLM Company System — repository guide

For what this project is, how to install it, and how to run it: `README.md`, `USAGE_GUIDE.md`, and `docs/`. This file only routes; it does not describe the system.

## Checks

- Tests: `python -m pytest -q`
- Interpreter used for local evidence runs: `/Volumes/personal/conda_envs/assi/bin/python3.12`
- CI: `.github/workflows/ci.yml` runs the same pytest command on every push and pull request, as the job named `test`.

## Rules for agent workers

- Implement only the scope in the approved plan. Do not edit `.github/workflows/`, dependencies, lockfiles, or unrelated tests unless the approved plan names them.
- Never read or print `.env` or any credential/auth file.
- Never push or merge. Direct pushes to `main` are blocked by branch protection; only the human merges pull requests.
- Report only what executed commands proved. No invented test results, screenshots, commits, or CI status.
