# Contributing

## Rules

- `main` is protected. Nobody pushes directly. Everything goes through a Pull Request reviewed and merged by the lead (P1).
- One branch per task: `<area>/<short-name>`, e.g. `dsp/nlms-filter`, `data/snr-mixer`, `rt/loopback-test`, `backend/results-api`, `feature/phase-3-part-1-dccrn`, `feature/phase-3-part-2-ai-enhancer`.
- Small PRs (one task, ideally < 400 lines). Link the issue when one exists, using `Closes #12`.
- Commit messages should follow the existing convention: `feat:`, `fix:`, `docs:`, `test:`, or `chore:`. Examples: `feat(anc): add DCCRN baseline`, `feat(anc): add AIEnhancer integration`. No `update` / `final2`.
- Never commit datasets, audio files, model checkpoints, virtual environments, generated build artifacts, or secrets. Share datasets through a drive link documented in `docs/datasets.md`.
- Every performance number reported in documentation or a PR must come from a logged experiment. Include the configuration, result file, and relevant environment details in `experiments/` or the project's designated experiment records.
- Follow `docs/INTERFACES.md` and existing model interfaces. Do not silently change shared input/output shapes, sample rates, configuration keys, or other interfaces.

## Workflow

```bash
git checkout main
git pull origin main
git checkout -b <area>/<short-name>

# Make changes and run the relevant tests
python -m pytest -q
python -m ruff check .

git diff --check
git add <files>
git commit -m "feat(anc): describe the change"
git push -u origin <area>/<short-name>

# Open a PR on GitHub, fill in the PR template,
# and request review from the lead.
```

If `main` has moved, update your branch before merging:

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

Use `--force-with-lease` only on your own feature branch after a rebase, never on `main` or another contributor's branch.

## Testing and Code Quality

- Add or update tests for new functionality and bug fixes.
- Run `python -m pytest -q` and `python -m ruff check .` before opening a PR.
- Run `git diff --check` to catch whitespace errors.
- Report the actual test results, including skipped or expected-failure tests.
- For model or streaming changes, test tensor shapes, output length, state reset, and offline/streaming consistency where applicable.
- Report benchmark conditions honestly. Do not claim real-time or embedded performance without supporting measurements.

## Review Checklist (Lead)

- Does the implementation match `docs/INTERFACES.md`, including sample rate, shapes, and configuration keys?
- Are existing shared components and utilities reused instead of duplicated?
- Are tests or runnable scripts included, with instructions for reproducing the results?
- Are performance claims supported by recorded experiments and environment details?
- Is CI green? Are there any large files, secrets, or generated artifacts?
- Does the change stay within the task's scope and respect folder ownership?
- Are shared interfaces and existing functionality preserved?
- Is the PR focused, understandable, and free of unrelated changes?
- Squash-merge with a clean commit message after approval.

## Audio and Model Guidelines

- Use the project's existing sample-rate, STFT, validation, and tensor-shape conventions.
- Preserve the expected input/output format and audio length.
- Streaming implementations must handle frame processing and state reset correctly.
- Keep model-specific logic within the appropriate module and avoid duplicating common functionality.
- Record latency measurements with the hardware, input size, and measurement method. Separate model inference latency from total pipeline latency.

## Repository Hygiene

- Keep PRs focused on one task and avoid unrelated formatting or refactoring.
- Do not commit local environment files, credentials, datasets, audio recordings, or model checkpoints.
- Follow the existing folder ownership and review process.
- Do not push directly to `main` or rewrite shared branch history.
