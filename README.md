# DRDO-ANC

AI-assisted adaptive noise cancellation / speech enhancement for defence and mission-critical communication.
Hybrid of classical adaptive filtering (LMS/NLMS) and deep-learning speech enhancement (candidate: DCCRN), benchmarked on objective metrics.

> We *propose to investigate* this architecture. No performance or novelty claim is made until backed by experiments and a prior-art search.

## Repo layout
| Folder | Owner | Contents |
|---|---|---|
| `ai_engine/models, training` | P1 | models, training, ONNX export |
| `signal_processing/` | P2 | spectral subtraction, Wiener, LMS, NLMS, hybrid stage |
| `ai_engine/data, evaluation` | P3 | dataset pipeline, metrics harness |
| `ai_engine/inference/realtime` | P4 | streaming pipeline, latency tools |
| `java-backend/` | P5 | Spring Boot experiment API + results DB |
| `experiments/` | all | configs + results |
| `docs/` | all | tasks, interfaces, notes |

## Start here
1. Read `CONTRIBUTING.md` (branch + PR rules).
2. Read `docs/INTERFACES.md` (shared audio/config contracts).
3. Find your tasks in `docs/TASKS.md` and the GitHub Issues assigned to you.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```
