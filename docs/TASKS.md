# Task assignment

P1 = lead/reviewer. Each task = one GitHub Issue + one PR.

## Sprint 1 — Foundation (Weeks 1–2)
| Who | Task | Branch | Done when |
|---|---|---|---|
| P1 | Repo + protections + INTERFACES; survey CRN/DCCRN | `ai/model-survey` | `docs/model-comparison.md` merged |
| P2 | Spectral subtraction + Wiener as `enhance()` | `dsp/spectral-wiener` | tests pass on synthetic noisy tone |
| P2 | LMS + NLMS; note on NLMS-only vs +RLS | `dsp/lms-nlms` | converges on synthetic noise-cancel test |
| P3 | Source clean speech + defence noise; licensing sheet | `data/sourcing` | `docs/datasets.md` lists source + license per set |
| P3 | SNR mixer + manifest | `data/snr-mixer` | script outputs manifest + mixed wavs |
| P4 | Mic loopback test with `sounddevice` | `rt/loopback-test` | script runs, round-trip latency reported |
| P5 | Spring Boot skeleton (`POST/GET /api/runs`) + DB | `backend/skeleton` | `mvn test` passes, CI builds it |

## Sprint 2 — Baselines & training skeleton (Weeks 3–4)
| Who | Task |
|---|---|
| P1 | Data loader on P3 manifest; DCCRN/CRN skeleton + training loop (overfit a tiny batch) |
| P2 | Tune 4 baselines; outputs on a small subset |
| P3 | Freeze split (`dataset-v1`); eval harness: SNR, SI-SNR, STOI, PESQ for any `enhance()` |
| P4 | Real-time passthrough pipeline (no AI) |
| P5 | Results dashboard (sort/compare runs); CI for Python + Java |

Later sprints (training → ONNX → hybrid → real-time demo → paper) follow the 5-phase plan; cut new Issues when the previous sprint merges.

## Rules
- Interface questions go in the Issue thread.
- Blocked > 1 day? Tell the lead.
- Weekly sync: each person demos what merged.
