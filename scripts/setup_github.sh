#!/usr/bin/env bash
# Run once by the lead from the repo root. Needs gh CLI: gh auth login
set -e
REPO_NAME="DRDO-ANC"
gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
OWNER=$(gh api user -q .login)

# Invite teammates (edit usernames, uncomment)
# for u in P2_USER P3_USER P4_USER P5_USER; do gh api -X PUT repos/$OWNER/$REPO_NAME/collaborators/$u -f permission=push; done

# Protect main: PR + 1 approval + CI must pass
gh api -X PUT "repos/$OWNER/$REPO_NAME/branches/main/protection" --input - <<JSON
{
  "required_status_checks": {"strict": true, "contexts": ["python"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 1},
  "restrictions": null
}
JSON

for l in dsp data realtime backend ai; do gh label create "$l" --force; done
gh issue create -t "P1: model survey + comparison notes" -l ai -b "Branch ai/model-survey. See docs/TASKS.md"
gh issue create -t "P2: spectral subtraction + Wiener" -l dsp -b "Branch dsp/spectral-wiener"
gh issue create -t "P2: LMS + NLMS (+ NLMS vs RLS note)" -l dsp -b "Branch dsp/lms-nlms"
gh issue create -t "P3: source datasets + licensing sheet" -l data -b "Branch data/sourcing"
gh issue create -t "P3: SNR mixer + manifest" -l data -b "Branch data/snr-mixer"
gh issue create -t "P4: mic loopback + latency test" -l realtime -b "Branch rt/loopback-test"
gh issue create -t "P5: Spring Boot skeleton + results API" -l backend -b "Branch backend/skeleton"
