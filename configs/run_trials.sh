#!/usr/bin/env bash
# Start three Harbor trials on the packaged task using Claude Code for both models.
# Usage: bash configs/run_trials.sh {opus|sol} [additional Harbor options]
set -euo pipefail

case "${1:-}" in
  opus)
    trial_model=anthropic/claude-opus-5-5
    trial_effort=max
    ;;
  sol)
    trial_model=openai/gpt-6.1-sol
    trial_effort=xhigh
    ;;
  *)
    echo "Usage: bash configs/run_trials.sh {opus|sol} [additional Harbor options]" >&2
    exit 2
    ;;
esac
shift

trial_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec harbor run \
  --path "$trial_repo_root/tasks/house-thermal-model" \
  --agent claude-code \
  --model "$trial_model" \
  --ak "reasoning_effort=$trial_effort" \
  --n-attempts 3 \
  --env docker \
  "$@"
