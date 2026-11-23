#!/usr/bin/env bash
# Per-task gate for the hatwu/m3.1-pro-infx agent-team overlay.
#
# Same test scope as CI (.github/scripts/run_unit_tests.sh): every test under
# tests/ except tests/plugin (needs sglang/vllm) and the server integration
# test (needs a GPU and a model). Then black over the repository, and ruff
# over the Python files this branch changes against its merge base with
# BASE_REF, since the repository as a whole carries pre-existing ruff findings.
# The gate pins ruff 0.16.0 so that a newer ruff cannot turn it red on code no
# task touched.
#
# Run from the tree's root, in a git checkout, in a Linux Python 3.12
# environment set up as the "Install dependencies" step of the unit-tests job
# in .github/workflows/pre-checks.yaml does, plus black and ruff==0.16.0.
#
# Env:
#   BASE_REF  ref the changed-file list is taken against
#             (default: origin/hatwu/m3.1-pro-infx)
set -euo pipefail

BASE_REF="${BASE_REF:-origin/hatwu/m3.1-pro-infx}"

python -m pytest tests/ \
  --ignore=tests/plugin \
  --ignore=tests/entrypoints/test_openai_server.py \
  -q -p no:cacheprovider

black --check .

python -m ruff --version | grep -q '^ruff 0\.16\.0$' || {
  echo "gate.sh: ruff 0.16.0 required (pip install ruff==0.16.0)" >&2
  exit 1
}
base=$(git merge-base HEAD "$BASE_REF")
mapfile -t changed < <(git diff --name-only --diff-filter=ACMR "$base" HEAD -- '*.py')
if ((${#changed[@]})); then
  python -m ruff check "${changed[@]}"
fi
