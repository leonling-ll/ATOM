---
repo: zhuyuhua-v/ATOM-M3
integration_branch: hatwu/m3.1-pro-infx
design_entry: recipes/MiniMax-M3-Pro.md
# CI scope (.github/scripts/run_unit_tests.sh) + black + ruff 0.16.0 on changed files; see scripts/m31pro/gate.sh
gate_task: bash scripts/m31pro/gate.sh
# gate_post_land: add the GPU accuracy gate (GSM8K 64-shot, 1319, >= 0.93) once M0 item 4 delivers its script
never_touch: ROCm/ATOM, ROCm/aiter
new_tests_dir: tests
publish_language: English
conversation_language: Chinese
max_tasks: 4
task_label: m31pro
---

Target and constraints: issue #7. Every brief and PR follows its section 4 (global constraints), section 5 (accuracy gate) and section 11 (InferenceX / SemiAnalysis rules: no FLOP reduction, the DSpark draft as released and excluded from online quantization, golden AL per draft length with the chat template, no integer-quantized all-reduce, CPU-DRAM-only KV offload within the per-server cap, the stream-interval cap, images run unpatched). A PR that cannot meet section 11 says so and escalates.

Recipe first: weights, quantization, launch flags and DSpark k=7 follow recipes/MiniMax-M3-Pro.md. A deviation needs before/after data in the PR.

Landing: PRs into hatwu/m3.1-pro-infx land automatically after a reviewer agent's APPROVE. The upstream and final merge branch is yuhua/m3.1-pro: it is merged into hatwu/m3.1-pro-infx daily (merge commit, never rebased or squashed), and one standing PR from hatwu/m3.1-pro-infx into yuhua/m3.1-pro goes to zhuyuhua-v for review; agents never push to yuhua/m3.1-pro.

Kernel work lands in zhuyuhua-v/aiter-M3 by PR, merged by its owner, not by `run`. The ATOM-side task depends on that PR and pins the aiter commit.

GPU: one 8xMI355X node per person. Unit tests and microbenchmarks may split GPUs with HIP_VISIBLE_DEVICES; end-to-end benchmarks and accuracy runs take the node lock and run one at a time.

Never push to ROCm upstream repositories before the model is public.

Gate: `gate_task` runs `scripts/m31pro/gate.sh` in a Linux Python 3.12 environment set up as the "Install dependencies" step of the `unit-tests` job in `.github/workflows/pre-checks.yaml` does, plus `black` and `ruff==0.16.0` (for example a `python:3.12` container on the GPU node). The tree must be a git checkout: ruff checks the files changed against the merge base, and one test calls `git rev-parse HEAD`.

Performance PRs: every PR that can change model output or serving performance shows, at its head and at its base, GSM8K 64-shot with the recipe's GSM8K command, passing the accuracy gate of #7 section 5 and a 15-minute agentic proxy run (`scripts/m31pro/proxy_bench.sh` at the concurrency its task targets; c32 for the c32 tasks), and states the change in tok/s/GPU and p90 E2E-normalized interactivity in the PR body.
