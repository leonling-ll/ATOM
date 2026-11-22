---
repo: zhuyuhua-v/ATOM-M3
integration_branch: hatwu/m3.1-pro-infx
design_entry: recipes/MiniMax-M3-Pro.md
# CLAUDE.md: CPU tests mock AITER and torch.cuda; black and ruff are CI enforced
gate_task: python -m pytest tests/ -q && black --check . && ruff check .
# gate_post_land: add the GPU accuracy gate (GSM8K 64-shot, 1319, >= 0.93) once M0 item 4 delivers its script
never_touch: ROCm/ATOM, ROCm/aiter
new_tests_dir: tests
publish_language: English
conversation_language: Chinese
max_tasks: 4
task_label: m31pro
---

Target and constraints: issue #7. Every brief and PR follows its section 4 (InferenceX rules) and section 5 (accuracy gate).

Recipe first: weights, quantization, launch flags and DSpark k=7 follow recipes/MiniMax-M3-Pro.md. A deviation needs before/after data in the PR.

Landing: PRs into hatwu/m3.1-pro-infx land automatically after a reviewer agent's APPROVE. When the work is accepted, one PR from hatwu/m3.1-pro-infx into yuhzhu/m3.1-pro goes to zhuyuhua-v for review; agents never push to yuhzhu/m3.1-pro.

Kernel work lands in zhuyuhua-v/aiter-M3 by PR, merged by its owner, not by `run`. The ATOM-side task depends on that PR and pins the aiter commit.

GPU: one 8xMI355X node per person. Unit tests and microbenchmarks may split GPUs with HIP_VISIBLE_DEVICES; end-to-end benchmarks and accuracy runs take the node lock and run one at a time.

Never push to ROCm upstream repositories before the model is public.
