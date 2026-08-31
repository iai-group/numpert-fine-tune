# verdict_stripped — the final cleaned QuanTemp corpus

**`{train,val,test}_claims_quantemp.json` in this directory is THE
production corpus** for all downstream stages (make_binary → Stage 1d
annotation → Stage 2 NLI → Stage 3 perturbation): QuanTemp evidence docs
with fact-checker verdict + footer boilerplate stripped.

**15,480 docs** (train 9,906 / val 3,080 / test 2,494). Produced from
pristine `../QuanTemp-main-data/` by the `verdict_strip` pipeline —
deterministic rule pass, then hybrid LLM locate + audit gate
(`greedy_v3`) — SLURM job 467303 (audit 467304), minus 34 dropped
records (`DROP_REASONS` in
`src/data_preprocessing/raw_dataset_preprocessing/verdict_strip/rules.py`).
Pipeline + provenance docs: repo `README.md` §3–§6,
`docs/verdict_strip_pipeline.md`, `docs/greedy_gate_experiment.md`.

## Layout

| Path | What |
|---|---|
| `{train,val,test}_claims_quantemp.json` | the final corpus (tracked) |
| `audit_reports/` | eyeball HTMLs: `audit_report.html` (100-sample, extent-correct highlighting), `audit_gate_rejections.html`, `audit_problem_samples.html` |
| `diff_logs/` | `llm_diff_log.jsonl` — per-record kept/rejected/unlocatable spans from the hybrid LLM pass (replayable; spans are per-handler-relative) |
| `reaudit/` | gate-prompt A/B artifacts (v1 vs v3 on the v2 rejections) |
| `run_keys/` | record-key lists used by the retry/problem-audit jobs |
| `rulebased/` | **untracked, regenerable** (~13 min CPU: `clean_splits` over `../QuanTemp-main-data/`) — rule-pass output, input to a hybrid job |
| `hybrid/` | **untracked** — working dir a hybrid SLURM job writes into; promote its splits here after verification |
| `archive/` | **untracked** — superseded runs (v1/v2 gates, baselines) + stale oversized audit HTMLs |

Never edit `../QuanTemp-main-data/` (pristine, positional ids). To
regenerate anything here, follow repo `README.md` §3 (run order) and §5
(verification loop) — do not hand-edit the corpus files.
