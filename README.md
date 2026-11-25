# How Do Algorithmic Recommendation, Information Exposure, and Emotion Shape Opinion Polarization?

**A Dual-Model Agent-Based Simulation with Rule-Based Agents and Large Language Model Agents**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 本仓库为学术论文的**可复现代码**。《算法推荐、信息暴露与情绪如何塑造舆论极化——基于规则多智能体与大语言模型智能体的双模型仿真研究》
>
> This repository contains the reproduction code for a study on how algorithmic recommendation shapes opinion polarization, using a **dual-model** design that compares rule-based agents against LLM-driven agents.

---

## 1. Research Question

Does the **strength of algorithmic recommendation** ($\alpha$) causally affect **group opinion polarization**? And does the answer hold when the behavioural mechanism changes from hand-coded update rules to a large language model?

## 2. Headline Finding

Raising the engagement weight $\alpha$ **lowers** the polarization index, and the direction holds across both agent types:

| Agent type | Effect of $\alpha$ on polarization index (PI) | Verdict |
|---|---|---|
| **Rule-based** (N=200, T=200, 500 runs) | standardized $\beta = -0.566$, $R^2=0.320$, $p<10^{-4}$ | strong, reproducible |
| **LLM (DeepSeek)** (N=30, T=30, 80 runs) | standardized $\beta = -0.405$, $R^2=0.164$, $p=1.9\times10^{-4}$ | significant, same sign |

The mechanism is **exposure widening**: raising $\alpha$ shifts the ranking anchor from "closest to you" toward "what everyone is reading", which lowers Top-$K$ homogeneity and slows the upward drift of polarization. The effect runs through exposure structure rather than emotional exposure (emotion channel: $b=-0.109$, $p=0.48$).

> **Prompt version matters.** Both results above use the **signed-feed prompt**: the LLM agent is told the *signed* mean stance of its recommended set (`feed_mean_opinion`), the same quantity a rule agent receives. With the earlier **unsigned-distance prompt** (only $\frac{1}{2}\overline{|o_i-p_j|}$), the LLM arm shows a null effect ($\beta=-0.031$, $p=0.782$): the agent cannot tell "same side but more extreme" from "opposite side", so $\alpha$ cannot move it. That contrast is itself a finding — the direction of the exposure channel has to be *perceivable* before ranking strength can act. See **Prompt variants** under Section 4.

> **Boundary.** The suppression is a *slowdown of the upward drift*, not a reduction of the polarization level; it fades beyond $T \approx 100$. The claim is stated for $H>0$: at $H=0$ the system degenerates (PI$\approx 2\times10^{-6}$), so no "reversal" is defined.

## 3. Model Design

### Recommendation score

```
Score(i, j) = H * sim(i, j) + alpha * engagement(j) + gamma * emotion(j)
```

- `H = 1.0` **fixed** — the similarity coefficient is *not* tunable.
- `alpha` in `[0, 2]` — weight on engagement / social proof.
- `gamma` in `[0, 2]` — weight on emotional content.
- Top-`K` (`K = 10`) items are pushed to each agent; an agent's own posts are excluded from its own candidates.

> **Why H is fixed.** Similarity is always weighted most heavily, so the Top-K set is *structurally dominated* by the K nearest items. Homophily is therefore a property of the mechanism rather than a free variable. `alpha` does not overturn this: it shifts the composition of the Top-K set only slightly, but measurably — in the rule arm the exposure-homophily index moves from 0.9998 (`alpha=0`) to 0.9974 (`alpha=2`). Small as it is, that shift is what carries the effect.

### Per-step loop

Each step, every agent performs three actions:

1. **Recommendation** — select Top-`K` items from the shared content pool (FIFO, capacity 5000).
2. **State update** — update opinion and emotion from the recommended feed.
3. **Expression decision** — decide whether to post; posts enter the shared pool.

> **No third-party ABM framework.** The scheduler is hand-written (`SimulationModel` in `model/model.py`), not built on Mesa or any other ABM library. The LLM arm issues one decision call per agent per step (~900 calls per N=30, T=30 run), which requires batch-concurrent dispatch, a shared immutable start-of-step snapshot, checkpoint/resume, and a per-call JSONL audit trail. The reasoning is documented in the thesis (Section 3.1).

### Two agent types

| | Rule-based | LLM-based |
|---|---|---|
| Opinion update | `mu = 0.03` toward feed mean | model-generated |
| Emotion update | `rho = 0.2`, sensitivity `lambda_i` in `[0.5, 1.5]` | model-generated |
| Decision | probabilistic expression rule | strict JSON decision |
| Receives | Top-K snapshot | **identical** Top-K snapshot |

The LLM arm consumes a **bit-identical recommendation snapshot** to the rule arm, so any difference is attributable to the decision mechanism alone.

## 4. Experimental Design

### Rule arm
Full factorial: **5 levels of `alpha` x 2 levels of `gamma` x 50 repeats = 500 runs** (200 agents x 200 steps).
Plus a reduced-scale control (30 agents x 30 steps x 5 repeats) and a spiral-of-silence extension (1,159 runs).

### LLM arm
- Main: `N=30` agents, `R=10` repeats per condition, 30 steps + 100-step horizon
- Scale check: `N=100`
- Cross-model replication: **Qwen3-Flash** (three arms)
- Online ablations: `no_post` / `no_emotion` / `positive_control` (positive control freezes stance delta)
- Zero-cost offline counterfactuals: **decision reallocation replay** (bit-verified to 1 ulp)

### Prompt variants (important for reproduction)

Two prompt variants exist and they give different answers:

| Variant | Field given to the agent | LLM-arm effect of $\alpha$ |
|---|---|---|
| **signed-feed** (used in Section 2) | `feed_mean_opinion`: signed mean stance of the recommended set, $[-1,1]$ | $\beta=-0.405$, $p=1.9\times10^{-4}$ |
| unsigned-distance (legacy variant, `LLM_FEED_SIGNED=0`) | `mean_stance_distance`: $\frac{1}{2}\overline{|o_i-p_j|}$, unsigned | $\beta=-0.031$, $p=0.782$ |

The rule agent has always received the **signed** feed mean (`o_feed = mean(p.opinion)` in `model/model.py`), so the unsigned LLM prompt is an *information-set asymmetry* rather than a behavioural difference. The signed variant restores the missing quantity and is the one used in the paper.

The signed variant was first implemented and validated in a separate probe directory (`probe_prompt_clarity.py`, `results_prompt_direction/`); see `诊断_信息集不对称_20261008.md` and `判定结果_方向版_20261008.md`. **It is now merged into the main code path and is the default** (`config.py: LLM_SIGNED_FEED`, environment variable `LLM_FEED_SIGNED`). The batch prompt uses the same labeled field table as the paper (7 fields when signed: `id/opinion/emotion/activity/feed_mean_opinion/feed_stance_distance/feed_emotion`), and `build_prompt()` (single-agent path) also includes the signed field. `_feed_features()` adds `mean_feed_opinion` when the flag is on.

```bash
# default: signed-feed prompt (reproduces the paper)
python main.py --mode llm

# legacy: unsigned-distance prompt (reproduces the null arm, beta=-0.031)
LLM_FEED_SIGNED=0 python main.py --mode llm
```

## 5. Metrics

PI is decomposed into **4 dimensions / 13 indicators**, including:

| Abbr. | Meaning |
|---|---|
| **PI** | polarization index (primary outcome) |
| **BC** | bimodality coefficient (co-primary in the LLM arm) |
| **SR** | silence rate |
| **ER / EM** | emotion range / emotion mean |
| **HE / EE** | homophily of exposure / echo-chamber index |

Shape family: **39/39 TOST** equivalence tests. Measurement limitations are disclosed explicitly in the paper (see section 10).

## 6. Repository Layout

```
.
├── main.py                       # CLI entry point
├── config.py                     # configuration loading
├── model/                        # agent, post, dynamics, recommendation, ABM
├── experiments/                  # rule & LLM experiment drivers
├── llm/                          # LLM client, prompt templates, robustness client
├── metrics/                      # polarization & exposure metrics
├── analysis/                     # statistical analysis, figures, robustness reports
├── utils/                        # seed control, cost guard, result IO, xlsx writer
├── scripts/                      # batch orchestration, monitoring & paper pipeline
│   ├── experiment/               #   stage runner, guards, cost monitor, launchers
│   ├── paper/                    #   interaction tests, table verification, PDF/DOCX build
│   └── probe/                    #   prompt-clarity probe and direction-version analysis
├── run_rule_main_fixed.py        # rule-arm main experiment
├── run_rule_spiral.py            # spiral-of-silence extension
├── run_llm_gradient_deepseek.py  # LLM-arm main experiment
├── run_llm_robust_grid.py        # LLM robustness grid
├── run_llm_ablation.py           # online ablations
├── experiment_llm_robust.py      # robustness pipeline
├── experiment_rule_small.py      # reduced-scale control
└── requirements.txt
```

### Batch orchestration scripts (`scripts/`)

The multi-hundred-run batches reported in the paper were driven by a small
orchestration layer that is included here for full transparency:

| Area | Representative files | Purpose |
|---|---|---|
| Runner | `experiment/run_stage_direction.py` | unified stage runner (`--stage qwen-t100`, `n100b`, …), checkpoint/resume |
| Survival | `experiment/_guard_qwen.sh`, `_guard_n100b.sh` | watchdog loops that restart the runner if it dies; heartbeat files |
| Cost / quality | `experiment/_monitor.py`, `_fb_watch.py` | fallback-rate audit (>2% triggers same-seed reroll) and cost ceiling / balance red lines |
| Launchers | `experiment/_launch_*.py`, `_launch_scheduler.ps1` | guard-proof, duplicate-safe launches |
| Causal tests | `paper/three_way_interaction.py`, `time_interaction.py`, `model_interaction.py`, `rule_time_interaction.py` | OLS interaction models reported in the paper |
| Data checks | `paper/verify_tables_new.py`, `verify_replay.py`, `verify_newdata.py` | per-table and bit-level replay verification |
| Aggregates | `paper/build_aggregates.py` | rebuild `results/aggregates/` from raw run outputs (paper-口径 datasets only) |
| Build | `paper/build_toc.py`, `build_outputs.py`, `final_audit.py`, `md2pdf.py` | two-pass TOC pagination, HTML→PDF/DOCX, format audit |

**Path configuration.** These scripts read their data/output roots from
environment variables with sensible defaults, so they run unchanged on the
original machine while remaining portable:

```bash
export ABM_PROJ_ROOT=/path/to/final-code-and-data   # repo root
export ABM_PROBE_ROOT=/path/to/probe-dir            # probe/output scratch dir
export ABM_PYTHON=$(which python)                   # interpreter used to spawn subprocesses
```


## 7. Quick Start

```bash
# 1. environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 2. credentials
cp .env.example .env              # then fill in your own API key

# 3. rule-based experiment (no API key needed)
python main.py --mode rule

# 4. LLM experiment (requires a valid API key)
python main.py --mode llm
```

`rule` and `debug` modes need **no** LLM configuration. `llm` and `robustness` modes **fail fast** if `LLM_API_KEY` / `LLM_API_ENDPOINT` / `LLM_MODEL` are absent — placeholder results are never mixed into experimental data.

### Resumption
`--mode rule` verifies the final step count of every run and only re-runs missing or interrupted ones.
`--mode llm` validates per `alpha x gamma x repeat` condition; each run writes `llm_run_<id>.calls.jsonl` so an interrupted run resumes from the last complete step.

### Resource control
```bash
RULE_WORKERS=1            # limit rule-arm parallelism (default: up to 4)
LLM_PARALLEL_WORKERS=4    # LLM request concurrency
```
Falls back to serial execution automatically if subprocess creation is unavailable.

## 8. Reproducibility Notes

- **Seeding** is explicit; `utils/seed.py` centralises seed control.
- **Cost guard** (`utils/cost_guard.py`) prevents runaway API spend.
- **Offline replay** reproduces the decision-reallocation counterfactual with **zero API cost** and bit-level verification (max deviation 1.1e-16, i.e. 1 ulp).
- A single 30-step / 30-agent run costs roughly **CNY 0.06** on the DeepSeek API (reference only; pricing changes).

## 9. Data Availability

Data are published in three layers, so that every table and figure can be recomputed while the raw logs stay out of git:

| Layer | Content | Size | Where |
|---|---|---:|---|
| **Aggregates** (in repo) | `results/aggregates/` — one row per run (last step), per-condition means with 95% CI, OLS effect sizes, and the 175-run parameter-sensitivity scan. 18 datasets: 3 rule arms (`rule_main_d4`, `rule_small_d4`, `rule_spiral`) and 15 LLM arms (signed-feed paper arms + the two legacy unsigned prompt variants) | **~0.42 MB** | this repository |
| Per-run time series | `llm_robust_summary.csv` / VPI time series for each arm | ~180 MB | GitHub Release / Zenodo (DOI on acceptance) |
| Raw call logs + checkpoints | one JSONL per LLM run, resumable checkpoint state (~700 MB total) | ~510 MB | Zenodo archive (DOI on acceptance) |

`results/aggregates/` is rebuilt from raw outputs with `python scripts/paper/build_aggregates.py` (`ABM_PROBE_ROOT` points at the probe directory; `sensitivity_runs.csv` keeps the shipped 175-run scan). It is the minimum needed to reproduce Tables 1–2, Figures 1–5 and supplementary tables S1/S3/S6/S10:

```
results/aggregates/runs_laststep.csv      # 3,429 runs × (dataset, α, γ, seed, run_id, steps, PI, HE, EE, EM, SR, spiral_c, spiral_m, mean_exposure_distance)
results/aggregates/pi_by_condition.csv    # per dataset × α × γ: n, mean, SD, 95% CI
results/aggregates/effect_sizes.csv       # per dataset × γ: raw slope, SE, t, p, R², standardized β
results/aggregates/sensitivity_runs.csv   # 175-run parameter perturbation scan
```

Only paper-口径 batches are included. Deprecated batches are quarantined outside the repo and excluded: early Qwen `r10/_fix/_fix_retry`, `slope_boost*`, `check`, `fb_fix`, `llm_gradient_r20`, plus `llm_decouple_*`, `llm_robust_grid` and the old unsigned-prompt ablation / N=100 / random-feed batches. Time-series directories and `*.jsonl` call logs are excluded from git by design — a few thousand small files would bloat the history without being reviewable as diffs.

## 10. Known Measurement Limitations

Disclosed in the paper for transparency:

1. **HE resolution** — the homophily-of-exposure index is compressed near its upper bound (cross-condition range 0.0027 in the rule arm) because the similarity coefficient is fixed at 1.0.
2. **EE == EM** — at steady state the exposure and emotion means coincide (max `|EE - EM| = 2.3e-5`), making the chain `alpha -> EE -> EM -> Y` mathematically unidentified.
3. **VPI == PI** — the variance-based index degenerates to the polarization index (max `|VPI - PI| = 0`) once all agents have posted at least once.

These constrain the identification of individual mediation channels but do not affect the direction of the reported conclusions. Fixes are specified in the paper.

## 11. Citation

```bibtex
@article{anonymous2026algo,
  title   = {How Do Algorithmic Recommendation, Information Exposure, and Emotion
             Shape Opinion Polarization? A Dual-Model Agent-Based Simulation
             with Rule-Based and Large Language Model Agents},
  author  = {Anonymous},
  year    = {2026},
  note    = {Code available at https://github.com/your-user/your-repo}
}
```

## 12. License

MIT — see [LICENSE](LICENSE).
