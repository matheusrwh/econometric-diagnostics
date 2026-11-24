---
name: econometric-diagnostics
description: Diagnose and stress-test an estimated econometric model (panel, time series or cross section, OLS or IV/2SLS including shift-share/Bartik instruments) in Python. Replicates the user's baseline, runs the specification and diagnostic tests that fit the data structure, decides the appropriate inference, runs a one-change-at-a-time robustness battery, and writes LaTeX tables plus an interpreted report with a methodology paragraph. Use when the user asks to validate, diagnose, check the robustness of, or report the tests for a regression they have estimated (statsmodels, linearmodels, or a Stata/R model that can be re-expressed as a formula).
---

# Econometric diagnostics and robustness

You are acting as a careful applied econometrician reviewing a colleague's model before
submission. The scripts compute; **you** decide which tests matter, what they imply and
what to recommend. Never let the scripts' flags replace reading the numbers.

## Ground rules

1. **Do not modify the user's code or data.** Everything you write goes to
   `<project>/reports/diagnostics/<model_set>/` (or a folder the user names).
2. **Replicate before you diagnose.** The baseline must reproduce the user's own
   coefficients, standard errors and N. If it does not, stop and reconcile the sample
   (filters, exclusions, missing values, inf values) before running anything else.
3. **Decide the inference before you look at robustness.** Choose the recommended standard
   errors and estimator from the diagnostic tests (step 4), write that decision down, and
   only then run the robustness battery. Never recommend a variant because it is
   significant.
4. **Report everything you ran**, including variants that weaken the result.
5. **Robustness is not heterogeneity.** A coefficient that differs between sub-periods may be
   the finding (parameter change), not a failure. Say which is which.
6. **No causal language the design cannot support.** Diagnostics test statistical
   assumptions; they do not make an OLS/FE coefficient causal.

## Scripts (in `scripts/`, run with `python <script> <config.json>`)

| Script | Structure | What it does | Output |
|---|---|---|---|
| `inspect_data.py` | all | structure, N, T, balance, gaps, missing/inf, extreme values, within-variation share | `inspection.json` |
| `panel_diagnostics.py` | panel | F entity/time effects, BP-LM, Hausman, Mundlak, Wooldridge AR(1), modified Wald, Pesaran CD, Fisher-ADF unit roots, VIF, outliers; `--skip-unit-root` | `panel_diagnostics.json` |
| `robustness.py` | panel | baseline + inference / estimator / sample variants + leave-one-entity-out jackknife + verdict; `--no-jackknife` | `robustness.json` |
| `ols_diagnostics.py` | time series, cross section | diagnostics **and** robustness for OLS | `ols_diagnostics.json` |
| `iv_diagnostics.py` | IV / 2SLS (cross section, pooled) | first stage (effective F, robust/KP F, Cragg-Donald, Sanderson-Windmeijer), reduced form, Anderson-Rubin set, control-function endogeneity, Hansen J / Sargan / C; shift-share: AKM and AKM0, Rotemberg weights, effective number of shocks, sum of shares; robustness + verdict; `--no-jackknife` | `iv_diagnostics.json` |
| `to_latex.py` | all | booktabs tables; `--lang en` or `--lang pt` | `table_diagnostics.tex`, `table_robustness.tex` (`table_iv_*.tex` for IV) |

Requirements: `pandas numpy scipy statsmodels linearmodels patsy` (see `requirements.txt`).
`python tests/validate_iv.py` checks the IV code against published textbook results and
exact identities (add `--mc 500` for a Monte Carlo of shift-share inference); run it after
installing or changing the scripts.
Use the project's own environment if it has them; otherwise ask before installing anything.
The jackknife refits the model once per entity (about 1 s per fit for a two-way FE panel
with ~3,000 obs); use `--no-jackknife` while iterating and run the full battery once.

## Workflow

### 1. Understand the model
Read the user's estimation script (or ask for the specification). Extract:
data file, entity and time identifiers, sample filters and exclusions, the formula for each
model, fixed effects, covariance type, and the **key coefficient(s)** the paper is about.
Note the reported results (table or `.txt` output) — you need them for step 2.

### 2. Write the config and replicate
Create `config.json` in the output folder; the schema is in
`references/config_schema.md`. Run `inspect_data.py`, then confirm that N per model matches
the user's results. The `Baseline` row of `robustness.py` (or `ols_diagnostics.py`) must
reproduce their coefficient and standard error to the reported precision. If it does not,
fix the config — never proceed with a baseline that differs from the paper.

Also read `inspection.json` for: unbalancedness, entities with gaps, regressors with low
within variation (FE will be imprecise), and heavy tails (excess kurtosis, values beyond
5×IQR).

### 3. Run the diagnostics
- Panel: `panel_diagnostics.py config.json`
- Time series / cross section: `ols_diagnostics.py config.json` (includes robustness)
- IV / 2SLS: `iv_diagnostics.py config.json` (includes robustness). Formula with a bracket
  block, `y ~ 1 + w + [x ~ z1 + z2]`. Check the line `numerical cross-check vs
  linearmodels` (should be ~1e-12) and that the baseline matches the paper.
- **Shift-share (Bartik) IV**: ask the user for the exposure (shares) and the shocks, not
  only the constructed instrument; without them the shift-share block cannot run. Add a
  `shift_share` block to the model (see `references/config_schema.md`). The script rebuilds
  the instrument as Σₖ sᵢₖ gₖ and stops if it does not match the data column — reconcile
  before going on (replicate first). Ask whether the paper's identification argument rests
  on the **shares** (Goldsmith-Pinkham, Sorkin and Swift) or on the **shocks** (Borusyak,
  Hull and Jaravel); the decision tree treats them differently.

### 4. Interpret and decide
Open `references/decision_tree.md` and walk through it **with the actual numbers** for every
model. For each test, read `references/interpretation.md` before drawing conclusions —
several tests are invalid or uninformative in specific settings (classic Hausman under
clustering, Pesaran CD after time effects, clustered Wald with more restrictions than
clusters, panel unit-root tests under cross-sectional dependence).

Write down, before step 5:
- the recommended estimator (pooled / RE / FE entity / FE two-way / FD) and why;
- the recommended standard errors and why;
- what the baseline in the paper gets right and what it should change.

For IV, write down in addition: whether the instruments are strong (effective F vs 23.1;
F vs 104.7 for the usual t-test with one instrument), which weak-IV-robust set you rely on
(Anderson-Rubin; **AKM0** for shift-share — the region-level AR set is not robust to
shock-level correlation), and the recommended standard errors (AKM for shift-share).
**Never claim the exclusion restriction is "tested"**: overidentification tests only
compare instruments, and assume at least one is valid.

### 5. Run the robustness battery (panel)
`robustness.py config.json`. Each variant changes exactly one thing relative to the
baseline (see the script docstring). Read the verdict per key coefficient:
- `ROBUST` — same sign, still significant, and within ±50 % of the baseline in every
  robustness variant; no jackknife sign change.
- `MOSTLY ROBUST` — same sign everywhere, but some variant loses significance or moves more
  than 50 %. Name the variants and explain them.
- `FRAGILE` — a sign flip in a robustness variant or in the jackknife.
- `HETEROGENEITY` lines list sub-samples where the coefficient differs by more than 30 %:
  discuss them as substantive findings or as a reason to model the change explicitly.

Rows marked `[info]` (pooled OLS, between) answer a different question and do not count
for the verdict; use them to discuss whether the relation is mainly within or between
entities.

IV verdict (from `iv_diagnostics.py`): same rules, plus the weak-IV-robust set (AR, or
AKM0 for shift-share) enters it — a set that contains zero while the baseline is
significant counts as losing significance; an unbounded set makes the coefficient
`FRAGILE` (the data do not identify it). `WEAK INSTRUMENTS` in the verdict line means: report
the robust set as the main inference. Leave-one-instrument-out rows are heterogeneity
(different instruments can identify different LATEs). OLS is `[info]`.

### 6. Tables
`to_latex.py config.json --lang <en|pt>` — match the language of the paper. The tables need
`booktabs` and `graphicx`. Point the user to `\input{...}` paths.

### 7. Report
Write `report.md` in the output folder following `templates/report_template.md`, in the
user's language. It must contain: verdict up front, a diagnostics table read in plain
words, the decision on inference/estimator with justification, the robustness summary,
the heterogeneity discussion, concrete recommendations for the paper (what to change, what
to add to the appendix), a ready-to-paste methodology paragraph, and the limitations of
the checks (see the end of `references/interpretation.md`).

Finish by telling the user, in a few lines: the overall verdict per key coefficient, the
one or two changes you recommend, and the paths of the report and tables.

## When the request is narrower
- "Just run the tests": steps 1–4, plus the diagnostics table.
- "Is my result robust?": steps 1, 2, 5, a short report.
- Time series or cross section: `ols_diagnostics.py` covers steps 3 and 5 together.
- IV / 2SLS and shift-share: `iv_diagnostics.py` covers steps 3 and 5 together. Panel IV
  with absorbed fixed effects is not covered (only FE as `C()` dummies, fine for few
  groups); dynamic panel GMM, shock-level (BHJ) regressions and recentering are not covered.
- Models the scripts do not cover (panel IV with many FE, dynamic GMM, nonlinear, VAR, survival): say so, run
  what applies (`inspect_data.py`, residual checks via a formula approximation if
  meaningful), and list the specific tests the user should run instead
  (`references/interpretation.md`, "Not covered").
