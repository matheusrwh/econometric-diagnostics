# econometric-diagnostics

A Claude Code skill that reviews an estimated econometric
model the way a careful referee would: it replicates the baseline, runs the specification
and diagnostic tests that fit the data structure, decides the appropriate estimator and
standard errors, stress-tests the key coefficients with a one-change-at-a-time robustness
battery, and writes LaTeX tables plus an interpreted report.

The Python scripts do the computing; the skill instructions (`SKILL.md`) and the reference
files tell Claude how to read the results — including when a test is invalid in a given
setting (classic Hausman under clustering, Pesaran CD after time effects, first-generation
unit-root tests under cross-sectional dependence).

## What it covers

| Data | Diagnostics | Robustness |
|---|---|---|
| **Panel** | F tests for entity and time effects, Breusch-Pagan LM, Hausman, Mundlak (cluster-robust), Wooldridge AR(1), modified Wald, Pesaran CD, Fisher-ADF unit roots, within VIF, outliers | SEs: iid, robust, clustered (entity, time, two-way), Driscoll-Kraay · Estimators: pooled, RE, FE, two-way FE, first differences, between, dynamic FE · Sample: winsorizing, trimming, balanced sub-panel, early/late split, leave-one-entity-out jackknife |
| **Time series** | ADF + KPSS integration order, Engle-Granger, Breusch-Godfrey, Ljung-Box, Breusch-Pagan, ARCH-LM, Jarque-Bera, RESET, CUSUM | SEs: iid, HC, HAC · Huber, LAD, first differences, lagged y · Cook's D, winsorizing, sample halves |
| **Cross section** | Breusch-Pagan, White, RESET, Jarque-Bera, VIF, Cook's distance | SEs: iid, HC1, HC3, clustered · Huber, LAD · Cook's D, winsorizing |
| **IV / 2SLS** | First stage: effective F (Montiel Olea-Pflueger), robust/Kleibergen-Paap F, Cragg-Donald + Stock-Yogo, Sanderson-Windmeijer · reduced form · Anderson-Rubin confidence set · control-function endogeneity · Hansen J, Sargan, C | SEs: iid, robust, clustered · OLS, 2SLS, LIML, Fuller, GMM · leave one instrument out · winsorizing, trimming, cluster jackknife |
| **Shift-share IV** | AKM and AKM0 (Adão-Kolesár-Morales), AKM first-stage F, Rotemberg weights (Goldsmith-Pinkham-Sorkin-Swift), effective number of shocks, sum-of-shares check (Borusyak-Hull-Jaravel) | AKM SEs (also with shock clusters) · sum-of-shares control · leave-one-shock-out jackknife |

Each coefficient gets a verdict (`ROBUST`, `MOSTLY ROBUST`, `FRAGILE`), and sub-sample
differences are reported separately as **heterogeneity**, not as fragility.

## Install

```bash
git clone https://github.com/matheusrwh/econometric-diagnostics ~/.claude/skills/econometric-diagnostics
pip install -r ~/.claude/skills/econometric-diagnostics/requirements.txt
```

Then, in Claude Code, inside your research project:

> Check the robustness of the models in `models/baseline.py`.

## Use the scripts directly

```bash
cd examples
python make_example_data.py
python ../scripts/inspect_data.py config_example.json
python ../scripts/panel_diagnostics.py config_example.json
python ../scripts/robustness.py config_example.json
python ../scripts/to_latex.py config_example.json --lang en

python make_iv_example.py
python ../scripts/iv_diagnostics.py config_iv.json
python ../scripts/iv_diagnostics.py config_shiftshare.json
python ../scripts/to_latex.py config_shiftshare.json --lang en
```

## Validation

```bash
python tests/validate_iv.py            # ~50 checks, a few seconds
python tests/validate_iv.py --mc 1000  # plus a Monte Carlo of shift-share inference
```

The IV code reproduces published textbook results (Mroz and Card examples in Wooldridge,
*Introductory Econometrics*, ch. 15), matches linearmodels for every covariance type with
and without weights, and satisfies exact identities (effective F = robust F with one
instrument; AKM = HC0 with one sector per region; Rotemberg weights reproduce 2SLS; AR and
AKM0 set endpoints). In the Monte Carlo (shares fixed, shocks redrawn, 1,000 replications)
the rejection rate of the true value at 5 % is 0.25 for robust and state-clustered SEs,
0.11 for AKM and 0.045 for AKM0. Not yet compared with the reference implementations in R
(`ShiftShareSE`) and Stata (`ivreg2`, `bartik_weight`) on the Autor-Dorn-Hanson data.

The config format is documented in [`references/config_schema.md`](references/config_schema.md).

## Layout

```
SKILL.md                  workflow and ground rules Claude follows
scripts/                  inspect_data, panel_diagnostics, robustness, ols_diagnostics,
                          iv_diagnostics (+ iv_core), to_latex
tests/                    validate_iv.py
references/               decision tree, interpretation of each test, config schema
templates/                report template
examples/                 synthetic panel, IV and shift-share data with known DGPs
```

## Limitations

Not covered: panel IV with absorbed fixed effects, dynamic panel GMM, shock-level
shift-share regressions and recentering, CLR tests, second-generation panel unit roots
(CIPS), panel cointegration, wild cluster bootstrap, nonlinear models. The skill says so and
recommends the right tests when a model needs them.
