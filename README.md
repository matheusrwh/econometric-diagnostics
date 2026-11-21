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

Each coefficient gets a verdict (`ROBUST`, `MOSTLY ROBUST`, `FRAGILE`), and sub-sample
differences are reported separately as **heterogeneity**, not as fragility.

## Install

```bash
git clone https://github.com/<user>/econometric-diagnostics ~/.claude/skills/econometric-diagnostics
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
```

The config format is documented in [`references/config_schema.md`](references/config_schema.md).

## Layout

```
SKILL.md                  workflow and ground rules Claude follows
scripts/                  inspect_data, panel_diagnostics, robustness, ols_diagnostics, to_latex
references/               decision tree, interpretation of each test, config schema
templates/                report template
examples/                 synthetic panel with a known data-generating process
```

## Limitations

Not covered: IV/2SLS and GMM diagnostics, second-generation panel unit roots (CIPS),
panel cointegration, wild cluster bootstrap, nonlinear models. The skill says so and
recommends the right tests when a model needs them.
