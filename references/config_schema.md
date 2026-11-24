# Config schema (`config.json`)

One config describes one **model set**: the same data and sample, one or more formulas.
Relative paths are resolved from the config file's folder.

```json
{
  "title": "Kaldor's laws - baseline",
  "data": "../../data/interim/growth.csv",
  "entity": "country_code",
  "time": "year",
  "exclude_entities": ["RWA"],
  "query": "year >= 1960",
  "output_dir": ".",
  "models": {
    "law1_manuf": {
      "formula": "cresc_va_non_manuf ~ 1 + cresc_va_manuf",
      "effects": "twoway",
      "cov": "clustered_entity",
      "key": ["cresc_va_manuf"],
      "label": "1st law, manufacturing"
    }
  }
}
```

| Key | Required | Meaning |
|---|---|---|
| `data` | yes | `.csv`, `.dta`, `.parquet`, `.xlsx` or `.tsv` |
| `entity`, `time` | no | both → panel; only `time` → time series; neither → cross section |
| `exclude_entities` | no | entity codes dropped before anything else (mirror the paper) |
| `query` | no | pandas `DataFrame.query` string applied after exclusions |
| `output_dir` | no | where JSON, tables and the report go (default: config folder) |
| `models.<name>.formula` | yes | patsy formula; transformations such as `np.log(x)` and `C(group)` work |
| `effects` | no | panel only: `twoway` (default), `entity`, `random`, `none` |
| `cov` | no | panel: `clustered_entity` (default), `clustered_time`, `clustered_twoway`, `robust`, `unadjusted`, `driscoll_kraay`. OLS: `HAC` (time-series default), `HC3` (cross-section default), `HC1`, `nonrobust`, `cluster:<column>` |
| `key` | no | coefficient(s) the paper is about; default: all regressors |
| `label` | no | column label in the LaTeX tables |
| `cluster` | no | OLS only: column for an extra clustered-SE variant |

Notes
- `inf` values are converted to missing automatically (growth rates of zero levels).
- Panel fixed effects are **not** written in the formula; use `effects`.
- Keep `1 +` in panel formulas when the user's model has a constant (linearmodels
  reports the constant of the within model the same way).
- To mirror a sample built in code (e.g. a balanced panel selected in the script), either
  save that sample to a file and point `data` to it, or express it with `query` and
  `exclude_entities`.

## IV / 2SLS models (`iv_diagnostics.py`)

```json
{
  "data": "regions.csv",
  "cluster": "state",
  "weights": "pop",
  "debiased": true,
  "models": {
    "main": {
      "formula": "y ~ 1 + w1 + w2 + [x ~ z1 + z2]",
      "estimator": "2sls",
      "cov": "cluster:state",
      "key": ["x"],
      "label": "2SLS"
    }
  }
}
```

| Key | Meaning |
|---|---|
| `formula` | exogenous regressors outside the brackets, `[endogenous ~ excluded instruments]` inside |
| `estimator` | `2sls` (default), `liml`, `fuller`, `gmm` (two-step, efficient weights) |
| `cov` | `robust` (default), `unadjusted`, `cluster:<column>`; default is `cluster:<cluster>` when the top-level `cluster` is set |
| `cluster` (top level) | column used for the clustered variant and the cluster jackknife |
| `weights` (top level) | regression weights (e.g. population), used everywhere |
| `debiased` (top level) | small-sample scaling of covariances (n/(n−k); clusters G(n−1)/((G−1)(n−k))), as in Stata's `small` / `reg`; default `true`. Match the paper's software |
| `key` | default: the endogenous regressors |

### Shift-share block

```json
"shift_share": {
  "instrument": "z_bartik",
  "exposure": "exposure_long.csv",
  "shocks": "shocks.csv",
  "obs_id": ["czone", "period"],
  "shock_id": ["industry", "period"],
  "share": "share",
  "shock": "g",
  "shock_cluster": "industry_3digit"
}
```

| Key | Meaning |
|---|---|
| `instrument` | name of the instrument in the formula. If the column is missing from the data the script builds it as Σₖ sᵢₖ gₖ; if it exists, it is rebuilt and must match |
| `exposure` | long file: one row per (observation, shock) with the share |
| `shocks` | one row per shock with the shock value and, optionally, a shock cluster |
| `obs_id`, `shock_id` | column(s) identifying observations and shocks (string or list). Stacked panels: include the period in both (shares nonzero only in their own period) |
| `shock_cluster` | optional; clusters of shocks for AKM (e.g. 3-digit industries when shocks are 4-digit) and the jackknife unit |

The block needs one endogenous regressor and the shift-share variable as the only excluded
instrument. A model can also use `"shift_share": true` to take a top-level `shift_share`.
