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
