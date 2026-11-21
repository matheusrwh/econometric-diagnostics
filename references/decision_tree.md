# Decision tree

Walk through the sections that apply, in order. Use the numbers, not only the flags.
Every decision goes into the report with its justification.

## Panel data

### 1. Is there unobserved heterogeneity? (pooled vs effects)
- `F test for entity effects` and/or `Breusch-Pagan LM` reject → entity effects exist;
  pooled OLS is inappropriate as the main estimator (`ENTITY_EFFECTS_PRESENT`).
- Neither rejects → pooled OLS is acceptable; keep FE as a robustness check.

### 2. Fixed or random effects?
- Use **Mundlak (clustered)** as the decision test. The classic Hausman assumes iid errors
  and is invalid when step 4 below finds heteroskedasticity or serial correlation; report
  it, but do not decide on it.
- Mundlak rejects (`RE_INCONSISTENT_USE_FE`) → FE. RE would be biased.
- Mundlak does not reject (`RE_NOT_REJECTED`) → RE is consistent and more efficient, but FE
  remains valid. In macro panels with countries as units, FE is the standard and safer
  choice; say that RE is not rejected and that FE and RE estimates are close (compare
  them in `estimates_clustered_se`).
- If the key regressor has low within variation (`LOW_WITHIN_VARIATION`), FE identifies the
  coefficient from little variation: report the within share and consider RE/CRE or
  between estimates as complements.

### 3. Time effects?
- `F test for time effects` rejects (`TIME_EFFECTS_PRESENT`) → include time effects
  (two-way FE). Omitting them confounds the key coefficient with common shocks
  (global cycles, oil shocks, crises).
- The clustered Wald version is only computed when there are fewer restrictions than
  clusters; when it is "not computed", rely on the F test.
- Not rejected → entity FE is enough; two-way FE as robustness.

### 4. Which standard errors?
Start from clustered by entity (robust to heteroskedasticity and to any within-entity
serial correlation). Then:

| Finding | Implication | Recommendation |
|---|---|---|
| `GROUPWISE_HETEROSKEDASTICITY` | iid SEs are wrong | already handled by entity clustering |
| `SERIAL_CORRELATION` | iid and HC SEs understate uncertainty | entity clustering handles it; FD as an estimator check |
| `CROSS_SECTIONAL_DEPENDENCE` and time effects included | common shocks with **equal** loadings are absorbed; compare `mean_abs_correlation_after_time_effects` with `mean_abs_correlation_expected_under_h0` (≈ √(2/(πT)), pure sampling noise) | if the excess is small (< ~0.03) entity clustering is fine; if it is clearly positive, dependence with heterogeneous loadings remains: report **Driscoll-Kraay** (needs T ≥ ~20) and two-way clustering next to entity clustering |
| `CROSS_SECTIONAL_DEPENDENCE` without time effects | errors correlated across entities | add time effects first; then Driscoll-Kraay or two-way clustering |
| `FEW_CLUSTERS` (N < 40) | cluster-robust SEs over-reject | wild cluster bootstrap (e.g. `wildboottest`), or Driscoll-Kraay if T is large |

Compare the SE rows in the robustness table: if the largest SE among the defensible
options (entity clustering, two-way clustering, Driscoll-Kraay) still leaves the key
coefficient significant, say so — it is the strongest statement available.

### 5. Nonstationarity
- Panel unit-root tests reject for all variables (growth rates, ratios) → no spurious
  regression concern.
- `UNIT_ROOT_SUSPECT:<var>` (not rejected) for levels → the regression in levels may be
  spurious. Recommend estimating in differences/growth rates, or testing for panel
  cointegration (Pedroni, Westerlund) before interpreting levels coefficients.
- With `CROSS_SECTIONAL_DEPENDENCE`, first-generation tests (Fisher-ADF) over-reject;
  trust the cross-section-demeaned version and mention CIPS (Pesaran 2007) as the proper
  test.

### 6. Outliers and influence
- `OUTLIERS`: compare `Winsorized` and `Trimmed` rows with the baseline. If the coefficient
  moves little, outliers do not drive the result. If it moves a lot, identify the
  observations (`outliers.top` in the diagnostics JSON) and discuss whether they are data
  errors or genuine episodes (wars, crises, hyperinflations).
- Jackknife: name the most influential entity and the coefficient range.

### 7. Dynamics
- If `Dynamic FE (+ lagged y)` changes the key coefficient a lot, the static model may
  omit persistence. With large T the Nickell bias (order 1/T) is small; with small T
  (< ~15) recommend a GMM estimator (Arellano-Bond / Blundell-Bond) rather than FE with a
  lag.

### 8. Robustness vs heterogeneity
- Robustness variants (inference, estimator, outliers, jackknife) test whether the result
  is an artefact of choices.
- Heterogeneity variants (early/late period, balanced sub-panel) test whether the
  parameter is stable. Large differences here are a finding: recommend modelling them
  (period interactions, rolling windows, Chow-type test via interaction terms) instead of
  calling the result fragile.

## Time series

1. **Integration order** (`ADF/KPSS integration order`):
   - all I(0) → OLS in levels is fine;
   - y and regressors I(1) and **Engle-Granger rejects** → cointegrated: the levels
     regression is a long-run relation (super-consistent); recommend an ECM for dynamics;
   - I(1) and **no cointegration** (`SPURIOUS_REGRESSION_RISK`) → do not interpret the
     levels regression; use first differences (see the `First differences` row);
   - `conflicting (ADF: I(0), KPSS: not stationary)` in growth rates usually signals a
     **shift in the mean** (structural break), not a unit root: check `CUSUM` and the
     early/late split, and consider a break dummy or Zivot-Andrews / Bai-Perron tests.
2. **Serial correlation** (Breusch-Godfrey, Ljung-Box) → HAC (Newey-West) SEs; if strong,
   add lags (ARDL) — see `With lagged dependent variable`.
3. **Heteroskedasticity / ARCH** → HAC or HC SEs; ARCH in finance-type data may call for a
   GARCH model of the variance.
4. **RESET** rejects → functional form: logs, squares, interactions or omitted variables.
5. **CUSUM** rejects or early/late halves differ → structural instability: split sample,
   break dummies, or rolling estimates.
6. **Small samples** (n < 50): unit-root tests have low power; say so.

## Cross section

1. Breusch-Pagan / White reject → use HC3 (default) — already the baseline.
2. RESET rejects (`FUNCTIONAL_FORM`) → try logs, quadratic terms, interactions; check
   omitted variables. This is a model problem, not an inference problem.
3. `HIGH_VIF` → collinearity inflates SEs of the collinear regressors; do not drop a
   theoretically required control only to gain significance — report joint tests instead.
4. `INFLUENTIAL_OBS` → compare `Drop Cook's D > 4/n`, `Huber` and `LAD` rows; name the
   influential units.
5. Clustered sampling (firms in sectors, people in regions) → set `cluster` and report
   clustered SEs.
