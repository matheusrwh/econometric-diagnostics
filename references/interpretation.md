# Interpreting each test

For each test: what H0 is, when the test is valid, and the usual traps.

## Panel

**F test for entity effects** — H0: all entity intercepts equal (pooled OLS ok). Non-robust
F. With large N it rejects almost always; the question is rarely whether, but which
estimator handles them.

**F test for time effects** — H0: all period effects zero, conditional on entity effects.
Non-robust. With many periods, even small common shocks reject.

**Wald test for time effects (clustered)** — robust version; needs fewer restrictions than
clusters (rank of the cluster covariance is at most N − 1). With T ≥ N it is not computed.

**Breusch-Pagan LM** — H0: variance of the random entity effect is zero. Baltagi-Li version
for unbalanced panels. Rejection says entity heterogeneity exists, not whether FE or RE.

**Hausman (classic)** — H0: RE consistent. Uses non-robust covariances, so it is invalid
under heteroskedasticity or serial correlation, which are almost always present. A
negative statistic or a singular difference matrix makes it uninformative. Report it,
decide with Mundlak.

**Mundlak / correlated random effects** — adds entity means of the regressors to an RE
model; H0: their coefficients are jointly zero. Cluster-robust Wald, so it is the valid
FE-vs-RE test in realistic settings. With two-way models, time dummies are included.

**Wooldridge AR(1)** — regresses first-differenced residuals on their lag; under no serial
correlation in the level errors that coefficient is −0.5. F(1, N−1), clustered.
Needs T ≥ 3 and consecutive periods (gaps are skipped).

**Modified Wald (groupwise heteroskedasticity)** — H0: σ²ᵢ = σ² for all entities, on FE
residuals. Very powerful with large T: rejection is the norm in country panels. Report
`max_min_variance_ratio` to convey magnitude.

**Pesaran CD** — H0: no cross-sectional correlation. Computed on **entity-FE residuals**,
because after time effects the average pairwise correlation is pushed towards zero by
construction (the test then has no power against common shocks). The script also reports
`mean_abs_correlation_after_time_effects`: what remains after the time effects. Compare it
with `mean_abs_correlation_expected_under_h0` — even independent series show
E|ρ| ≈ √(2/(πT)) (≈ 0.12 with T = 45), so a raw mean |ρ| is never zero; only the excess
over that benchmark signals remaining dependence. Designed for large N; with N < 10
interpret with care.

**Fisher-ADF panel unit root (Maddala-Wu P, Choi Z)** — H0: every series has a unit root;
H1: at least some are stationary. ADF per entity with AIC lag selection; series with
fewer than 10 observations are skipped. First-generation test: assumes cross-sectional
independence and over-rejects with common shocks, so the cross-section demeaned version is
also reported. Rejection does **not** mean all series are stationary
(`share_individual_p_below_05` helps).

**VIF (within)** — computed on entity-demeaned regressors, i.e. the variation FE uses.
Above 10: severe collinearity.

**Outliers** — standardized FE residuals. Count above 3 and 5 sd and the top 10
observations. With heavy-tailed macro data, 1–2 % beyond 3 sd is common.

## Robustness rows

- **SE rows** change only inference: the coefficient is identical by construction.
- **Driscoll-Kraay** (Bartlett kernel, automatic bandwidth) is robust to cross-sectional and
  serial correlation; consistent as T → ∞, poor with T < ~20.
- **Two-way clustering** needs many entities **and** many periods.
- **First differences** with time dummies: same estimand as FE under strict exogeneity;
  a large FE–FD gap signals measurement error (FD amplifies it, attenuating the
  coefficient), dynamic misspecification or violations of strict exogeneity.
- **Between** uses entity means only: long-run cross-sectional relation. A big gap with FE
  means the within and between relations differ — interesting, not a failure.
- **Dynamic FE**: lagged y with FE has Nickell bias of order 1/T.
- **Winsorized 1/99** caps every model variable at its pooled 1st and 99th percentiles.
- **Trimmed** drops observations with |standardized residual| > 3 from the baseline fit.
- **Balanced sub-panel**: the largest N×T window in which all retained entities are fully
  observed (end year searched among the last 5 periods).
- **Early/late**: split at the median observation (equal sizes), not the median year.
- **Jackknife**: refits leaving one entity out; reports range, most influential entities,
  share with the same sign and share significant at 5 %.

## Time series and cross section (ols_diagnostics.py)

- **ADF** (H0 unit root) and **KPSS** (H0 stationarity) are combined: agreement gives I(0)
  or I(1); disagreement is reported as `conflicting` — typically a structural break or
  near-unit-root persistence.
- **Engle-Granger** runs only when y and at least one regressor are classified I(1).
- **Breusch-Godfrey / Ljung-Box**: lag order n^(1/4) rounded. Durbin-Watson is reported for
  reference only (invalid with lagged y).
- **HAC** lags: floor(4·(n/100)^(2/9)) (Newey-West rule of thumb).
- **CUSUM** on OLS residuals (Ploberger-Krämer): H0 parameter stability.
- **Cook's distance** threshold 4/n; flag only when more than 10 % exceed it or any D > 1.
- **Huber RLM** and **LAD** are outlier-robust alternatives (different loss functions).

## Not covered (say so and recommend)

- Second-generation panel unit roots (CIPS), panel cointegration (Pedroni, Westerlund).
- Wild cluster bootstrap for few clusters.
- IV/2SLS diagnostics: first-stage F / Kleibergen-Paap, over-identification (Hansen J),
  endogeneity (Durbin-Wu-Hausman).
- Dynamic panel GMM: Arellano-Bond AR(2), Hansen J, instrument count.
- Nonlinear models (logit/probit, Poisson PPML), survival models, VAR/VECM.
- Multiple structural breaks (Bai-Perron), heterogeneous slopes (mean group, CCE).
