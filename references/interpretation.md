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

## IV / 2SLS (iv_diagnostics.py)

Notation: K1 endogenous regressors, L excluded instruments. Formulas are in
`scripts/iv_core.py`; `tests/validate_iv.py` checks them.

**Effective F (Montiel Olea and Pflueger 2013)** — π̂′(Z̃′Z̃)π̂ / tr(V̂π Z̃′Z̃), with V̂π the
robust/clustered covariance of the first-stage coefficients. Valid under
heteroskedasticity, clustering and autocorrelation; equals the non-robust F under iid
errors and the robust Wald F when L = 1. Simplified critical value 23.1 (worst-case Nagar
bias of 10 % of the OLS bias, 5 % test). K1 = 1 only.

**Robust first-stage F (Kleibergen-Paap Wald)** — robust Wald test of the excluded
instruments divided by L. With K1 = 1 it is the Kleibergen-Paap rk Wald F up to Stata's
small-sample scaling. Note: linearmodels prints the Wald χ² (= L × F) for robust covariances.

**Cragg-Donald F + Stock-Yogo** — the non-robust F with Stock-Yogo critical values (10 %
maximal size: 16.38 for L = 1, 19.93 for L = 2, 22.30 for L = 3). Valid only with iid
errors; report as a reference, as with the classic Hausman.

**Sanderson-Windmeijer F** (K1 > 1) — conditional first-stage F for each endogenous
regressor, net of the others; robust Wald divided by L − K1 + 1. A low value for one
regressor means the instruments do not separate it from the other endogenous ones.

**Reduced form** — y on the instruments and controls. With one endogenous regressor its
joint test *is* the Anderson-Rubin test of β = 0.

**Anderson-Rubin set** — all β₀ for which y − β₀x is unrelated to the instruments (Wald
with the chosen covariance). Exact size whatever the instrument strength; with L = 1 it is
the efficient weak-IV-robust test. With L = 1 the set is computed exactly (it can be an
interval, the union of two rays, or the whole line); with L > 1 by grid inversion, and
`(grid edge)` means it may be unbounded. Less powerful than CLR with many instruments (CLR
is not implemented: R `ivmodel`, Stata `weakiv`).

**Control-function test (robust Durbin-Wu-Hausman)** — adds first-stage residuals to the
structural equation; H0: coefficients zero (regressors exogenous). Equals the Wu-Hausman F
under iid errors. Low power with weak instruments.

**Hansen J / Sargan / C** — H0: all instruments valid, given at least K1 of them are.
Hansen J uses two-step GMM with robust/clustered weights; Sargan only under iid. The C
statistic for an instrument is J(all) − J(without it), with separately estimated weight
matrices, so it can be negative (then uninformative). With heterogeneous effects (LATE),
valid instruments can still reject.

**Robustness rows** — OLS `[info]`; LIML and Fuller(1) (less biased than 2SLS with weak
or many instruments; LIML = 2SLS when just identified); two-step GMM; leave one
instrument out and strongest instrument only (heterogeneity rows); winsorized y; winsorized
y, x and instruments together (not for shift-share); trimmed residuals; jackknife by
cluster with the minimum effective F across drops.

Never winsorize x alone in IV: when x and the instruments share long tails (e.g. migration
inflows to frontier regions) capping x destroys the first stage and the row looks fragile
for a mechanical reason. When winsorizing y, x and Z together leaves a weak first stage
(F < 10), the row is excluded from the verdict and `IDENTIFICATION_FROM_TAILS` is raised:
the instrument identifies the effect mainly from a few extreme observations — name them
and report the estimate without them; it is a finding about identification, not fragility.

## Shift-share IV

**AKM standard error (Adão, Kolesár and Morales 2019)** — variance from shock-level
variation: Σ_c (Σ_{k∈c} X̂ₖ R̂ₖ)² / (Σ wᵢ z̃ᵢ xᵢ)², with R̂ₖ = Σᵢ wᵢ sᵢₖ ε̂ᵢ and X̂ the shocks
recovered by regressing the residualized instrument on the shares. Reduces to the HC0 SE
when each region has one sector (checked in the validation). Needs the shares and shocks.
With few effective shocks it over-rejects; prefer AKM0.

Two practical traps:
- **Raw shocks instead of X̂.** Plugging the raw shocks gₖ into the formula (instead of X̂)
  keeps their common mean, which controls such as the sum of shares already absorb; when
  the mean is large relative to the dispersion the SE is badly inflated (a factor of 3–4
  in an application with mean 0.15 and sd 0.04). Compare `shock_mean` and `shock_sd`.
- **Many shocks per observation** (`AKM_MANY_SHOCKS_PER_OBS`, K/N > 0.5): X̂ comes from a
  regression on K exposure columns with N rows and becomes noisy (low `corr_xhat_shock`,
  ill-conditioned shares). Report AKM and AKM0, but say that their reliability is limited;
  shock clusters (fewer, larger units) and the shock-level regression of BHJ help.

**AKM0** — the null-imposed version: residuals computed under H0, set obtained by inverting
t(β₀)² ≤ 1.96² (exact quadratic, may be unbounded). Correct size in the validation Monte
Carlo and robust to weak instruments. The main shift-share inference.

**AKM first-stage F** — the first-stage t² with AKM variance; region-level first-stage F
statistics overstate strength for the same reason SEs are too small.

**Rotemberg weights (Goldsmith-Pinkham, Sorkin and Swift 2020)** — αₖ = gₖ sₖ′w x̃ /
Σⱼ gⱼ sⱼ′w x̃, with β̂ = Σ αₖ β̂ₖ and β̂ₖ the IV that uses only sector k's shares. They
show which sectors' shares drive identification. Exact identity checked in the validation.

**Effective number of shocks** — 1 / HHI of average exposure (BHJ). Shock-level
identification needs it to be large.

**Sum of shares** — when it varies across observations and is not a control, the
instrument mixes shock variation with the total share (BHJ). Detected by checking whether
the sum lies in the span of the controls.

**Shock jackknife** — drops one shock (or shock cluster) from the instrument; closed form
when controls stay fixed, refit when the sum of shares is a control.

## Not covered (say so and recommend)

- Second-generation panel unit roots (CIPS), panel cointegration (Pedroni, Westerlund).
- Wild cluster bootstrap for few clusters.
- Panel IV with absorbed high-dimensional fixed effects (use `pyfixest` / R `fixest` /
  Stata `ivreghdfe`); CLR and Kleibergen-Paap rk with K1 > 1; tF critical values.
- Shift-share: shock-level regression and shock balance (Stata `ssaggregate`), recentering
  (Borusyak and Hull 2023), pre-trend and balance tests of top-Rotemberg shares (need
  variables chosen with the user), AKM with several instruments.
- Plausibly exogenous sensitivity (Conley, Hansen and Rossi 2012).
- Dynamic panel GMM: Arellano-Bond AR(2), Hansen J, instrument count.
- Nonlinear models (logit/probit, Poisson PPML), survival models, VAR/VECM.
- Multiple structural breaks (Bai-Perron), heterogeneous slopes (mean group, CCE).
