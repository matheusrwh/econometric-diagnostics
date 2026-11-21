"""Step 2 (panel data) — specification and diagnostic tests.

Usage:  python panel_diagnostics.py <config.json> [--skip-unit-root]
Writes: <output_dir>/panel_diagnostics.json and prints a summary table.

Tests (all on the model as specified in the config):
  F test for entity effects ............ pooled OLS vs FE
  Time effects (F and clustered Wald) .. FE(entity) vs FE(entity + time dummies)
  Breusch-Pagan LM (Baltagi-Li) ........ pooled OLS vs RE (unbalanced-panel version)
  Hausman (classic) .................... FE vs RE, non-robust covariances
  Mundlak / CRE (cluster-robust) ....... FE vs RE, valid with heteroskedasticity/clustering
  Wooldridge AR(1) ..................... serial correlation in idiosyncratic errors
  Modified Wald ........................ groupwise heteroskedasticity in FE residuals
  Pesaran CD ........................... cross-sectional dependence in FE residuals
  Fisher-ADF (Maddala-Wu, Choi) ........ panel unit roots, raw and cross-section demeaned
  VIF on within-transformed regressors . multicollinearity (k > 1)
  Largest standardized FE residuals .... outliers / influential observations
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels import PanelOLS, PooledOLS, RandomEffects
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from common import (design, drop_intercept, failed_test, fmt_p, load_config, load_data,
                    test_result, write_json)

warnings.filterwarnings("ignore")
ALPHA = 0.05


# ---------------------------------------------------------------- helpers


def time_dummies(index: pd.MultiIndex) -> pd.DataFrame:
    t = pd.Series(index.get_level_values(1), index=index)
    d = pd.get_dummies(t, prefix="_t", drop_first=True, dtype=float)
    return d


def fe_model(y, X, effects: str):
    return PanelOLS(y, X, entity_effects=True, time_effects=(effects == "twoway"),
                    drop_absorbed=True, check_rank=False)


def coef_summary(res, names) -> dict:
    return {n: {"coef": float(res.params[n]), "se": float(res.std_errors[n]),
                "pvalue": float(res.pvalues[n])} for n in names if n in res.params.index}


def consecutive_diff(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """First difference within entity, NaN when the previous period is missing."""
    t = pd.Series(s.index.get_level_values(1), index=s.index).astype(float)
    step = np.nanmin(np.diff(np.sort(t.unique())))
    gap = t.groupby(level=0).diff()
    d = s.groupby(level=0).diff()
    ok = (gap == step)
    return d[ok] if isinstance(d, pd.Series) else d[ok.values]


def lag1(s: pd.Series) -> pd.Series:
    t = pd.Series(s.index.get_level_values(1), index=s.index).astype(float)
    step = np.nanmin(np.diff(np.sort(t.unique())))
    gap = t.groupby(level=0).diff()
    lagged = s.groupby(level=0).shift(1)
    return lagged.where(gap == step)


# ---------------------------------------------------------------- tests


def test_entity_effects(y, X) -> dict:
    res = PanelOLS(y, X, entity_effects=True, drop_absorbed=True, check_rank=False).fit()
    f = res.f_pooled
    return test_result("F test for entity effects", "all entity effects are zero (pooled OLS ok)",
                       float(f.stat), float(f.pval), df=[int(f.df), int(f.df_denom)],
                       note="non-robust F; a rejection means pooled OLS is inappropriate")


def test_time_effects(y, X) -> list[dict]:
    out = []
    D = time_dummies(y.index)
    if D.shape[1] == 0:
        return out
    Xd = pd.concat([X, D], axis=1)
    r0 = PanelOLS(y, X, entity_effects=True, drop_absorbed=True, check_rank=False).fit()
    r1 = PanelOLS(y, Xd, entity_effects=True, drop_absorbed=True, check_rank=False).fit()
    q = D.shape[1]
    f = ((r0.resid_ss - r1.resid_ss) / q) / (r1.resid_ss / r1.df_resid)
    out.append(test_result("F test for time effects", "all time effects are zero",
                           float(f), float(stats.f.sf(f, q, r1.df_resid)),
                           df=[q, int(r1.df_resid)], note="non-robust F"))
    n_clusters = y.index.get_level_values(0).nunique()
    if q >= n_clusters - 1:
        # the cluster-robust covariance of q dummies has rank <= n_clusters - 1: Wald is undefined
        out.append(test_result("Wald test for time effects (clustered)",
                               "all time effects are zero", None, None, df=q,
                               note=f"not computed: {q} restrictions >= {n_clusters} clusters - 1 "
                                    "(singular cluster covariance); rely on the F test"))
        return out
    try:
        rc = PanelOLS(y, Xd, entity_effects=True, drop_absorbed=True,
                      check_rank=False).fit(cov_type="clustered", cluster_entity=True)
        names = [c for c in D.columns if c in rc.params.index]
        R = np.zeros((len(names), len(rc.params)))
        for i, n in enumerate(names):
            R[i, list(rc.params.index).index(n)] = 1
        w = rc.wald_test(R, np.zeros(len(names)))
        out.append(test_result("Wald test for time effects (clustered)",
                               "all time effects are zero", float(w.stat), float(w.pval),
                               df=int(w.df), note="cluster-robust Wald"))
    except Exception as e:  # noqa: BLE001
        out.append(failed_test("Wald test for time effects (clustered)",
                               "all time effects are zero", e))
    return out


def test_bp_lm(y, X) -> dict:
    """Breusch-Pagan LM for random effects, Baltagi-Li (1990) unbalanced version."""
    e = PooledOLS(y, X).fit().resids
    n = len(e)
    sum_i = e.groupby(level=0).sum()
    T_i = e.groupby(level=0).size()
    A = (sum_i ** 2).sum() / (e ** 2).sum() - 1
    denom = 2 * ((T_i ** 2).sum() - n)
    lm = n ** 2 / denom * A ** 2
    return test_result("Breusch-Pagan LM (random effects)",
                       "variance of entity effects is zero (pooled OLS ok)",
                       float(lm), float(stats.chi2.sf(lm, 1)), df=1,
                       note="Baltagi-Li version valid for unbalanced panels")


def test_hausman(y, X, keys, effects) -> dict:
    Xh = X if effects != "twoway" else pd.concat([X, time_dummies(y.index)], axis=1)
    fe = PanelOLS(y, Xh, entity_effects=True, drop_absorbed=True, check_rank=False).fit()
    re = RandomEffects(y, Xh, check_rank=False).fit()
    names = [k for k in keys if k in fe.params.index and k in re.params.index]
    d = (fe.params[names] - re.params[names]).to_numpy()
    V = (fe.cov.loc[names, names] - re.cov.loc[names, names]).to_numpy()
    stat = float(d @ np.linalg.pinv(V) @ d)
    rank = int(np.linalg.matrix_rank(V))
    note = "non-robust covariances: invalid under heteroskedasticity or serial correlation"
    if stat < 0 or rank < len(names):
        note += "; V_FE - V_RE not positive definite, rely on the Mundlak test"
    return test_result("Hausman (classic)", "RE is consistent (entity effects uncorrelated with X)",
                       stat, float(stats.chi2.sf(max(stat, 0), max(rank, 1))), df=rank, note=note,
                       theta_re=float(np.mean(np.asarray(re.theta))))


def test_mundlak(y, X, keys, effects) -> dict:
    Xr = drop_intercept(X)
    means = Xr.groupby(level=0).transform("mean").add_prefix("_mean_")
    parts = [X, means]
    if effects == "twoway":
        parts.append(time_dummies(y.index))
    Xm = pd.concat(parts, axis=1)
    res = RandomEffects(y, Xm, check_rank=False).fit(cov_type="clustered", cluster_entity=True)
    names = [f"_mean_{k}" for k in keys if f"_mean_{k}" in res.params.index]
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, list(res.params.index).index(n)] = 1
    w = res.wald_test(R, np.zeros(len(names)))
    return test_result("Mundlak / correlated random effects (clustered)",
                       "entity means of X are jointly zero (RE is consistent)",
                       float(w.stat), float(w.pval), df=int(w.df),
                       note="cluster-robust alternative to Hausman; prefer it when errors are "
                            "heteroskedastic or serially correlated")


def test_wooldridge_ar1(y, X) -> dict:
    Xr = drop_intercept(X)
    dy = consecutive_diff(y)
    dX = consecutive_diff(Xr).loc[dy.index]
    ok = dy.notna() & dX.notna().all(axis=1)
    dy, dX = dy[ok], dX[ok]
    groups = pd.factorize(dy.index.get_level_values(0))[0]
    r1 = sm.OLS(dy, sm.add_constant(dX)).fit(cov_type="cluster", cov_kwds={"groups": groups})
    u = pd.Series(r1.resid, index=dy.index)
    u_lag = lag1(u)
    ok = u_lag.notna()
    groups = pd.factorize(u[ok].index.get_level_values(0))[0]
    r2 = sm.OLS(u[ok], u_lag[ok]).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b, se = float(r2.params.iloc[0]), float(r2.bse.iloc[0])
    F = ((b + 0.5) / se) ** 2
    n_ent = len(np.unique(groups))
    return test_result("Wooldridge test for AR(1)", "no first-order serial correlation in the "
                       "idiosyncratic errors", F, float(stats.f.sf(F, 1, n_ent - 1)),
                       df=[1, n_ent - 1], rho_fd_resid=b,
                       note="under H0 the coefficient on lagged FD residuals is -0.5")


def test_groupwise_het(resid: pd.Series) -> dict:
    """Modified Wald test (Greene) for groupwise heteroskedasticity."""
    e2 = resid ** 2
    g = e2.groupby(level=0)
    T_i = g.size()
    s2_i = g.mean()
    keep = T_i > 1
    V_i = g.apply(lambda v: ((v - v.mean()) ** 2).sum()) / (T_i * (T_i - 1))
    s2 = e2.mean()
    W = float((((s2_i - s2) ** 2) / V_i)[keep].sum())
    N = int(keep.sum())
    ratio = float(s2_i.max() / s2_i.min()) if s2_i.min() > 0 else float("inf")
    return test_result("Modified Wald (groupwise heteroskedasticity)",
                       "error variance is equal across entities", W, float(stats.chi2.sf(W, N)),
                       df=N, max_min_variance_ratio=ratio,
                       note="computed on FE residuals; very powerful with large T")


def pairwise_corr(resid: pd.Series) -> tuple[float, float, int, int, float]:
    """Returns (sum sqrt(T_ij)*rho_ij, sum |rho_ij|, pairs, N, sum E|rho_ij| under H0).

    Under independence E|rho_ij| ~ sqrt(2 / (pi * T_ij)): the benchmark for mean |rho|."""
    A = resid.unstack(level=1).to_numpy(dtype=float)
    N = A.shape[0]
    total, abs_rho, pairs, null_abs = 0.0, 0.0, 0, 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            m = ~np.isnan(A[i]) & ~np.isnan(A[j])
            T_ij = int(m.sum())
            if T_ij < 3:
                continue
            a, b = A[i, m], A[j, m]
            a, b = a - a.mean(), b - b.mean()
            den = math.sqrt((a @ a) * (b @ b))
            if den == 0:
                continue
            rho = float(a @ b / den)
            total += math.sqrt(T_ij) * rho
            abs_rho += abs(rho)
            null_abs += math.sqrt(2 / (math.pi * T_ij))
            pairs += 1
    return total, abs_rho, pairs, N, null_abs


def test_pesaran_cd(resid_entity_fe: pd.Series, resid_model: pd.Series, effects: str) -> dict:
    """CD on entity-FE residuals (time effects would mechanically push CD towards zero).

    Mean |rho| of the residuals of the model as specified is reported as well: if it stays
    high after time effects, dependence is heterogeneous (factor loadings differ by entity)."""
    total, abs_rho, pairs, N, null_abs = pairwise_corr(resid_entity_fe)
    cd = math.sqrt(2 / (N * (N - 1))) * total
    extra = {"mean_abs_correlation": abs_rho / max(pairs, 1), "pairs": pairs,
             "mean_abs_correlation_expected_under_h0": null_abs / max(pairs, 1)}
    note = "computed on entity-FE residuals (without time effects)"
    if effects == "twoway":
        _, abs_rho_m, pairs_m, _, _ = pairwise_corr(resid_model)
        extra["mean_abs_correlation_after_time_effects"] = abs_rho_m / max(pairs_m, 1)
        extra["excess_abs_correlation_after_time_effects"] = (
            extra["mean_abs_correlation_after_time_effects"]
            - extra["mean_abs_correlation_expected_under_h0"])
        note += ("; time effects absorb common shocks with equal loadings - the mean |rho| after "
                 "time effects measures what is left (heterogeneous loadings)")
    return test_result("Pesaran CD", "no cross-sectional dependence in the errors", cd,
                       float(2 * stats.norm.sf(abs(cd))), note=note, **extra)


def fisher_adf(s: pd.Series, demean: bool) -> dict:
    if demean:
        s = s - s.groupby(level=1).transform("mean")
    pvals = []
    for _, v in s.dropna().groupby(level=0):
        x = v.droplevel(0).sort_index().to_numpy(dtype=float)
        if len(x) < 10 or np.std(x) == 0:
            continue
        maxlag = int(12 * (len(x) / 100) ** 0.25)
        try:
            pvals.append(adfuller(x, maxlag=maxlag, regression="c", autolag="AIC")[1])
        except Exception:  # noqa: BLE001
            continue
    p = np.clip(np.array(pvals), 1e-10, 1 - 1e-10)
    N = len(p)
    if N == 0:
        return {"n_series": 0}
    P = float(-2 * np.log(p).sum())
    Z = float(stats.norm.ppf(p).sum() / math.sqrt(N))
    return {"n_series": N,
            "maddala_wu_P": P, "maddala_wu_pvalue": float(stats.chi2.sf(P, 2 * N)),
            "choi_Z": Z, "choi_pvalue": float(stats.norm.cdf(Z)),
            "share_individual_p_below_05": float((p < 0.05).mean())}


def test_unit_roots(df_used: pd.DataFrame, variables: list[str]) -> list[dict]:
    out = []
    for v in variables:
        raw = fisher_adf(df_used[v], demean=False)
        dem = fisher_adf(df_used[v], demean=True)
        if raw.get("n_series", 0) == 0:
            continue
        p_main = dem.get("choi_pvalue", raw["choi_pvalue"])
        rec = test_result(f"Fisher-ADF panel unit root: {v}",
                          "all series have a unit root", raw["choi_Z"], raw["choi_pvalue"],
                          note="Choi Z on raw series; 'demeaned' removes the cross-section mean "
                               "each period (partial control for cross-sectional dependence)",
                          raw=raw, cross_section_demeaned=dem)
        rec["reject_h0"] = bool(raw["choi_pvalue"] < ALPHA and p_main < ALPHA)
        out.append(rec)
    return out


def vif_within(X: pd.DataFrame) -> dict:
    Xr = drop_intercept(X)
    if Xr.shape[1] < 2:
        return {}
    W = Xr - Xr.groupby(level=0).transform("mean")
    out = {}
    for c in W.columns:
        others = W.drop(columns=c)
        r2 = sm.OLS(W[c], others).fit().rsquared
        out[c] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return out


def outliers(resid: pd.Series, k: int = 10) -> dict:
    z = (resid - resid.mean()) / resid.std(ddof=1)
    top = z.abs().sort_values(ascending=False).head(k)
    return {"n_abs_z_above_3": int((z.abs() > 3).sum()),
            "n_abs_z_above_5": int((z.abs() > 5).sum()),
            "share_abs_z_above_3": float((z.abs() > 3).mean()),
            "top": [{"entity": str(i[0]), "time": str(i[1]), "z": float(z[i])} for i in top.index]}


# ---------------------------------------------------------------- flags


def make_flags(tests: list[dict], n_clusters: int, inspection_low_within: list[str],
               outl: dict, vif: dict) -> list[str]:
    by = {t["test"]: t for t in tests}
    f = []

    def rej(name):
        return by.get(name, {}).get("reject_h0")

    if rej("F test for entity effects") or rej("Breusch-Pagan LM (random effects)"):
        f.append("ENTITY_EFFECTS_PRESENT")
    if rej("F test for time effects"):
        f.append("TIME_EFFECTS_PRESENT")
    if rej("Mundlak / correlated random effects (clustered)"):
        f.append("RE_INCONSISTENT_USE_FE")
    elif by.get("Mundlak / correlated random effects (clustered)", {}).get("reject_h0") is False:
        f.append("RE_NOT_REJECTED")
    if rej("Wooldridge test for AR(1)"):
        f.append("SERIAL_CORRELATION")
    if rej("Modified Wald (groupwise heteroskedasticity)"):
        f.append("GROUPWISE_HETEROSKEDASTICITY")
    if rej("Pesaran CD"):
        f.append("CROSS_SECTIONAL_DEPENDENCE")
    for t in tests:
        if t["test"].startswith("Fisher-ADF") and t.get("reject_h0") is False:
            f.append("UNIT_ROOT_SUSPECT:" + t["test"].split(": ", 1)[1])
    if n_clusters < 40:
        f.append("FEW_CLUSTERS")
    if inspection_low_within:
        f.append("LOW_WITHIN_VARIATION:" + ",".join(inspection_low_within))
    if outl.get("n_abs_z_above_5", 0) > 0 or outl.get("share_abs_z_above_3", 0) > 0.02:
        f.append("OUTLIERS")
    if any(v > 10 for v in vif.values()):
        f.append("HIGH_VIF")
    return f


# ---------------------------------------------------------------- main


def run_model(name: str, m: dict, df: pd.DataFrame, skip_ur: bool) -> dict:
    y, X = design(m["formula"], df)
    keys = m.get("key") or list(drop_intercept(X).columns)
    effects = m["effects"]
    used = df.loc[y.index]
    tests: list[dict] = []

    def attempt(fn, label, h0, *args):
        try:
            r = fn(*args)
            tests.extend(r if isinstance(r, list) else [r])
        except Exception as e:  # noqa: BLE001
            tests.append(failed_test(label, h0, e))

    attempt(test_entity_effects, "F test for entity effects", "no entity effects", y, X)
    attempt(test_time_effects, "time effects", "no time effects", y, X)
    attempt(test_bp_lm, "Breusch-Pagan LM (random effects)", "no RE", y, X)
    attempt(test_hausman, "Hausman (classic)", "RE consistent", y, X, keys, effects)
    attempt(test_mundlak, "Mundlak / correlated random effects (clustered)", "RE consistent",
            y, X, keys, effects)
    attempt(test_wooldridge_ar1, "Wooldridge test for AR(1)", "no serial correlation", y, X)

    fe = fe_model(y, X, effects).fit()
    resid = fe.resids
    attempt(test_groupwise_het, "Modified Wald (groupwise heteroskedasticity)", "homoskedastic",
            resid)
    resid_e = PanelOLS(y, X, entity_effects=True, drop_absorbed=True,
                       check_rank=False).fit().resids
    attempt(test_pesaran_cd, "Pesaran CD", "no cross-sectional dependence", resid_e, resid,
            effects)
    if not skip_ur:
        variables = [y.name] + [c for c in drop_intercept(X).columns if c in used.columns]
        attempt(test_unit_roots, "Fisher-ADF", "unit root", used, variables)

    vif = vif_within(X)
    outl = outliers(resid)

    Xr = drop_intercept(X)
    within = Xr - Xr.groupby(level=0).transform("mean")
    low_within = [c for c in Xr.columns if Xr[c].var() > 0 and within[c].var() / Xr[c].var() < 0.1]
    n_clusters = int(y.index.get_level_values(0).nunique())

    estimates = {
        "pooled_ols": coef_summary(PooledOLS(y, X).fit(cov_type="clustered", cluster_entity=True),
                                   keys),
        "fe_entity": coef_summary(PanelOLS(y, X, entity_effects=True, drop_absorbed=True,
                                           check_rank=False).fit(cov_type="clustered",
                                                                 cluster_entity=True), keys),
        "fe_twoway": coef_summary(PanelOLS(y, X, entity_effects=True, time_effects=True,
                                           drop_absorbed=True, check_rank=False).fit(
                                               cov_type="clustered", cluster_entity=True), keys),
        "random_effects": coef_summary(RandomEffects(y, X, check_rank=False).fit(
            cov_type="clustered", cluster_entity=True), keys),
    }
    return {
        "formula": m["formula"], "effects": effects, "key": keys,
        "nobs": int(len(y)), "n_entities": n_clusters,
        "n_periods": int(y.index.get_level_values(1).nunique()),
        "estimates_clustered_se": estimates,
        "tests": tests, "vif_within": vif, "outliers": outl,
        "flags": make_flags(tests, n_clusters, low_within, outl, vif),
    }


def main(argv: list[str]) -> None:
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    cfg = load_config(argv[0])
    if not (cfg.get("entity") and cfg.get("time")):
        sys.exit("panel_diagnostics.py needs 'entity' and 'time' in the config")
    skip_ur = "--skip-unit-root" in argv
    df, log = load_data(cfg)
    results = {"sample_log": log, "models": {}}
    for name, m in cfg["models"].items():
        print(f"\n=== {name}: {m['formula']}  [{m['effects']} effects]")
        r = run_model(name, m, df, skip_ur)
        results["models"][name] = r
        for t in r["tests"]:
            dec = ("REJECT" if t["reject_h0"] else "keep" if t["reject_h0"] is False
                   else "error" if "error" in t else "not computed")
            stat = f"{t['statistic']:.3f}" if isinstance(t.get("statistic"), float) else "n/a"
            print(f"  {t['test'][:52]:<52} stat={stat:>10}  p={fmt_p(t.get('pvalue')):>7}  {dec}")
            if "error" in t:
                print(f"      error: {t['error']}")
        print(f"  flags: {', '.join(r['flags']) or 'none'}")
    out = Path(cfg["output_dir"]) / "panel_diagnostics.json"
    write_json(results, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
