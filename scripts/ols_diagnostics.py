"""Steps 2 + 3 for time series and cross sections — diagnostics and robustness of OLS models.

Usage:  python ols_diagnostics.py <config.json>
Writes: <output_dir>/ols_diagnostics.json and prints a summary.

The structure comes from the config: 'time' without 'entity' -> time series;
neither -> cross section. Optional per-model key "cluster": column name for clustered SEs.

Time series:   ADF + KPSS per variable (levels and differences), Engle-Granger cointegration
               when variables look I(1), Breusch-Godfrey, Ljung-Box, Durbin-Watson,
               Breusch-Pagan, ARCH-LM, Jarque-Bera, RESET, CUSUM, VIF.
Cross section: Breusch-Pagan, White, RESET, Jarque-Bera, VIF, Cook's distance.
Robustness:    alternative SEs, outlier-robust estimators and sample variants (see code).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, acorr_ljungbox,
                                          breaks_cusumolsresid, het_arch, het_breuschpagan,
                                          het_white, linear_reset)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, coint, kpss

from common import (design, drop_intercept, failed_test, fmt_p, load_config, load_data, stars,
                    structure_of, test_result, write_json)
from robustness import verdict

warnings.filterwarnings("ignore")
ALPHA = 0.05


# ---------------------------------------------------------------- helpers


def integration_order(x: pd.Series) -> dict:
    """ADF (H0 unit root) + KPSS (H0 stationary) in levels and first differences."""
    x = x.dropna().astype(float)
    out = {}
    for label, s in (("level", x), ("diff", x.diff().dropna())):
        if len(s) < 12:
            continue
        adf_p = adfuller(s, autolag="AIC", regression="c")[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
        out[label] = {"adf_pvalue": float(adf_p), "kpss_pvalue": float(kpss_p)}
    lv = out.get("level", {})
    df_ = out.get("diff", {})
    adf_lv = lv.get("adf_pvalue", 1) < ALPHA      # True: unit root rejected in levels
    kpss_lv = lv.get("kpss_pvalue", 0) >= ALPHA    # True: stationarity not rejected in levels
    if adf_lv and kpss_lv:
        order = "I(0)"
    elif adf_lv and not kpss_lv:
        order = "conflicting (ADF: I(0), KPSS: not stationary)"
    elif df_ and df_["adf_pvalue"] < ALPHA:
        order = "I(1)" if kpss_lv is False else "conflicting (ADF: I(1), KPSS: stationary)"
    else:
        order = "inconclusive (possibly I(2) or short sample)"
    out["order"] = order
    return out


def vif(X: pd.DataFrame) -> dict:
    if drop_intercept(X).shape[1] < 2:
        return {}
    Xc = X if "Intercept" in X.columns else sm.add_constant(X)
    return {c: float(variance_inflation_factor(Xc.values, i))
            for i, c in enumerate(Xc.columns) if c != "Intercept" and c != "const"}


def coefs(res, keys) -> dict:
    out = {"nobs": int(res.nobs)}
    for k in keys:
        if k in res.params.index:
            out[k] = {"coef": float(res.params[k]), "se": float(res.bse[k]),
                      "pvalue": float(res.pvalues[k])}
    return out


# ---------------------------------------------------------------- tests


def ts_tests(y, X, res, used, variables) -> tuple[list[dict], dict]:
    tests = []
    n = int(res.nobs)
    lags = max(1, int(round(n ** 0.25)))
    orders = {}
    for v in variables:
        try:
            orders[v] = integration_order(used[v])
            o = orders[v]
            lv = o.get("level", {})
            rec = test_result(f"ADF/KPSS integration order: {v}",
                              "ADF: unit root / KPSS: stationary (p-value shown: ADF in levels)",
                              None, lv.get("adf_pvalue"), order=o["order"], detail=o)
            rec["reject_h0"] = None  # a classification, not a single test decision
            tests.append(rec)
        except Exception as e:  # noqa: BLE001
            tests.append(failed_test(f"ADF/KPSS integration order: {v}", "unit root", e))
    i1 = [v for v, o in orders.items() if o.get("order") == "I(1)"]
    if y.name in i1 and len(i1) >= 2:
        try:
            xs = drop_intercept(X)[[c for c in drop_intercept(X).columns if c in i1]]
            stat, p, _ = coint(y, xs)
            tests.append(test_result("Engle-Granger cointegration", "no cointegration",
                                     float(stat), float(p),
                                     note="if H0 is not rejected the levels regression may be "
                                          "spurious: estimate in differences or an ECM"))
        except Exception as e:  # noqa: BLE001
            tests.append(failed_test("Engle-Granger cointegration", "no cointegration", e))
    bg = acorr_breusch_godfrey(res, nlags=lags)
    tests.append(test_result("Breusch-Godfrey", f"no serial correlation up to lag {lags}",
                             float(bg[0]), float(bg[1]), df=lags))
    lb = acorr_ljungbox(res.resid, lags=[lags], return_df=True)
    tests.append(test_result("Ljung-Box", f"no autocorrelation up to lag {lags}",
                             float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0]),
                             df=lags, durbin_watson=float(durbin_watson(res.resid))))
    bp = het_breuschpagan(res.resid, res.model.exog)
    tests.append(test_result("Breusch-Pagan", "homoskedasticity", float(bp[0]), float(bp[1])))
    arch = het_arch(res.resid, nlags=lags)
    tests.append(test_result("ARCH-LM", "no ARCH effects", float(arch[0]), float(arch[1]),
                             df=lags))
    jb = jarque_bera(res.resid)
    tests.append(test_result("Jarque-Bera", "normal residuals", float(jb[0]), float(jb[1])))
    try:
        rs = linear_reset(res, power=3, use_f=True)
        tests.append(test_result("Ramsey RESET", "no omitted nonlinearity", float(rs.fvalue),
                                 float(rs.pvalue)))
    except Exception as e:  # noqa: BLE001
        tests.append(failed_test("Ramsey RESET", "no omitted nonlinearity", e))
    cs = breaks_cusumolsresid(res.resid, ddof=int(res.df_model) + 1)
    tests.append(test_result("CUSUM (OLS residuals)", "parameter stability", float(cs[0]),
                             float(cs[1])))
    return tests, orders


def cs_tests(res) -> list[dict]:
    tests = []
    bp = het_breuschpagan(res.resid, res.model.exog)
    tests.append(test_result("Breusch-Pagan", "homoskedasticity", float(bp[0]), float(bp[1])))
    try:
        w = het_white(res.resid, res.model.exog)
        tests.append(test_result("White", "homoskedasticity", float(w[0]), float(w[1])))
    except Exception as e:  # noqa: BLE001
        tests.append(failed_test("White", "homoskedasticity", e))
    try:
        rs = linear_reset(res, power=3, use_f=True)
        tests.append(test_result("Ramsey RESET", "no omitted nonlinearity", float(rs.fvalue),
                                 float(rs.pvalue)))
    except Exception as e:  # noqa: BLE001
        tests.append(failed_test("Ramsey RESET", "no omitted nonlinearity", e))
    jb = jarque_bera(res.resid)
    tests.append(test_result("Jarque-Bera", "normal residuals", float(jb[0]), float(jb[1])))
    return tests


def influence(res) -> dict:
    cook = res.get_influence().cooks_distance[0]
    n = len(cook)
    thr = 4 / n
    idx = np.argsort(-cook)[:10]
    return {"threshold_4_over_n": thr, "n_above": int((cook > thr).sum()),
            "top": [{"index": str(res.model.data.row_labels[i]), "cooks_d": float(cook[i])}
                    for i in idx], "_mask_above": cook > thr}


# ---------------------------------------------------------------- robustness


def robustness(name, m, df, y, X, res, keys, struct, infl) -> list[dict]:
    variants = []
    n = int(res.nobs)
    hac_lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    base_cov = m["cov"]

    def fitcov(cov, yy=y, XX=X):
        model = sm.OLS(yy, XX)
        if cov == "HAC":
            return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        if cov.startswith("cluster:"):
            g = pd.factorize(df.loc[yy.index, cov.split(":", 1)[1]])[0]
            return model.fit(cov_type="cluster", cov_kwds={"groups": g})
        if cov == "nonrobust":
            return model.fit()
        return model.fit(cov_type=cov)

    def add(group, label, fn, kind="robustness", counts=True, note=None):
        try:
            rec = {"group": group, "variant": label, "kind": kind,
                   "counts_for_verdict": counts, **fn()}
        except Exception as e:  # noqa: BLE001
            rec = {"group": group, "variant": label, "kind": kind, "counts_for_verdict": False,
                   "error": f"{type(e).__name__}: {e}"}
        if note:
            rec["note"] = note
        variants.append(rec)

    add("Baseline", f"Baseline (OLS, {base_cov})", lambda: coefs(fitcov(base_cov), keys))
    covs = ["nonrobust", "HC1", "HC3"] + (["HAC"] if struct == "time_series" else [])
    if m.get("cluster"):
        covs.append(f"cluster:{m['cluster']}")
    for c in covs:
        if c != base_cov:
            add("A. Inference", f"SE: {c}" + (f" ({hac_lags} lags)" if c == "HAC" else ""),
                lambda c=c: coefs(fitcov(c), keys))

    add("B. Estimator", "Huber robust regression (RLM)",
        lambda: coefs(sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit(), keys),
        note="downweights outliers")
    add("B. Estimator", "Median regression (LAD)",
        lambda: coefs(sm.QuantReg(y, X).fit(q=0.5), keys),
        note="conditional median instead of mean")
    if struct == "time_series":
        def fd():
            dy, dX = y.diff(), drop_intercept(X).diff()
            ok = dy.notna() & dX.notna().all(axis=1)
            return coefs(sm.OLS(dy[ok], sm.add_constant(dX[ok])).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_lags}), keys)
        add("B. Estimator", "First differences (HAC)", fd,
            note="short-run relation; robust to spurious regression in levels")

        def ldv():
            Xl = pd.concat([X, y.shift(1).rename("_y_lag1")], axis=1).dropna()
            return coefs(sm.OLS(y.loc[Xl.index], Xl).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_lags}), keys)
        add("B. Estimator", "With lagged dependent variable (HAC)", ldv,
            note="short-run coefficient; long-run = b / (1 - rho)")

    mask = infl["_mask_above"]
    add("C. Sample", "Drop Cook's D > 4/n",
        lambda: {**coefs(fitcov(base_cov, y[~mask], X[~mask]), keys), "dropped": int(mask.sum())})
    cols = [c for c in [y.name] + list(drop_intercept(X).columns) if c in df.columns]
    used = df.loc[y.index].copy()
    for c in cols:
        lo, hi = used[c].quantile([0.01, 0.99])
        used[c] = used[c].clip(lo, hi)
    add("C. Sample", "Winsorized 1%/99%",
        lambda: coefs(fitcov(base_cov, *design(m["formula"], used)), keys))
    if struct == "time_series":
        half = len(y) // 2
        add("C. Sample", f"First half (to {y.index[half - 1]})",
            lambda: coefs(fitcov(base_cov, y.iloc[:half], X.iloc[:half]), keys),
            kind="heterogeneity")
        add("C. Sample", f"Second half (from {y.index[half]})",
            lambda: coefs(fitcov(base_cov, y.iloc[half:], X.iloc[half:]), keys),
            kind="heterogeneity")
    return variants


# ---------------------------------------------------------------- main


def flags_from(tests, orders, vifs, infl, struct) -> list[str]:
    by = {t["test"]: t for t in tests}
    f = []

    def rej(n):
        return by.get(n, {}).get("reject_h0")
    if rej("Breusch-Godfrey") or rej("Ljung-Box"):
        f.append("SERIAL_CORRELATION")
    if rej("Breusch-Pagan") or rej("White"):
        f.append("HETEROSKEDASTICITY")
    if rej("ARCH-LM"):
        f.append("ARCH_EFFECTS")
    if rej("Ramsey RESET"):
        f.append("FUNCTIONAL_FORM")
    if rej("CUSUM (OLS residuals)"):
        f.append("STRUCTURAL_BREAK")
    if rej("Jarque-Bera"):
        f.append("NON_NORMAL_RESIDUALS")
    for v, o in orders.items():
        if o.get("order") != "I(0)":
            f.append(f"NONSTATIONARY_OR_UNCLEAR:{v}")
    if "Engle-Granger cointegration" in by and by["Engle-Granger cointegration"]["reject_h0"] is False:
        f.append("SPURIOUS_REGRESSION_RISK")
    if any(v > 10 for v in vifs.values()):
        f.append("HIGH_VIF")
    if infl["n_above"] > 0.10 * len(infl["_mask_above"]) or \
            (infl["top"] and infl["top"][0]["cooks_d"] > 1):
        f.append("INFLUENTIAL_OBS")
    return f


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    cfg = load_config(argv[0])
    struct = structure_of(cfg)
    if struct == "panel":
        sys.exit("panel config: use panel_diagnostics.py and robustness.py")
    df, log = load_data(cfg)
    results = {"structure": struct, "sample_log": log, "models": {}}
    for name, m in cfg["models"].items():
        if m["cov"] in ("clustered_entity", "twoway"):
            m["cov"] = "HAC" if struct == "time_series" else "HC3"
        y, X = design(m["formula"], df)
        keys = m.get("key") or list(drop_intercept(X).columns)
        res = sm.OLS(y, X).fit()
        used = df.loc[y.index]
        variables = [c for c in [y.name] + list(drop_intercept(X).columns) if c in used.columns]
        if struct == "time_series":
            tests, orders = ts_tests(y, X, res, used, variables)
        else:
            tests, orders = cs_tests(res), {}
        vifs = vif(X)
        infl = influence(res)
        variants = robustness(name, m, df, y, X, res, keys, struct, infl)
        r = {"formula": m["formula"], "cov": m["cov"], "key": keys, "label": m.get("label", name),
             "nobs": int(res.nobs), "tests": tests, "vif": vifs,
             "influence": {k: v for k, v in infl.items() if not k.startswith("_")},
             "flags": flags_from(tests, orders, vifs, infl, struct),
             "variants": variants, "verdict": verdict(variants, keys, None)}
        results["models"][name] = r
        print(f"\n=== {name}: {m['formula']}  [{struct}, n={r['nobs']}]")
        for t in tests:
            dec = ("REJECT" if t["reject_h0"] else "keep" if t["reject_h0"] is False
                   else "error" if "error" in t else "-")
            extra = f" order={t['order']}" if "order" in t else ""
            print(f"  {t['test'][:50]:<50} p={fmt_p(t.get('pvalue')):>7}  {dec}{extra}")
        print(f"  flags: {', '.join(r['flags']) or 'none'}")
        for v in variants:
            if "error" in v:
                print(f"  {v['variant'][:44]:<44} ERROR {v['error'][:60]}")
                continue
            cells = [f"{v[k]['coef']:9.4f}{stars(v[k]['pvalue']):<3} ({v[k]['se']:.4f})"
                     for k in keys if k in v]
            print(f"  {v['variant'][:44]:<44} {'  '.join(cells)}  n={v.get('nobs')}")
        for k, v in r["verdict"].items():
            print(f"  VERDICT {k}: {v['status']}")
    out = Path(cfg["output_dir"]) / "ols_diagnostics.json"
    write_json(results, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
