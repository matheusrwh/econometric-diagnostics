"""Step 3 (panel data) — robustness battery for the key coefficients.

Usage:  python robustness.py <config.json> [--no-jackknife]
Writes: <output_dir>/robustness.json and prints one table per model.

The baseline is the model exactly as in the config (formula, effects, cov). Each variant
changes ONE thing relative to the baseline:

  A. Inference      unadjusted, heteroskedasticity-robust, clustered by entity / time / both,
                    Driscoll-Kraay (robust to cross-sectional dependence and autocorrelation)
  B. Estimator      pooled OLS, random effects, FE entity, FE two-way, first differences,
                    between (cross-sectional variation only), dynamic FE with lagged y
  C. Sample         winsorized 1/99, trimmed |FE residual| > 3 sd, largest balanced sub-panel,
                    first and second half of the period, leave-one-entity-out jackknife

Variants that answer a different question (pooled, between) are reported but do not count
for the stability verdict.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects

from common import design, drop_intercept, fmt_p, load_config, load_data, stars, write_json

warnings.filterwarnings("ignore")
ALPHA = 0.05

COV = {
    "unadjusted": dict(cov_type="unadjusted"),
    "robust": dict(cov_type="robust"),
    "clustered_entity": dict(cov_type="clustered", cluster_entity=True),
    "clustered_time": dict(cov_type="clustered", cluster_time=True),
    "clustered_twoway": dict(cov_type="clustered", cluster_entity=True, cluster_time=True),
    "driscoll_kraay": dict(cov_type="kernel", kernel="bartlett"),
}
COV_LABEL = {
    "unadjusted": "SE: unadjusted (iid)",
    "robust": "SE: heteroskedasticity-robust",
    "clustered_entity": "SE: clustered by entity",
    "clustered_time": "SE: clustered by time",
    "clustered_twoway": "SE: two-way clustered",
    "driscoll_kraay": "SE: Driscoll-Kraay",
}


def time_dummies(index: pd.MultiIndex) -> pd.DataFrame:
    t = pd.Series(index.get_level_values(1), index=index)
    return pd.get_dummies(t, prefix="_t", drop_first=True, dtype=float)


def fit(y, X, estimator: str, cov: str):
    kw = COV[cov]
    if estimator in ("fe_entity", "fe_twoway"):
        return PanelOLS(y, X, entity_effects=True, time_effects=(estimator == "fe_twoway"),
                        drop_absorbed=True, check_rank=False).fit(**kw)
    if estimator == "pooled":
        return PooledOLS(y, X).fit(**kw)
    if estimator == "re":
        return RandomEffects(y, X, check_rank=False).fit(**kw)
    if estimator == "between":
        return BetweenOLS(y, X).fit(cov_type="robust")
    if estimator == "fd":
        kw = kw if cov in ("unadjusted", "robust", "clustered_entity") else COV["clustered_entity"]
        return FirstDifferenceOLS(y, drop_intercept(X)).fit(**kw)
    raise ValueError(estimator)


def base_estimator(effects: str) -> str:
    return {"twoway": "fe_twoway", "entity": "fe_entity", "none": "pooled",
            "random": "re"}[effects]


def extract(res, keys) -> dict:
    out = {"nobs": int(res.nobs)}
    for k in keys:
        if k in res.params.index:
            out[k] = {"coef": float(res.params[k]), "se": float(res.std_errors[k]),
                      "pvalue": float(res.pvalues[k])}
    return out


def winsorize(df: pd.DataFrame, cols, q=0.01) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        lo, hi = df[c].quantile([q, 1 - q])
        df[c] = df[c].clip(lo, hi)
    return df


def balanced_window(idx: pd.MultiIndex, min_entities: int = 10):
    """Window [t0, t1] maximising N*T among windows where N entities are fully observed.

    t1 is searched among the last 5 periods, so one late-ending entity group does not
    force a tiny sub-panel."""
    tim = pd.Series(idx.get_level_values(1), index=idx)
    times = np.sort(tim.unique())
    best = None
    for t1 in times[-5:]:
        for t0 in times[times <= t1][: max(1, int((times <= t1).sum()) - 5)]:
            window = times[(times >= t0) & (times <= t1)]
            sub = tim[(tim >= t0) & (tim <= t1)]
            counts = sub.groupby(level=0).size()
            ents = counts[counts == len(window)].index
            score = len(ents) * len(window)
            if len(ents) >= min_entities and (best is None or score > best[0]):
                best = (score, t0, t1, list(ents))
    return best


def lag_within(s: pd.Series) -> pd.Series:
    """One-period lag within entity; NaN when the previous period is missing."""
    t = pd.Series(s.index.get_level_values(1), index=s.index).astype(float)
    step = np.nanmin(np.diff(np.sort(t.unique())))
    return s.groupby(level=0).shift(1).where(t.groupby(level=0).diff() == step)


def run_model(name: str, m: dict, df: pd.DataFrame, cfg: dict, jackknife: bool,
              diag_flags: list[str]) -> dict:
    y, X = design(m["formula"], df)
    keys = m.get("key") or list(drop_intercept(X).columns)
    effects, cov = m["effects"], m["cov"]
    est = base_estimator(effects)
    variants: list[dict] = []
    re_inconsistent = "RE_INCONSISTENT_USE_FE" in diag_flags

    def add(group, label, fn, counts=True, note=None, kind="robustness"):
        try:
            r = fn()
            rec = {"group": group, "variant": label, "counts_for_verdict": counts,
                   "kind": kind, **r}
        except Exception as e:  # noqa: BLE001
            rec = {"group": group, "variant": label, "counts_for_verdict": False,
                   "kind": kind, "error": f"{type(e).__name__}: {e}"}
        if note:
            rec["note"] = note
        variants.append(rec)

    # ---- baseline
    add("Baseline", f"Baseline ({est}, {cov})", lambda: extract(fit(y, X, est, cov), keys))

    # ---- A. inference
    for c in COV:
        if c != cov:
            add("A. Inference", COV_LABEL[c], lambda c=c: extract(fit(y, X, est, c), keys))

    # ---- B. estimator
    cov_b = cov if cov in COV else "clustered_entity"
    add("B. Estimator", "Pooled OLS", lambda: extract(fit(y, X, "pooled", cov_b), keys),
        counts=False, note="uses between and within variation; biased if entity effects matter")
    add("B. Estimator", "Random effects", lambda: extract(fit(y, X, "re", cov_b), keys),
        counts=not re_inconsistent,
        note="inconsistent here: Mundlak test rejects" if re_inconsistent else None)
    if est != "fe_entity":
        add("B. Estimator", "FE entity only", lambda: extract(fit(y, X, "fe_entity", cov_b), keys))
    if est != "fe_twoway":
        add("B. Estimator", "FE two-way", lambda: extract(fit(y, X, "fe_twoway", cov_b), keys))

    def fd():
        Xf = X if effects != "twoway" else pd.concat([X, time_dummies(y.index)], axis=1)
        return extract(fit(y, Xf, "fd", cov_b), keys)
    add("B. Estimator", "First differences" + (" + time dummies" if effects == "twoway" else ""),
        fd, note="removes entity effects by differencing; efficient if errors are a random walk")
    add("B. Estimator", "Between (entity means)", lambda: extract(fit(y, X, "between", cov_b), keys),
        counts=False, note="long-run cross-sectional relation; different estimand")

    n_periods = y.index.get_level_values(1).nunique()
    if n_periods >= 20:
        def dyn():
            ylag = lag_within(y).rename("_y_lag1")
            Xl = pd.concat([X, ylag], axis=1).dropna()
            r = extract(fit(y.loc[Xl.index], Xl, est, cov), keys + ["_y_lag1"])
            return r
        add("B. Estimator", "Dynamic FE (+ lagged y)", dyn,
            note=f"Nickell bias of order 1/T = {1 / n_periods:.3f}; short-run coefficient")

    # ---- C. sample
    yname = y.name
    used = df.loc[y.index]
    model_cols = [c for c in [yname] + list(drop_intercept(X).columns) if c in used.columns]

    def winsor():
        yw, Xw = design(m["formula"], winsorize(used, model_cols))
        return extract(fit(yw, Xw, est, cov), keys)
    add("C. Sample", "Winsorized 1%/99%", winsor)

    def trimmed():
        e = fit(y, X, est, "unadjusted").resids
        z = (e - e.mean()) / e.std(ddof=1)
        keep = z.abs() <= 3
        r = extract(fit(y[keep], X[keep], est, cov), keys)
        r["dropped"] = int((~keep).sum())
        return r
    add("C. Sample", "Trimmed |residual| > 3 sd", trimmed)

    bw = balanced_window(y.index)
    if bw is not None and bw[0] < len(y):  # skip when the panel is already balanced
        _, t0, t1, ents = bw
        sel = (y.index.get_level_values(0).isin(ents)) & (y.index.get_level_values(1) >= t0)
        sel = sel & (y.index.get_level_values(1) <= t1)
        add("C. Sample", f"Balanced sub-panel {t0}-{t1} ({len(ents)} entities)",
            lambda: extract(fit(y[sel], X[sel], est, cov), keys), kind="heterogeneity",
            note="different entities and period: differences may reflect heterogeneity")

    # split at the median observation, so both halves have about the same number of obs.
    tvals = y.index.get_level_values(1)
    mid = np.sort(np.asarray(tvals))[len(tvals) // 2]
    add("C. Sample", f"Early period (< {mid})",
        lambda: extract(fit(y[tvals < mid], X[tvals < mid], est, cov), keys),
        kind="heterogeneity", note="parameter stability over time")
    add("C. Sample", f"Late period (>= {mid})",
        lambda: extract(fit(y[tvals >= mid], X[tvals >= mid], est, cov), keys),
        kind="heterogeneity", note="parameter stability over time")

    # ---- jackknife
    jk = None
    if jackknife:
        ents_all = y.index.get_level_values(0)
        rows = []
        for e in ents_all.unique():
            keep = ents_all != e
            try:
                r = fit(y[keep], X[keep], est, cov)
                rows.append({"dropped": str(e), **{k: (float(r.params[k]), float(r.pvalues[k]))
                                                     for k in keys if k in r.params.index}})
            except Exception:  # noqa: BLE001
                continue
        base = variants[0]
        jk = {}
        for k in keys:
            vals = [(r["dropped"], *r[k]) for r in rows if k in r]
            if not vals or k not in base:
                continue
            coefs = np.array([v[1] for v in vals])
            ps = np.array([v[2] for v in vals])
            b0 = base[k]["coef"]
            imin, imax = int(coefs.argmin()), int(coefs.argmax())
            n = len(coefs)
            jk[k] = {
                "n_fits": n,
                "min": float(coefs[imin]), "min_when_dropping": vals[imin][0],
                "max": float(coefs[imax]), "max_when_dropping": vals[imax][0],
                "share_same_sign": float((np.sign(coefs) == np.sign(b0)).mean()),
                "share_significant_5pct": float((ps < ALPHA).mean()),
                "jackknife_se": float(np.sqrt((n - 1) / n * ((coefs - coefs.mean()) ** 2).sum())),
                "most_influential": sorted(
                    [{"entity": v[0], "coef": float(v[1]), "shift": float(v[1] - b0)}
                     for v in vals], key=lambda d: -abs(d["shift"]))[:5],
            }

    return {"formula": m["formula"], "effects": effects, "cov": cov, "key": keys,
            "label": m.get("label", name), "variants": variants, "jackknife": jk,
            "verdict": verdict(variants, keys, jk)}


def verdict(variants, keys, jk) -> dict:
    base = variants[0]
    out = {}
    for k in keys:
        if k not in base:
            continue
        b0 = base[k]["coef"]
        sig0 = base[k]["pvalue"] < ALPHA
        checked = [v for v in variants[1:] if v.get("counts_for_verdict") and k in v]
        rob = [v for v in checked if v["kind"] == "robustness"]
        het = [v for v in checked if v["kind"] == "heterogeneity"]
        sign_flips = [v["variant"] for v in checked if np.sign(v[k]["coef"]) != np.sign(b0)]
        lost_sig = [v["variant"] for v in rob if sig0 and v[k]["pvalue"] >= ALPHA]
        big_moves = [v["variant"] for v in rob
                     if b0 != 0 and abs(v[k]["coef"] - b0) / abs(b0) > 0.5]
        het_moves = [f"{v['variant']}: {v[k]['coef']:.4f}" for v in het
                     if b0 != 0 and abs(v[k]["coef"] - b0) / abs(b0) > 0.3]
        jk_flip = bool(jk and k in jk and jk[k]["share_same_sign"] < 1)
        if sign_flips or jk_flip:
            status = "FRAGILE"
        elif lost_sig or big_moves:
            status = "MOSTLY ROBUST"
        else:
            status = "ROBUST"
        out[k] = {"status": status, "baseline_coef": b0, "baseline_significant": bool(sig0),
                  "n_variants_checked": len(checked), "sign_flips": sign_flips,
                  "loses_significance": lost_sig, "moves_more_than_50pct": big_moves,
                  "jackknife_sign_flip": jk_flip,
                  "heterogeneity_moves_more_than_30pct": het_moves}
    return out


def print_table(name: str, r: dict) -> None:
    keys = r["key"]
    print(f"\n=== {name}: {r['formula']}")
    for v in r["variants"]:
        if "error" in v:
            print(f"  {v['variant'][:46]:<46} ERROR {v['error'][:60]}")
            continue
        cells = []
        for k in keys:
            if k in v:
                c = v[k]
                cells.append(f"{c['coef']:9.4f}{stars(c['pvalue']):<3} ({c['se']:.4f})")
        mark = "" if v["counts_for_verdict"] else "  [info]"
        print(f"  {v['variant'][:46]:<46} {'  '.join(cells)}  n={v.get('nobs', '')}{mark}")
    if r["jackknife"]:
        for k, j in r["jackknife"].items():
            print(f"  jackknife {k}: [{j['min']:.4f} (drop {j['min_when_dropping']}), "
                  f"{j['max']:.4f} (drop {j['max_when_dropping']})], same sign "
                  f"{j['share_same_sign']:.0%}, significant {j['share_significant_5pct']:.0%}")
    for k, v in r["verdict"].items():
        print(f"  VERDICT {k}: {v['status']}"
              + (f" | loses significance: {v['loses_significance']}" if v['loses_significance'] else "")
              + (f" | moves >50%: {v['moves_more_than_50pct']}" if v['moves_more_than_50pct'] else "")
              + (f" | sign flips: {v['sign_flips']}" if v['sign_flips'] else ""))
        if v["heterogeneity_moves_more_than_30pct"]:
            print(f"  HETEROGENEITY {k}: {v['heterogeneity_moves_more_than_30pct']}")


def main(argv: list[str]) -> None:
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    cfg = load_config(argv[0])
    if not (cfg.get("entity") and cfg.get("time")):
        sys.exit("robustness.py currently supports panel data; for time series / cross "
                 "sections use ols_diagnostics.py, which includes its own robustness block")
    df, log = load_data(cfg)
    results = {"sample_log": log, "models": {}}
    diag_path = Path(cfg["output_dir"]) / "panel_diagnostics.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8"))["models"] if diag_path.exists() else {}
    if not diag:
        print("note: panel_diagnostics.json not found - run it first so RE can be excluded "
              "from the verdict when Mundlak rejects")
    for name, m in cfg["models"].items():
        r = run_model(name, m, df, cfg, jackknife="--no-jackknife" not in argv,
                      diag_flags=diag.get(name, {}).get("flags", []))
        results["models"][name] = r
        print_table(name, r)
    out = Path(cfg["output_dir"]) / "robustness.json"
    write_json(results, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
