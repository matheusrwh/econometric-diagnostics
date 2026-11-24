"""IV / 2SLS (and shift-share IV) — diagnostics and robustness in one pass.

Usage:  python iv_diagnostics.py <config.json> [--no-jackknife]
Writes: <output_dir>/iv_diagnostics.json and prints a summary per model.

Models are written with a bracket block:  "y ~ 1 + w1 + w2 + [x ~ z1 + z2]".
Covered: cross sections and pooled data (fixed effects only as C() dummies in the formula).

Diagnostics
  first stage ......... coefficients, partial R2, robust F (= Kleibergen-Paap Wald F when
                        K1 = 1), effective F (Montiel Olea-Pflueger), Cragg-Donald F (iid),
                        Sanderson-Windmeijer conditional F when K1 > 1
  reduced form ........ instruments in the y equation (no first stage -> no IV effect)
  weak-IV-robust ...... Anderson-Rubin test and confidence set (K1 = 1)
  endogeneity ......... control-function (regression-based Durbin-Wu-Hausman) test
  overidentification .. Hansen J (robust / clustered), Sargan (iid), C statistic per instrument
  shift-share ......... AKM and AKM0 inference, AKM first-stage F, Rotemberg weights,
                        effective number of shocks, sum-of-shares check (model key "shift_share")
Robustness (each row changes one thing)
  A. inference: unadjusted / robust / clustered, AKM for shift-share
  B. estimator: OLS [info], 2SLS, LIML, Fuller(1), two-step GMM when overidentified
  C. instruments: leave one instrument out, strongest instrument only, sum-of-shares control
  D. sample: winsorized y and x, trimmed residuals, jackknife by cluster or by shock
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS, IVGMM, IVLIML
from scipy import stats

import iv_core as C
from common import (fmt_p, load_config, load_data, read_any, stars, test_result, write_json)
from robustness import verdict

warnings.filterwarnings("ignore")
ALPHA = 0.05
MOP_CRIT = 23.109        # Montiel Olea-Pflueger simplified test, tau = 10 % bias, 5 % level
LMMP_F = 104.7           # Lee et al. (2022): F needed for the usual 1.96 t-test (L = 1)
SY_SIZE_10 = {1: 16.38, 2: 19.93, 3: 22.30, 4: 24.58, 5: 26.87}  # Stock-Yogo, K1 = 1, 10 % size


# ---------------------------------------------------------------- helpers


def cov_spec(cov: str, rows: pd.DataFrame, debiased: bool) -> C.CovSpec:
    if cov in ("unadjusted", "nonrobust"):
        return C.CovSpec("unadjusted", None, debiased)
    if cov in ("robust", "HC0", "HC1"):
        return C.CovSpec("robust", None, debiased)
    if cov.startswith("cluster:"):
        return C.CovSpec("cluster", rows[cov.split(":", 1)[1]].to_numpy(), debiased)
    sys.exit(f"iv_diagnostics: unsupported cov '{cov}' (use unadjusted, robust or cluster:<col>)")


def cov_label(spec: C.CovSpec, col: str | None = None) -> str:
    return {"unadjusted": "SE: unadjusted (iid)", "robust": "SE: heteroskedasticity-robust",
            "cluster": f"SE: clustered by {col}"}[spec.kind]


def lm_fit(kind, y, W, X, Z, w, spec: C.CovSpec):
    """Estimate with linearmodels (the numbers users compare with their own output)."""
    kw = {"weights": w} if w is not None else {}
    exog = W if W.shape[1] else None
    clusters = (pd.Series(pd.factorize(np.asarray(spec.groups))[0], index=y.index)
                if spec.kind == "cluster" else None)
    cov = {"unadjusted": {"cov_type": "unadjusted"}, "robust": {"cov_type": "robust"},
           "cluster": {"cov_type": "clustered", "clusters": clusters}}
    fit_kw = {**cov[spec.kind], "debiased": spec.debiased}
    if kind == "ols":
        return IV2SLS(y, pd.concat([W, X], axis=1), None, None, **kw).fit(**fit_kw)
    if kind == "2sls":
        return IV2SLS(y, exog, X, Z, **kw).fit(**fit_kw)
    if kind == "liml":
        return IVLIML(y, exog, X, Z, **kw).fit(**fit_kw)
    if kind == "fuller":
        return IVLIML(y, exog, X, Z, fuller=1, **kw).fit(**fit_kw)
    if kind == "gmm":
        wt = {"unadjusted": {"weight_type": "unadjusted"}, "robust": {"weight_type": "robust"},
              "cluster": {"weight_type": "clustered", "clusters": clusters}}
        return IVGMM(y, exog, X, Z, **kw, **wt[spec.kind]).fit(**fit_kw)
    raise ValueError(kind)


def extract(res, keys) -> dict:
    out = {"nobs": int(res.nobs)}
    for k in keys:
        if k in res.params.index:
            out[k] = {"coef": float(res.params[k]), "se": float(res.std_errors[k]),
                      "pvalue": float(res.pvalues[k])}
    return out


def fmt_set(s: dict) -> str:
    if s.get("empty"):
        return "empty"
    parts = []
    for a, b in s["intervals"]:
        fa = "-inf" if a == -math.inf else f"{a:.3f}"
        fb = "+inf" if b == math.inf else f"{b:.3f}"
        parts.append(f"[{fa}; {fb}]")
    return " U ".join(parts) + (" (grid edge)" if s.get("touches_grid_edge") else "")


def strength(F: float) -> str:
    if F >= MOP_CRIT:
        return f"strong (>= {MOP_CRIT:.1f})"
    return "moderately weak (10-23.1)" if F >= 10 else "weak (< 10)"


# ---------------------------------------------------------------- shift-share data


def _norm(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s) and np.allclose(s.dropna() % 1, 0):
        s = s.astype("Int64")
    return s.astype(str)


def _keys(frame: pd.DataFrame, cols) -> pd.Index:
    cols = [cols] if isinstance(cols, str) else list(cols)
    miss = [c for c in cols if c not in frame.columns]
    if miss:
        raise KeyError(f"columns {miss} not found")
    if len(cols) == 1:
        return pd.Index(_norm(frame[cols[0]]))
    return pd.MultiIndex.from_arrays([_norm(frame[c]) for c in cols])


class ShiftShare:
    """Exposure (long: obs id, shock id, share) and shocks (shock id, shock, [cluster])."""

    def __init__(self, ss: dict, cfg_dir: str):
        def p(f):
            f = Path(f)
            return str(f if f.is_absolute() else Path(cfg_dir) / f)
        self.cfg = ss
        self.exp = read_any(p(ss["exposure"]))
        shk = read_any(p(ss["shocks"])).dropna(subset=[ss["shock"]])
        self.shock_keys = _keys(shk, ss["shock_id"])
        if self.shock_keys.has_duplicates:
            sys.exit("shift_share: duplicated shock ids in the shocks file")
        self.g = shk[ss["shock"]].to_numpy(float)
        cl = ss.get("shock_cluster")
        self.clusters = shk[cl].astype(str).to_numpy() if cl else None
        self.names = np.array(["|".join(map(str, k)) if isinstance(k, tuple) else str(k)
                               for k in self.shock_keys])

    def matrix(self, rows: pd.DataFrame):
        ss = self.cfg
        obs = _keys(rows, ss["obs_id"])
        if obs.has_duplicates:
            sys.exit("shift_share: obs_id does not identify the estimation rows uniquely")
        r = obs.get_indexer(_keys(self.exp, ss["obs_id"]))
        c = self.shock_keys.get_indexer(_keys(self.exp, ss["shock_id"]))
        ok = (r >= 0) & (c >= 0)
        S = np.zeros((len(rows), len(self.shock_keys)))
        np.add.at(S, (r[ok], c[ok]), self.exp[ss["share"]].to_numpy(float)[ok])
        used = S.any(axis=0)
        info = {"exposure_rows_outside_sample": int((r < 0).sum()),
                "exposure_rows_without_shock": int(((r >= 0) & (c < 0)).sum()),
                "sample_obs_without_exposure": int((~S.any(axis=1)).sum()),
                "shocks_with_exposure": int(used.sum())}
        cl = None if self.clusters is None else self.clusters[used]
        return S[:, used], self.g[used], self.names[used], cl, info


# ---------------------------------------------------------------- model


def run_model(name: str, m: dict, df: pd.DataFrame, cfg: dict, jackknife: bool) -> dict:
    debiased = cfg.get("debiased", True)
    cov = m.get("cov")
    if cov in (None, "HC3", "HAC", "clustered_entity"):  # defaults inserted by load_config
        cov = (f"cluster:{cfg['cluster']}" if cfg.get("cluster")
               else f"cluster:{cfg['entity']}" if cfg.get("entity") else "robust")
    wcol = cfg.get("weights")
    ccol = cov.split(":", 1)[1] if cov.startswith("cluster:") else cfg.get("cluster")
    need = [c for c in (wcol, ccol) if c]
    d = df.dropna(subset=need) if need else df

    ss_cfg = m.get("shift_share")
    if ss_cfg is True:
        ss_cfg = cfg["shift_share"]
    ssd = ShiftShare(ss_cfg, cfg["_config_dir"]) if ss_cfg else None
    constructed = False
    if ssd and ss_cfg["instrument"] not in d.columns:
        S_all, g_all, *_ = ssd.matrix(d)
        z = S_all @ g_all
        d = d.copy()
        d[ss_cfg["instrument"]] = np.where(S_all.any(axis=1), z, np.nan)
        constructed = True

    y, W, X, Z = C.iv_design(m["formula"], d)
    rows = d.loc[y.index]
    w = rows[wcol].to_numpy(float) if wcol else None
    spec = cov_spec(cov, rows, debiased)
    est = m.get("estimator", "2sls")
    endog, instr = list(X.columns), list(Z.columns)
    K1, L = len(endog), len(instr)
    if L < K1:
        sys.exit(f"model {name}: fewer instruments ({L}) than endogenous regressors ({K1})")
    keys = m.get("key") or endog
    Wn, Xn, Zn, yn = W.to_numpy(float), X.to_numpy(float), Z.to_numpy(float), y.to_numpy(float)
    wn = np.ones(len(yn)) if w is None else w
    tests: list[dict] = []
    flags: list[str] = []

    # ---- baseline + numerical cross-check
    base_res = lm_fit(est, y, W, X, Z, w, spec)
    check = None
    if est == "2sls":
        b, V, _ = C.tsls_fit(yn, np.hstack([Wn, Xn]), np.hstack([Wn, Zn]), wn, spec)
        names = list(W.columns) + endog
        check = float(max(np.max(np.abs(b - base_res.params[names].to_numpy())),
                          np.max(np.abs(np.sqrt(np.diag(V)) - base_res.std_errors[names].to_numpy()))))
    b0 = {k: float(base_res.params[k]) for k in keys}
    se0 = {k: float(base_res.std_errors[k]) for k in keys}

    estimates = {}
    for kind in ["ols", "2sls", "liml"] + (["gmm"] if L > K1 else []):
        try:
            estimates[kind] = extract(lm_fit(kind, y, W, X, Z, w, spec), keys)
        except Exception as e:  # noqa: BLE001
            estimates[kind] = {"error": f"{type(e).__name__}: {e}"}

    # ---- first stage
    fs = C.first_stage(Wn, Xn, Zn, wn, spec)
    first_stage = {}
    for j, xname in enumerate(endog):
        r = fs["by_endog"][j]
        first_stage[xname] = {k: v for k, v in r.items() if not k.startswith("_")}
        first_stage[xname]["instruments"] = instr
    if K1 == 1:
        r = fs["by_endog"][0]
        sy = SY_SIZE_10.get(L)
        tests.append(test_result("First stage: effective F (Montiel Olea-Pflueger)",
                                 "instruments are weak (simplified test, 10 % worst-case bias)",
                                 r["effective_F"], None, display=f"{r['effective_F']:.2f}",
                                 critical_value=MOP_CRIT, classification=strength(r["effective_F"])))
        tests.append(test_result("First stage: robust F (Kleibergen-Paap Wald)",
                                 "excluded instruments have zero first-stage coefficients",
                                 r["robust_F"], r["robust_F_pvalue"], df=L))
        tests.append(test_result("First stage: Cragg-Donald F (iid)",
                                 "weak instruments (Stock-Yogo, valid only under iid errors)",
                                 r["cragg_donald_F"], None,
                                 display=f"{r['cragg_donald_F']:.2f}" + (f" [SY {sy}]" if sy else ""),
                                 stock_yogo_10pct_size=sy))
        tests[-1]["reject_h0"] = None
        tests[-3]["reject_h0"] = bool(r["effective_F"] >= MOP_CRIT)
    else:
        for xname, sw in zip(endog, C.sanderson_windmeijer(Wn, Xn, Zn, wn, spec)):
            first_stage[xname]["sanderson_windmeijer_F"] = sw["F"]
            tests.append(test_result(f"First stage: Sanderson-Windmeijer F ({xname})",
                                     "instruments do not identify this regressor given the others",
                                     sw["F"], sw["pvalue"], df=L - K1 + 1))

    # ---- reduced form
    WZ = np.hstack([Wn, Zn])
    br, Vr, _ = C.tsls_fit(yn, WZ, WZ, wn, spec)
    idx = list(range(Wn.shape[1], WZ.shape[1]))
    Frf = C.wald(br, Vr, idx) / L
    reduced_form = {"coef": dict(zip(instr, br[idx].tolist())),
                    "se": dict(zip(instr, np.sqrt(np.diag(Vr)[idx]).tolist())),
                    "F": Frf, "pvalue": C.f_pvalue(Frf, L, C.dof(spec, len(yn), WZ.shape[1]))}
    # with one endogenous regressor this is also the Anderson-Rubin test of beta = 0
    tests.append(test_result("Reduced form (= Anderson-Rubin test of beta = 0)",
                             "instruments have no effect on y", Frf, reduced_form["pvalue"], df=L))

    # ---- Anderson-Rubin
    ar = None
    if K1 == 1:
        x0 = endog[0]
        hw = max(20 * float(base_res.std_errors[x0]), 2 * abs(float(base_res.params[x0])), 1.0)
        ar = C.anderson_rubin(Wn, Xn[:, 0], Zn, yn, wn, spec,
                              center=float(base_res.params[x0]), grid_halfwidth=hw)
        tests.append(test_result("Anderson-Rubin 95% confidence set", "weak-IV-robust set",
                                 None, None, display=fmt_set(ar)))

    # ---- endogeneity
    cf = C.control_function(Wn, Xn, Zn, yn, wn, spec)
    tests.append(test_result("Endogeneity: control function (robust DWH)",
                             "regressors are exogenous (OLS consistent)", cf["F"], cf["pvalue"],
                             df=K1))

    # ---- overidentification
    overid = {}
    if L > K1:
        try:
            g_res = lm_fit("gmm", y, W, X, Z, w, spec)
            J = g_res.j_stat
            overid["hansen_J"] = {"stat": float(J.stat), "pvalue": float(J.pval), "df": L - K1}
            tests.append(test_result("Hansen J (overidentification)", "all instruments valid",
                                     float(J.stat), float(J.pval), df=L - K1))
        except Exception as e:  # noqa: BLE001
            overid["hansen_J"] = {"error": str(e)}
        s = lm_fit("2sls", y, W, X, Z, w, C.CovSpec("unadjusted", None, debiased)).sargan
        overid["sargan"] = {"stat": float(s.stat), "pvalue": float(s.pval)}
        tests.append(test_result("Sargan (iid)", "all instruments valid", float(s.stat),
                                 float(s.pval), df=L - K1))
        if L - 1 >= K1 and "stat" in overid.get("hansen_J", {}):
            cstats = {}
            for zj in instr:
                try:
                    Jr = lm_fit("gmm", y, W, X, Z.drop(columns=zj), w, spec).j_stat
                    cs = overid["hansen_J"]["stat"] - float(Jr.stat)
                    cstats[zj] = {"stat": cs, "pvalue": float(stats.chi2.sf(max(cs, 0), 1))}
                except Exception as e:  # noqa: BLE001
                    cstats[zj] = {"error": str(e)}
            overid["c_statistic"] = cstats

    # ---- shift-share
    ss_out = None
    if ssd:
        if K1 != 1 or L != 1:
            ss_out = {"error": "shift-share block needs one endogenous regressor and the "
                               "shift-share variable as the only excluded instrument"}
        else:
            S, g, snames, scl, info = ssd.matrix(rows)
            zname = ss_cfg["instrument"]
            if zname != instr[0]:
                sys.exit(f"model {name}: shift_share.instrument '{zname}' is not the instrument "
                         f"in the formula ({instr[0]})")
            z_rebuilt = S @ g
            diff = float(np.max(np.abs(z_rebuilt - Zn[:, 0])))
            scale = float(np.max(np.abs(Zn[:, 0]))) or 1.0
            match = diff <= 1e-6 * scale
            ss_out = {"instrument_constructed_by_script": constructed,
                      "instrument_rebuilt_max_abs_diff": diff, "instrument_matches": match,
                      "data_checks": info}
            if not match:
                flags.append("SHIFT_SHARE_INSTRUMENT_MISMATCH")
                ss_out["error"] = ("the instrument in the data is not sum_k share_ik * shock_k "
                                   "built from the exposure and shock files: fix the files first")
            else:
                ak = C.akm(Wn, Xn[:, 0], Zn[:, 0], yn, S, g, wn)
                ak_c = C.akm(Wn, Xn[:, 0], Zn[:, 0], yn, S, g, wn, scl) if scl is not None else None
                rot = C.rotemberg(Wn, Xn[:, 0], yn, S, g, wn, snames)
                shocks = C.shock_summary(S, g, wn, scl)
                sums = C.share_sums(S)
                controlled = sums["complete"] or C.is_spanned(sums["_e"], Wn, wn)
                main_ak = ak_c or ak
                ss_out.update({
                    "akm": ak, "akm_clustered_shocks": ak_c,
                    "rotemberg": {k: v for k, v in rot.items() if not k.startswith("_")},
                    "shocks": shocks,
                    "share_sums": {k: v for k, v in sums.items() if not k.startswith("_")},
                    "share_sum_controlled": bool(controlled),
                    "akm_to_baseline_se_ratio": main_ak["se"] / se0[endog[0]],
                })
                lbl = " (shock clusters)" if ak_c else ""
                tests.append(test_result(f"Shift-share: AKM standard error{lbl}",
                                         "shock-level inference", None, None,
                                         display=f"{main_ak['se']:.4f} (baseline {se0[endog[0]]:.4f})"))
                tests.append(test_result(f"Shift-share: AKM first-stage F{lbl}",
                                         "instrument irrelevant (shock-level inference)",
                                         main_ak["first_stage_F"],
                                         float(stats.chi2.sf(main_ak["first_stage_F"], 1)), df=1))
                tests.append(test_result(f"Shift-share: AKM0 test (beta = 0){lbl}",
                                         "beta = 0, null-imposed shock-level inference",
                                         main_ak["akm0"]["stat_beta0"],
                                         main_ak["akm0"]["pvalue_beta0"], df=1))
                tests.append(test_result(f"Shift-share: AKM0 95% confidence set{lbl}",
                                         "weak-IV-robust shock-level set", None, None,
                                         display=fmt_set(main_ak["akm0"])))
                tests.append(test_result("Shift-share: effective number of shocks",
                                         "1 / HHI of average exposure", None, None,
                                         display=f"{shocks['effective_n_shocks']:.1f}"))
                tests.append(test_result("Shift-share: top-5 Rotemberg weight share",
                                         "share of |alpha| in the 5 largest", None, None,
                                         display=f"{rot['top5_abs_alpha_share']:.2f}"))
                tests.append(test_result("Shift-share: sum of negative Rotemberg weights",
                                         "negative weights", None, None,
                                         display=f"{rot['sum_negative_alpha']:.3f}"))
                tests.append(test_result("Shift-share: sum of shares (min-max)",
                                         "complete shares", None, None,
                                         display=f"{sums['min']:.2f}-{sums['max']:.2f}"
                                                 + ("" if controlled else " (not controlled)")))
                if not controlled:
                    flags.append("INCOMPLETE_SHARES_NOT_CONTROLLED")
                if shocks["effective_n_shocks"] < 20:
                    flags.append("FEW_EFFECTIVE_SHOCKS")
                if rot["sum_negative_alpha"] < -0.10:
                    flags.append("NEGATIVE_ROTEMBERG_WEIGHTS")
                if rot["top5_abs_alpha_share"] > 0.5:
                    flags.append("CONCENTRATED_ROTEMBERG_WEIGHTS")
                if ss_out["akm_to_baseline_se_ratio"] > 1.25:
                    flags.append("AKM_SE_LARGER_THAN_BASELINE")
                if ak["shocks_per_obs"] > 0.5:
                    flags.append("AKM_MANY_SHOCKS_PER_OBS")
                ss_out["_S"], ss_out["_g"], ss_out["_cl"], ss_out["_names"] = S, g, scl, snames
                ss_out["_e"], ss_out["_main"] = sums["_e"], main_ak

    # ---- flags
    if K1 == 1:
        F_eff = fs["by_endog"][0]["effective_F"]
        F_ref = min(F_eff, ss_out["_main"]["first_stage_F"]) if ss_out and "_main" in ss_out else F_eff
        if F_ref < MOP_CRIT:
            flags.append("WEAK_INSTRUMENTS")
        if L == 1 and F_ref < LMMP_F:
            flags.append("T_TEST_NEEDS_F_ABOVE_104.7")
        robust_set = ss_out["_main"]["akm0"] if ss_out and "_main" in ss_out else ar
        if robust_set and not robust_set["bounded"] and not robust_set["empty"]:
            flags.append("ROBUST_SET_UNBOUNDED")
        elif robust_set and robust_set["contains_zero"]:
            flags.append("ROBUST_SET_INCLUDES_ZERO")
    elif any(first_stage[x]["sanderson_windmeijer_F"] < 10 for x in endog):
        flags.append("WEAK_INSTRUMENTS")
    if cf["pvalue"] is not None and cf["pvalue"] >= ALPHA:
        flags.append("ENDOGENEITY_NOT_REJECTED")
    if overid.get("hansen_J", {}).get("pvalue", 1) < ALPHA:
        flags.append("OVERID_REJECTED")
    if spec.kind == "cluster" and spec.n_groups < 40:
        flags.append("FEW_CLUSTERS")

    # ---- robustness battery
    variants: list[dict] = []

    def add(group, label, fn, counts=True, kind="robustness", note=None):
        try:
            rec = {"group": group, "variant": label, "counts_for_verdict": counts, "kind": kind,
                   **fn()}
        except Exception as e:  # noqa: BLE001
            rec = {"group": group, "variant": label, "counts_for_verdict": False, "kind": kind,
                   "error": f"{type(e).__name__}: {e}"}
        if note:
            rec["note"] = note
        variants.append(rec)

    add("Baseline", f"Baseline ({est}, {cov})", lambda: extract(base_res, keys))
    alt_specs = [C.CovSpec("unadjusted", None, debiased), C.CovSpec("robust", None, debiased)]
    if ccol and ccol in rows.columns:
        alt_specs.append(C.CovSpec("cluster", rows[ccol].to_numpy(), debiased))
    for s in alt_specs:
        if s.kind != spec.kind:
            add("A. Inference", cov_label(s, ccol), lambda s=s: extract(lm_fit(est, y, W, X, Z, w, s), keys))
    if ss_out and "_main" in ss_out:
        for lab, a_ in (("SE: AKM (shift-share)", ss_out["akm"]),
                        ("SE: AKM, shock clusters", ss_out["akm_clustered_shocks"])):
            if a_ is not None and endog[0] in keys:
                add("A. Inference", lab, lambda a_=a_: {
                    "nobs": int(len(yn)),
                    endog[0]: {"coef": a_["beta"], "se": a_["se"], "pvalue": a_["pvalue"]}})

    add("B. Estimator", "OLS", lambda: extract(lm_fit("ols", y, W, X, Z, w, spec), keys),
        counts=False, note="biased if the regressor is endogenous; shows the direction of the IV correction")
    for kind, lab in (("2sls", "2SLS"), ("liml", "LIML"), ("fuller", "Fuller (1)"),
                      ("gmm", "Two-step GMM")):
        if kind == est or (L == K1 and kind in ("liml", "gmm")):
            continue  # LIML and GMM coincide with 2SLS when just identified
        add("B. Estimator", lab, lambda kind=kind: extract(lm_fit(kind, y, W, X, Z, w, spec), keys))

    if L > K1:
        for zj in instr:
            add("C. Instruments", f"Without instrument {zj}",
                lambda zj=zj: extract(lm_fit(est if est != "gmm" or L - 1 > K1 else "2sls",
                                             y, W, X, Z.drop(columns=zj), w, spec), keys),
                kind="heterogeneity",
                note="different instruments can identify different LATEs")
        if K1 == 1 and L > 2:  # with L = 2 it repeats a leave-one-out row
            tstat = np.abs(np.array(fs["by_endog"][0]["coef"]) / np.array(fs["by_endog"][0]["se"]))
            zbest = instr[int(tstat.argmax())]
            add("C. Instruments", f"Strongest instrument only ({zbest})",
                lambda: extract(lm_fit("2sls", y, W, X, Z[[zbest]], w, spec), keys),
                kind="heterogeneity", note="just-identified with the strongest instrument")
    if ss_out and "_main" in ss_out and not ss_out["share_sum_controlled"]:
        Wsum = pd.concat([W, pd.Series(ss_out["_e"], index=W.index, name="_sum_of_shares")], axis=1)
        add("C. Instruments", "Control for the sum of shares",
            lambda: extract(lm_fit(est, y, Wsum, X, Z, w, spec), keys),
            note="Borusyak-Hull-Jaravel: needed with incomplete shares")

    # Winsorizing x without the instruments breaks the first stage where both have long
    # tails (e.g. inflows to large metros): only y alone, or y, x and Z together. Shift-share
    # instruments are not winsorized (z = S g would no longer hold).
    def cap(frame):
        return frame.apply(lambda c: c.clip(*c.quantile([0.01, 0.99])))

    def winsor(with_xz: bool):
        yw = y.clip(*y.quantile([0.01, 0.99]))
        Xw, Zw = (cap(X), cap(Z)) if with_xz else (X, Z)
        r = extract(lm_fit(est, yw, W, Xw, Zw, w, spec), keys)
        if with_xz:
            Xa, Za = Xw.to_numpy(float), Zw.to_numpy(float)
            F = ([C.first_stage(Wn, Xa, Za, wn, spec)["by_endog"][0]["effective_F"]] if K1 == 1
                 else [s["F"] for s in C.sanderson_windmeijer(Wn, Xa, Za, wn, spec)])
            r["first_stage_F"] = F
            if min(F) < 10:
                # a weak first stage makes this row's Wald inference unreliable: it shows
                # that the identifying variation sits in the tails, it does not test fragility
                r["counts_for_verdict"] = False
                r["note"] = (f"first stage weak after winsorizing (F = {min(F):.1f}): the "
                             "identifying variation is in the tails of x and the instruments")
        return r
    add("D. Sample", "Winsorized y 1%/99%", lambda: winsor(False))
    if not ss_cfg:
        add("D. Sample", "Winsorized y, x and instruments 1%/99%", lambda: winsor(True))
        if variants[-1].get("counts_for_verdict") is False and "error" not in variants[-1]:
            flags.append("IDENTIFICATION_FROM_TAILS")

    def trimmed():
        _, _, u = C.tsls_fit(yn, np.hstack([Wn, Xn]), np.hstack([Wn, Zn]), wn, spec)
        z = (u - u.mean()) / u.std(ddof=1)
        keep = np.abs(z) <= 3
        r = extract(lm_fit(est, y[keep], W[keep], X[keep], Z[keep],
                           None if w is None else w[keep], spec.subset(keep)), keys)
        r["dropped"] = int((~keep).sum())
        return r
    add("D. Sample", "Trimmed |residual| > 3 sd", trimmed)

    # ---- jackknife
    jk = None
    if jackknife:
        jk = shock_jackknife(ss_out, Wn, Xn, Zn, yn, wn, keys, endog, b0) if ss_out and "_main" in ss_out \
            else cluster_jackknife(spec, Wn, Xn, Zn, yn, wn, keys, endog, b0)

    v = verdict(variants, keys, jk)
    # weak-IV-robust inference enters the verdict for the endogenous key coefficient
    if K1 == 1 and endog[0] in v:
        robust_set = ss_out["_main"]["akm0"] if ss_out and "_main" in ss_out else ar
        label = "AKM0 confidence set" if ss_out and "_main" in ss_out else "Anderson-Rubin confidence set"
        vv = v[endog[0]]
        vv["weak_iv_robust_set"] = fmt_set(robust_set)
        vv["weak_instruments"] = "WEAK_INSTRUMENTS" in flags
        if not robust_set["bounded"] and not robust_set["empty"]:
            vv["status"] = "FRAGILE"
            vv["robust_set_unbounded"] = True
        elif robust_set["contains_zero"] and vv["baseline_significant"]:
            vv["loses_significance"].append(label)
            if vv["status"] == "ROBUST":
                vv["status"] = "MOSTLY ROBUST"

    if ss_out:
        for k in [k for k in ss_out if k.startswith("_")]:
            ss_out.pop(k)
    return {"formula": m["formula"], "estimator": est, "cov": cov, "key": keys,
            "label": m.get("label", name), "nobs": int(len(yn)),
            "n_clusters": spec.n_groups, "endogenous": endog, "instruments": instr,
            "numpy_crosscheck_max_abs_diff": check, "estimates": estimates,
            "first_stage": first_stage, "reduced_form": reduced_form,
            "anderson_rubin": ar, "control_function": cf, "overid": overid,
            "shift_share": ss_out, "tests": tests, "flags": flags,
            "variants": variants, "jackknife": jk, "verdict": v}


# ---------------------------------------------------------------- jackknives


def _jk_summary(labels, coefs, pvals, b0, extra=None) -> dict:
    coefs = np.asarray(coefs, float)
    ok = np.isfinite(coefs)
    labels, coefs = [l for l, o in zip(labels, ok) if o], coefs[ok]
    imin, imax = int(coefs.argmin()), int(coefs.argmax())
    n = len(coefs)
    out = {"n_fits": n, "min": float(coefs[imin]), "min_when_dropping": labels[imin],
           "max": float(coefs[imax]), "max_when_dropping": labels[imax],
           "share_same_sign": float((np.sign(coefs) == np.sign(b0)).mean()),
           "share_significant_5pct": None if pvals is None
           else float((np.asarray(pvals, float)[ok] < ALPHA).mean()),
           "jackknife_se": float(np.sqrt((n - 1) / n * ((coefs - coefs.mean()) ** 2).sum())),
           "most_influential": sorted([{"dropped": l, "coef": float(c), "shift": float(c - b0)}
                                       for l, c in zip(labels, coefs)],
                                      key=lambda d: -abs(d["shift"]))[:5]}
    if extra:
        out.update(extra)
    return out


def cluster_jackknife(spec, Wn, Xn, Zn, yn, wn, keys, endog, b0):
    if spec.kind != "cluster" or spec.n_groups > 300:
        return None
    R, Q = np.hstack([Wn, Xn]), np.hstack([Wn, Zn])
    groups = np.asarray(spec.groups)
    labels, rows_b, rows_p, fmin = [], [], [], []
    kw = Wn.shape[1]
    for gval in pd.unique(groups):
        keep = groups != gval
        s = spec.subset(keep)
        try:
            b, V, _ = C.tsls_fit(yn[keep], R[keep], Q[keep], wn[keep], s)
            se = np.sqrt(np.diag(V))
            d2 = C.dof(s, int(keep.sum()), R.shape[1])
            t = b / se
            p = 2 * (stats.t.sf(np.abs(t), d2) if d2 else stats.norm.sf(np.abs(t)))
            if Xn.shape[1] == 1:
                fmin.append(C.first_stage(Wn[keep], Xn[keep], Zn[keep], wn[keep], s)
                            ["by_endog"][0]["effective_F"])
        except Exception:  # noqa: BLE001
            continue
        labels.append(str(gval)); rows_b.append(b); rows_p.append(p)
    if not labels:
        return None
    out = {}
    for k in keys:
        if k not in endog:
            continue
        j = kw + endog.index(k)
        extra = {"unit": "cluster", "min_first_stage_effective_F": float(min(fmin)) if fmin else None}
        out[k] = _jk_summary(labels, [r[j] for r in rows_b], [p[j] for p in rows_p], b0[k], extra)
    return out or None


def shock_jackknife(ss, Wn, Xn, Zn, yn, wn, keys, endog, b0):
    k = endog[0]
    if k not in keys:
        return None
    S, g, cl, names = ss["_S"], ss["_g"], ss["_cl"], ss["_names"]
    groups = cl if cl is not None else names
    if ss["share_sum_controlled"] and not ss["share_sums"]["complete"]:
        # the sum of shares is a control: dropping a shock changes it, so refit with the
        # dropped exposure as an extra control (spans the corrected sum of shares)
        codes, uniq = pd.factorize(np.asarray(groups))
        betas, pis = [], []
        for c in range(len(uniq)):
            sel = codes == c
            Sc = S[:, sel]
            Wc = np.hstack([Wn, Sc.sum(axis=1, keepdims=True)])
            zc = Zn[:, 0] - Sc @ g[sel]
            yt, xt, zt = (C.residualize(a, Wc, wn) for a in (yn, Xn[:, 0], zc))
            betas.append(float((wn * zt * yt).sum() / (wn * zt * xt).sum()))
            pis.append(float((wn * zt * xt).sum() / (wn * zt * zt).sum()))
        labels, beta, pi = [str(u) for u in uniq], np.array(betas), np.array(pis)
        method = "refit with corrected sum-of-shares control"
    else:
        r = C.shock_jackknife(Wn, Xn[:, 0], Zn[:, 0], yn, S, g, wn, groups)
        labels, beta, pi = r["labels"], r["beta"], r["pi"]
        method = "closed form, controls fixed"
    extra = {"unit": "shock cluster" if cl is not None else "shock", "method": method,
             "first_stage_coef_range": [float(np.nanmin(pi)), float(np.nanmax(pi))]}
    return {k: _jk_summary(labels, beta, None, b0[k], extra)}


# ---------------------------------------------------------------- main


def print_model(name: str, r: dict) -> None:
    print(f"\n=== {name}: {r['formula']}  [{r['estimator']}, {r['cov']}, n={r['nobs']}]")
    if r["numpy_crosscheck_max_abs_diff"] is not None:
        print(f"  numerical cross-check vs linearmodels: max |diff| = "
              f"{r['numpy_crosscheck_max_abs_diff']:.2e}")
    for t in r["tests"]:
        if t.get("display") is not None:
            val = t["display"]
        else:
            val = f"stat={t['statistic']:.3f}  p={fmt_p(t.get('pvalue'))}" \
                if t.get("statistic") is not None else "n/a"
        print(f"  {t['test'][:58]:<58} {val}")
    print(f"  flags: {', '.join(r['flags']) or 'none'}")
    for v in r["variants"]:
        if "error" in v:
            print(f"  {v['variant'][:46]:<46} ERROR {v['error'][:70]}")
            continue
        cells = [f"{v[k]['coef']:9.4f}{stars(v[k]['pvalue']):<3} ({v[k]['se']:.4f})"
                 for k in r["key"] if k in v]
        mark = "" if v["counts_for_verdict"] else "  [info]"
        print(f"  {v['variant'][:46]:<46} {'  '.join(cells)}  n={v.get('nobs', '')}{mark}")
    for k, j in (r["jackknife"] or {}).items():
        print(f"  jackknife ({j['unit']}) {k}: [{j['min']:.4f} (drop {j['min_when_dropping']}), "
              f"{j['max']:.4f} (drop {j['max_when_dropping']})], same sign {j['share_same_sign']:.0%}")
    for k, v in r["verdict"].items():
        print(f"  VERDICT {k}: {v['status']}"
              + ("" if v["baseline_significant"] else " (baseline not significant at 5%)")
              + (" | WEAK INSTRUMENTS: rely on the robust set" if v.get("weak_instruments") else "")
              + (f" | weak-IV-robust set {v['weak_iv_robust_set']}" if "weak_iv_robust_set" in v else "")
              + (f" | loses significance: {v['loses_significance']}" if v["loses_significance"] else "")
              + (f" | moves >50%: {v['moves_more_than_50pct']}" if v["moves_more_than_50pct"] else "")
              + (f" | sign flips: {v['sign_flips']}" if v["sign_flips"] else ""))
        if v["heterogeneity_moves_more_than_30pct"]:
            print(f"  HETEROGENEITY {k}: {v['heterogeneity_moves_more_than_30pct']}")


def main(argv: list[str]) -> None:
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    cfg = load_config(argv[0])
    df, log = load_data(cfg)
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    df = df.reset_index(drop=True)
    results = {"structure": "iv", "sample_log": log, "models": {}}
    for name, m in cfg["models"].items():
        r = run_model(name, m, df, cfg, jackknife="--no-jackknife" not in argv)
        results["models"][name] = r
        print_model(name, r)
    out = Path(cfg["output_dir"]) / "iv_diagnostics.json"
    write_json(results, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
