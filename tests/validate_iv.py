"""Validation of the IV / shift-share code (no pytest needed).

Usage:  python tests/validate_iv.py          # replication + identities (a few seconds)
        python tests/validate_iv.py --mc 500 # also a Monte Carlo of shift-share inference

1. Published numbers (Wooldridge, Introductory Econometrics, ch. 15; data bundled with
   linearmodels): Mroz (1987) return to education and Card (1995) college proximity.
2. Cross-checks against linearmodels for every covariance type, with and without weights.
3. Exact identities: F_eff = robust F when L = 1 and = Cragg-Donald F under iid errors;
   control-function test = Wu-Hausman F under iid errors; AR(0) = reduced-form F;
   the AR and AKM0 statistics equal the critical value at the set's endpoints;
   AKM with one sector per region = heteroskedasticity-robust (HC0) SE;
   Rotemberg weights sum to one and reproduce the 2SLS estimate;
   closed-form shock jackknife = brute-force refit.
4. (--mc) Rejection rates at the true value for robust, clustered, AR, AKM and AKM0
   inference when regions with similar shares have correlated errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.datasets import card, mroz
from linearmodels.iv import IV2SLS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import iv_core as C  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def close(a, b, tol):
    return abs(a - b) <= tol


def mats(df, y, w, x, z):
    W = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in w])
    return (df[y].to_numpy(float), W, df[x].to_numpy(float)[:, None],
            np.column_stack([df[c].to_numpy(float) for c in z]))


# ---------------------------------------------------------------- 1. published numbers


def published():
    m = mroz.load().dropna(subset=["lwage"])
    r = IV2SLS.from_formula("lwage ~ 1 + [educ ~ fatheduc]", m).fit(cov_type="unadjusted", debiased=True)
    check("Mroz Ex. 15.1: educ 0.059 (0.035)",
          close(r.params["educ"], 0.0592, 5e-4) and close(r.std_errors["educ"], 0.0351, 5e-4),
          f"{r.params['educ']:.4f} ({r.std_errors['educ']:.4f})")
    r = IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ motheduc + fatheduc]", m).fit(
        cov_type="unadjusted", debiased=True)
    check("Mroz Ex. 15.5: educ 0.0614 (0.0314)",
          close(r.params["educ"], 0.0614, 5e-5) and close(r.std_errors["educ"], 0.0314, 5e-5),
          f"{r.params['educ']:.4f} ({r.std_errors['educ']:.4f})")
    check("Mroz Ex. 15.8: Sargan nR2 = 0.378", close(r.sargan.stat, 0.378, 5e-4), f"{r.sargan.stat:.4f}")
    y, W, X, Z = mats(m, "lwage", ["exper", "expersq"], "educ", ["motheduc", "fatheduc"])
    # Ex. 15.7: coefficient on the first-stage residual 0.058 (0.035), t = 1.67
    cf = C.control_function(W, X, Z, y, None, C.CovSpec("unadjusted", None, True))
    t = np.sqrt(cf["F"])
    check("Mroz Ex. 15.7: control-function coef 0.058, t = 1.67",
          close(cf["coef_first_stage_resid"][0], 0.0582, 5e-4) and close(t, 1.67, 0.01),
          f"coef {cf['coef_first_stage_resid'][0]:.4f}, t {t:.3f}")

    c = card.load()
    ctrl = ["exper", "expersq", "black", "smsa", "south", "smsa66"] + [f"reg66{i}" for i in range(2, 10)]
    f = "lwage ~ 1 + " + " + ".join(ctrl) + " + [educ ~ nearc4]"
    r = IV2SLS.from_formula(f, c).fit(cov_type="unadjusted", debiased=True)
    check("Card Ex. 15.4: educ 0.132 (0.055)",
          close(r.params["educ"], 0.132, 1e-3) and close(r.std_errors["educ"], 0.055, 1e-3),
          f"{r.params['educ']:.4f} ({r.std_errors['educ']:.4f})")
    y, W, X, Z = mats(c, "lwage", ctrl, "educ", ["nearc4"])
    fs = C.first_stage(W, X, Z, None, C.CovSpec("unadjusted", None, True))["by_endog"][0]
    check("Card Ex. 15.4: first stage nearc4 0.320 (0.088)",
          close(fs["coef"][0], 0.320, 1e-3) and close(fs["se"][0], 0.088, 1e-3),
          f"{fs['coef'][0]:.4f} ({fs['se'][0]:.4f})")


# ---------------------------------------------------------------- 2. linearmodels


def crosscheck_linearmodels():
    c = card.load()
    ctrl = ["exper", "expersq", "black", "smsa", "south"]
    c = c.dropna(subset=["lwage", "educ", "nearc4", "nearc2", "weight"] + ctrl)
    region = c[[f"reg66{i}" for i in range(1, 10)]].to_numpy().argmax(axis=1)
    groups = region * 2 + c["smsa66"].to_numpy()          # 18 clusters
    y, W, X, Z = mats(c, "lwage", ctrl, "educ", ["nearc4", "nearc2"])
    dfW = pd.DataFrame(W, index=c.index).add_prefix("w")
    for kind in ("unadjusted", "robust", "cluster"):
        for deb in (True, False):
            for weighted in (False, True):
                w = c["weight"].to_numpy(float) if weighted else None
                spec = C.CovSpec(kind, groups if kind == "cluster" else None, deb)
                kw = ({"cov_type": "clustered", "clusters": pd.Series(groups, index=c.index)}
                      if kind == "cluster" else {"cov_type": kind})
                lm = IV2SLS(pd.Series(y, index=c.index), dfW, pd.Series(X[:, 0], index=c.index),
                            pd.DataFrame(Z, index=c.index).add_prefix("z"),
                            weights=None if w is None else pd.Series(w, index=c.index)).fit(
                    **kw, debiased=deb)
                b, V, _ = C.tsls_fit(y, np.hstack([W, X]), np.hstack([W, Z]), w, spec)
                d = max(np.max(np.abs(b - lm.params.to_numpy())),
                        np.max(np.abs(np.sqrt(np.diag(V)) - lm.std_errors.to_numpy())))
                tag = f"{kind}, debiased={deb}, weighted={weighted}"
                check(f"2SLS params and SE = linearmodels ({tag})", d < 1e-9, f"max diff {d:.1e}")
                fsd = lm.first_stage.diagnostics
                fs = C.first_stage(W, X, Z, w, spec)["by_endog"][0]
                # linearmodels reports F under iid errors and the Wald chi2 (= L * F) otherwise
                L = Z.shape[1]
                lm_stat = float(fsd["f.stat"].iloc[0])
                mine = fs["robust_F"] * (L if str(fsd["f.dist"].iloc[0]).startswith("chi2") else 1)
                check(f"first-stage F/Wald = linearmodels ({tag})",
                      close(mine, lm_stat, 1e-6 * lm_stat), f"{mine:.4f} vs {lm_stat:.4f}")
    # Wu-Hausman = control function under iid errors
    lm = IV2SLS(pd.Series(y), pd.DataFrame(W).add_prefix("w"), pd.Series(X[:, 0]), pd.DataFrame(Z).add_prefix("z")).fit(
        cov_type="unadjusted", debiased=True)
    cf = C.control_function(W, X, Z, y, None, C.CovSpec("unadjusted", None, True))
    wh = lm.wu_hausman()
    check("control function (iid) = Wu-Hausman F", close(cf["F"], wh.stat, 1e-8 * wh.stat),
          f"{cf['F']:.6f} vs {wh.stat:.6f}")


# ---------------------------------------------------------------- 3. identities


def identities():
    c = card.load()
    ctrl = ["exper", "expersq", "black", "smsa", "south"]
    y, W, X, Z = mats(c, "lwage", ctrl, "educ", ["nearc4", "nearc2"])
    groups = c[[f"reg66{i}" for i in range(1, 10)]].to_numpy().argmax(axis=1)
    rob = C.CovSpec("robust", None, True)
    one = C.first_stage(W, X, Z[:, :1], None, rob)["by_endog"][0]
    check("F_eff = robust F when L = 1", close(one["effective_F"], one["robust_F"], 1e-9 * one["robust_F"]))
    iid = C.first_stage(W, X, Z, None, C.CovSpec("unadjusted", None, True))["by_endog"][0]
    check("F_eff = Cragg-Donald F under iid errors",
          close(iid["effective_F"], iid["cragg_donald_F"], 1e-9 * iid["cragg_donald_F"]))

    for spec, lab in ((rob, "robust"), (C.CovSpec("cluster", groups, True), "cluster")):
        ar = C.anderson_rubin(W, X[:, 0], Z[:, :1], y, None, spec)
        WZ = np.hstack([W, Z[:, :1]])
        idx = [WZ.shape[1] - 1]

        def F(b0):
            b, V, _ = C.tsls_fit(y - b0 * X[:, 0], WZ, WZ, None, spec)
            return C.wald(b, V, idx)
        ends = [e for iv in ar["intervals"] for e in iv if np.isfinite(e)]
        ok = all(close(F(e), ar["critical_value"], 1e-6 * ar["critical_value"]) for e in ends)
        check(f"AR set endpoints solve AR(b) = critical value ({lab})", ok and len(ends) > 0,
              C.__name__ and str(ar["intervals"]))
        b, V, _ = C.tsls_fit(y, WZ, WZ, None, spec)
        check(f"AR(0) = reduced-form Wald ({lab})", close(ar["stat_beta0"], C.wald(b, V, idx), 1e-8))
    ar2 = C.anderson_rubin(W, X[:, 0], Z, y, None, rob, center=0.13, grid_halfwidth=2.0)
    check("AR grid (L = 2) runs and is bounded", ar2["bounded"], str(ar2["intervals"]))

    # AKM with one sector per region = HC0 robust SE of 2SLS (weighted and unweighted)
    rng = np.random.default_rng(1)
    sub = c.sample(400, random_state=3)
    ys, Ws, Xs, _ = mats(sub, "lwage", ctrl, "educ", ["nearc4"])
    g = rng.normal(size=400)
    S = np.eye(400)
    z = S @ g + 0.3 * Xs[:, 0]          # instrument correlated with educ
    g = g + 0.3 * Xs[:, 0]
    for weighted in (False, True):
        w = sub["weight"].to_numpy(float) if weighted else None
        ak = C.akm(Ws, Xs[:, 0], z, ys, S, g, w)
        _, V, _ = C.tsls_fit(ys, np.hstack([Ws, Xs]), np.hstack([Ws, z[:, None]]), w,
                             C.CovSpec("robust", None, False))
        se = float(np.sqrt(V[-1, -1]))
        check(f"AKM with identity shares = HC0 SE (weighted={weighted})", close(ak["se"], se, 1e-9 * se),
              f"{ak['se']:.6f} vs {se:.6f}")

    # Rotemberg identity, AKM0 endpoints and shock jackknife on the synthetic shift-share data
    ex = Path(__file__).resolve().parents[1] / "examples"
    if not (ex / "ss_regions.csv").exists():
        check("shift-share example data present (run examples/make_iv_example.py)", False)
        return
    reg = pd.read_csv(ex / "ss_regions.csv")
    exp = pd.read_csv(ex / "ss_exposure.csv")
    shk = pd.read_csv(ex / "ss_shocks.csv")
    S = exp.pivot_table(index="region", columns="sector", values="share", fill_value=0.0)
    S = S.reindex(index=reg["region"], columns=shk["sector"], fill_value=0.0).to_numpy()
    g = shk["g"].to_numpy()
    w = reg["pop"].to_numpy(float)
    Wr = np.ones((len(reg), 1))
    xr, yr, zr = reg["x"].to_numpy(), reg["y"].to_numpy(), S @ g
    check("example instrument = S g", np.max(np.abs(zr - reg["z_bartik"])) < 1e-9)
    rot = C.rotemberg(Wr, xr, yr, S, g, w)
    beta = C.akm(Wr, xr, zr, yr, S, g, w)["beta"]
    check("Rotemberg weights sum to 1", close(rot["sum_alpha"], 1.0, 1e-10))
    check("sum alpha_k beta_k = 2SLS beta", close(rot["implied_beta"], beta, 1e-8), f"{rot['implied_beta']:.6f} vs {beta:.6f}")
    for cl in (None, shk["sector_group"].to_numpy()):
        ak = C.akm(Wr, xr, zr, yr, S, g, w, cl)
        ends = [e for iv in ak["akm0"]["intervals"] for e in iv if np.isfinite(e)]
        yt, xt = C.residualize(yr, Wr, w), C.residualize(xr, Wr, w)
        zt = C.residualize(zr, Wr, w)
        Xhat = C.wls_coef(zt, S, w)
        cc = np.arange(len(g)) if cl is None else pd.factorize(cl)[0]

        def t2(b0):
            r = S.T @ (w * (yt - b0 * xt))
            num = float((w * zt * (yt - b0 * xt)).sum())
            den = float((C._cluster_sum(Xhat * r, cc) ** 2).sum())
            return num * num / den
        ok = all(close(t2(e), 1.959964 ** 2, 1e-4) for e in ends)
        check(f"AKM0 endpoints solve t^2 = 1.96^2 (clusters={cl is not None})", ok and len(ends) > 0,
              str(ak["akm0"]["intervals"]))
    jk = C.shock_jackknife(Wr, xr, zr, yr, S, g, w)
    for kk in (0, 7, 33):
        z_k = zr - S[:, kk] * g[kk]
        r = IV2SLS(pd.Series(yr), pd.DataFrame(Wr), pd.Series(xr), pd.Series(z_k),
                   weights=pd.Series(w)).fit()
        check(f"closed-form shock jackknife = refit (drop {kk})",
              close(jk["beta"][kk], float(r.params.iloc[-1]), 1e-9))


# ---------------------------------------------------------------- 4. Monte Carlo


def monte_carlo(reps: int):
    """Shares fixed, shocks and sector-level errors redrawn: the design-based setting of
    Adao, Kolesar and Morales (2019). True beta = 0.8."""
    rng = np.random.default_rng(2024)
    N, K = 600, 80
    state = rng.integers(0, 40, N)
    alpha = rng.gamma(0.3, 1.0, K)
    S = rng.dirichlet(alpha + 0.02, N) * rng.uniform(0.3, 0.9, N)[:, None]
    W = np.column_stack([np.ones(N), S.sum(axis=1)])
    rej = {k: 0 for k in ("robust", "cluster_state", "anderson_rubin", "akm", "akm0")}
    betas = []
    for _ in range(reps):
        g, eta = rng.normal(size=K), rng.normal(size=K)
        eps = 2 * S @ eta + rng.normal(0, 0.5, N)
        z = S @ g
        x = z + 0.5 * eps + rng.normal(0, 0.5, N)
        y = 0.8 * x + eps
        R, Q = np.hstack([W, x[:, None]]), np.hstack([W, z[:, None]])
        for lab, spec in (("robust", C.CovSpec("robust")),
                          ("cluster_state", C.CovSpec("cluster", state))):
            b, V, _ = C.tsls_fit(y, R, Q, None, spec)
            rej[lab] += abs((b[-1] - 0.8) / np.sqrt(V[-1, -1])) > 1.96
        ar = C.anderson_rubin(W, x, z[:, None], y - 0.8 * x, None, C.CovSpec("robust"))
        rej["anderson_rubin"] += ar["pvalue_beta0"] < 0.05
        ak = C.akm(W, x, z, y, S, g)
        rej["akm"] += abs((ak["beta"] - 0.8) / ak["se"]) > 1.96
        ak0 = C.akm(W, x, z, y - 0.8 * x, S, g)
        rej["akm0"] += ak0["akm0"]["pvalue_beta0"] < 0.05
        betas.append(ak["beta"])
    rates = {k: v / reps for k, v in rej.items()}
    print(f"\nMonte Carlo ({reps} reps, nominal 5 %): rejection of the true beta")
    for k, v in rates.items():
        print(f"  {k:<16} {v:.3f}")
    print(f"  mean beta_hat {np.mean(betas):.3f} (true 0.8), median {np.median(betas):.3f}")
    se_mc = np.sqrt(0.05 * 0.95 / reps)
    check("MC: AKM0 rejection within 3 MC s.e. of 5 %", abs(rates["akm0"] - 0.05) < 3 * se_mc + 0.01,
          f"{rates['akm0']:.3f}")
    check("MC: region-level robust SE over-rejects (> 2x nominal)", rates["robust"] > 0.10,
          f"{rates['robust']:.3f}")


def main(argv):
    published()
    crosscheck_linearmodels()
    identities()
    if "--mc" in argv:
        monte_carlo(int(argv[argv.index("--mc") + 1]))
    width = max(len(n) for n, _, _ in RESULTS)
    for n, ok, d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {n:<{width}}  {d}")
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
