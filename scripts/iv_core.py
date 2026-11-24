"""Numerical core for IV/2SLS and shift-share diagnostics (numpy only, no I/O).

Kept separate from iv_diagnostics.py so every formula can be validated on its own
(tests/validate_iv.py). Notation: n observations, W exogenous controls (incl. constant),
X endogenous regressors (K1 columns), Z excluded instruments (L columns), w regression
weights (ones when unweighted). A tilde means "residualized on W with weights w".

Covariance conventions follow linearmodels, so numbers match the estimation output:
  unadjusted  s2 = sum(w u^2) / n, times n / (n - k) when debiased
  robust      HC0 sandwich, times n / (n - k) when debiased
  cluster     sum over clusters of score sums, times G (n - 1) / ((G - 1)(n - k)) when debiased
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy import stats

# ---------------------------------------------------------------- formula and data

_BRACKET = re.compile(r"\[([^\]~]+)~([^\]]+)\]")


def parse_iv_formula(formula: str) -> tuple[str, str, str]:
    """'y ~ 1 + w + [x ~ z1 + z2]' -> ('y ~ 1 + w', 'x', 'z1 + z2')."""
    m = _BRACKET.search(formula)
    if not m:
        raise ValueError("IV formula needs a '[endogenous ~ instruments]' block")
    base = formula[: m.start()] + formula[m.end():]
    base = re.sub(r"\+\s*(\+|$)", r"\1", base.strip()).strip()
    base = re.sub(r"~\s*\+", "~ ", base)
    if base.endswith("~"):
        base += " 1"
    return base, m.group(1).strip(), m.group(2).strip()


def iv_design(formula: str, df: pd.DataFrame):
    """(y, W, X, Z) as DataFrames on the rows complete for every part of the formula."""
    base, endog, instr = parse_iv_formula(formula)
    y, W = patsy.dmatrices(base, df, return_type="dataframe", NA_action="drop")
    X = patsy.dmatrix("0 + " + endog, df, return_type="dataframe", NA_action="drop")
    Z = patsy.dmatrix("0 + " + instr, df, return_type="dataframe", NA_action="drop")
    keep = y.index[y.index.isin(X.index) & y.index.isin(Z.index)]
    return y.iloc[:, 0].loc[keep], W.loc[keep], X.loc[keep], Z.loc[keep]


# ---------------------------------------------------------------- covariance


@dataclass
class CovSpec:
    kind: str = "robust"            # unadjusted | robust | cluster
    groups: np.ndarray | None = None
    debiased: bool = True

    def subset(self, mask: np.ndarray) -> "CovSpec":
        g = None if self.groups is None else np.asarray(self.groups)[mask]
        return CovSpec(self.kind, g, self.debiased)

    @property
    def n_groups(self) -> int | None:
        return None if self.groups is None else int(len(np.unique(self.groups)))


def _cluster_sum(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(np.asarray(groups))
    out = np.zeros((codes.max() + 1,) + a.shape[1:])
    np.add.at(out, codes, a)
    return out


def _scale(spec: CovSpec, n: int, k: int) -> float:
    if not spec.debiased:
        return 1.0
    if spec.kind == "cluster":
        G = spec.n_groups
        return G * (n - 1) / ((G - 1) * (n - k))
    return n / (n - k)


def sandwich(Ainv, H, u, w, spec: CovSpec, k: int) -> np.ndarray:
    """Covariance of b = Ainv H' diag(w) y-type estimators with residual u."""
    n = len(u)
    if spec.kind == "unadjusted":
        s2 = float((w * u * u).sum()) / n * _scale(spec, n, k)
        return s2 * Ainv @ (H.T @ (w[:, None] * H)) @ Ainv.T
    sc = (w * u)[:, None] * H
    if spec.kind == "cluster":
        sc = _cluster_sum(sc, spec.groups)
    elif spec.kind != "robust":
        raise ValueError(f"unknown covariance kind {spec.kind}")
    return _scale(spec, n, k) * Ainv @ (sc.T @ sc) @ Ainv.T


def dof(spec: CovSpec, n: int, k: int) -> float | None:
    """Denominator degrees of freedom for F/t reference distributions (None -> asymptotic)."""
    if not spec.debiased:
        return None
    return spec.n_groups - 1 if spec.kind == "cluster" else n - k


# ---------------------------------------------------------------- linear algebra


def _w(w, n):
    return np.ones(n) if w is None else np.asarray(w, dtype=float)


def wls_coef(y, X, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], (y.T * sw).T, rcond=None)[0]


def residualize(A, W, w):
    """Weighted residuals of the columns of A (n or n x m) on W."""
    A = np.asarray(A, dtype=float)
    if W is None or W.shape[1] == 0:
        return A.copy()
    return A - W @ wls_coef(A, W, w)


def tsls_fit(y, R, Q, w=None, spec: CovSpec | None = None):
    """2SLS of y on regressors R using instruments Q (both include the exogenous W).

    OLS is the special case Q = R. Returns (params, cov, residuals)."""
    y, R, Q = (np.asarray(a, dtype=float) for a in (y, R, Q))
    w = _w(w, len(y))
    spec = spec or CovSpec()
    Rhat = Q @ wls_coef(R, Q, w)
    # pinv: a regressor that is all zero in a subsample (a dummy for a dropped cluster in the
    # jackknife) gets coefficient 0 instead of breaking the fit; identical when full rank
    A = Rhat.T @ (w[:, None] * R)
    Ainv = np.linalg.inv(A) if np.linalg.matrix_rank(A) == A.shape[0] else np.linalg.pinv(A)
    b = Ainv @ (Rhat.T @ (w * y))
    u = y - R @ b
    return b, sandwich(Ainv, Rhat, u, w, spec, R.shape[1]), u


def wald(b, V, idx) -> float:
    b, V = np.asarray(b)[idx], np.asarray(V)[np.ix_(idx, idx)]
    return float(b @ np.linalg.pinv(V) @ b)


def f_pvalue(stat_f: float, q: int, df2: float | None) -> float:
    """p-value of an F-type statistic (Wald / q); chi2(q)/q when df2 is None."""
    if df2 is None:
        return float(stats.chi2.sf(stat_f * q, q))
    return float(stats.f.sf(stat_f, q, df2))


# ---------------------------------------------------------------- first stage


def first_stage(W, X, Z, w=None, spec: CovSpec | None = None) -> dict:
    """Per endogenous regressor: coefficients, partial R2, robust F, effective F, CD F.

    effective_F (Montiel Olea and Pflueger 2013):  pi' (Z~'Z~) pi / tr(V_pi Z~'Z~),
    with V_pi the covariance of the instrument coefficients under `spec`. It equals the
    non-robust F under homoskedasticity and the robust Wald F when L = 1.
    """
    W, X, Z = (np.asarray(a, dtype=float) for a in (W, X, Z))
    n, L, kw = len(X), Z.shape[1], W.shape[1]
    w = _w(w, n)
    spec = spec or CovSpec()
    WZ = np.hstack([W, Z])
    idx = list(range(kw, kw + L))
    Zt = residualize(Z, W, w)
    QZZ = Zt.T @ (w[:, None] * Zt)
    out = []
    for j in range(X.shape[1]):
        b, V, u = tsls_fit(X[:, j], WZ, WZ, w, spec)
        _, V0, _ = tsls_fit(X[:, j], WZ, WZ, w, CovSpec("unadjusted", None, spec.debiased))
        pi, Vpi = b[idx], V[np.ix_(idx, idx)]
        xt = residualize(X[:, j], W, w)
        partial_r2 = 1 - float((w * u * u).sum()) / float((w * xt * xt).sum())
        d2 = dof(spec, n, kw + L)
        robust_F = wald(b, V, idx) / L
        out.append({
            "coef": pi.tolist(), "se": np.sqrt(np.diag(Vpi)).tolist(),
            "partial_r2": partial_r2,
            "robust_F": robust_F, "robust_F_pvalue": f_pvalue(robust_F, L, d2),
            "effective_F": float(pi @ QZZ @ pi / np.trace(Vpi @ QZZ)),
            "cragg_donald_F": wald(b, V0, idx) / L,
            "_resid": u,
        })
    return {"L": L, "K1": X.shape[1], "by_endog": out}


def sanderson_windmeijer(W, X, Z, w=None, spec: CovSpec | None = None) -> list[dict]:
    """Conditional first-stage F for each endogenous regressor when K1 > 1."""
    W, X, Z = (np.asarray(a, dtype=float) for a in (W, X, Z))
    n, L, kw, K1 = len(X), Z.shape[1], W.shape[1], X.shape[1]
    w = _w(w, n)
    spec = spec or CovSpec()
    WZ = np.hstack([W, Z])
    idx = list(range(kw, kw + L))
    out = []
    for j in range(K1):
        others = np.delete(X, j, axis=1)
        b, _, _ = tsls_fit(X[:, j], np.hstack([W, others]), WZ, w, spec)
        e = X[:, j] - np.hstack([W, others]) @ b
        b2, V2, _ = tsls_fit(e, WZ, WZ, w, spec)
        q = L - K1 + 1
        F = wald(b2, V2, idx) / q
        out.append({"F": F, "pvalue": f_pvalue(F, q, dof(spec, n, kw + L))})
    return out


# ---------------------------------------------------------------- weak-IV-robust sets


def quadratic_set(A: float, B: float, C: float) -> dict:
    """Solve A b^2 + B b + C <= 0. Returns the set as a list of intervals (inf allowed)."""
    tol = 1e-12 * max(1.0, abs(A), abs(B), abs(C))
    if abs(A) <= tol:
        if abs(B) <= tol:
            iv = [[-math.inf, math.inf]] if C <= 0 else []
        else:
            r = -C / B
            iv = [[-math.inf, r]] if B > 0 else [[r, math.inf]]
    else:
        disc = B * B - 4 * A * C
        if disc < 0:
            iv = [] if A > 0 else [[-math.inf, math.inf]]
        else:
            r1, r2 = sorted([(-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A)])
            iv = [[r1, r2]] if A > 0 else [[-math.inf, r1], [r2, math.inf]]
    bounded = bool(iv) and all(math.isfinite(a) and math.isfinite(b) for a, b in iv)
    return {"intervals": iv, "bounded": bounded, "empty": not iv,
            "contains_zero": any(a <= 0 <= b for a, b in iv)}


def _ar_pieces(W, x, Z, y, w, spec):
    """Cross products that make the AR statistic a ratio of quadratics in beta0 (L = 1)."""
    yt, xt, zt = (residualize(a, W, w) for a in (y, x, Z[:, 0]))
    zz = float((w * zt * zt).sum())
    a, b = float((w * zt * yt).sum()), float((w * zt * xt).sum())
    uy, ux = yt - zt * a / zz, xt - zt * b / zz
    n, k = len(y), W.shape[1] + 1
    if spec.kind == "unadjusted":
        q = lambda p, r: zz * float((w * p * r).sum()) / n
    elif spec.kind == "robust":
        q = lambda p, r: float((w * w * zt * zt * p * r).sum())
    else:
        def q(p, r):
            sp = _cluster_sum(w * zt * p, spec.groups)
            sr = _cluster_sum(w * zt * r, spec.groups)
            return float((sp * sr).sum())
    s = _scale(spec, n, k)
    return a, b, s * q(uy, uy), s * q(ux, uy), s * q(ux, ux), dof(spec, n, k)


def anderson_rubin(W, x, Z, y, w=None, spec: CovSpec | None = None, level: float = 0.95,
                   grid_halfwidth: float | None = None, center: float = 0.0) -> dict:
    """Anderson-Rubin test of beta = 0 and confidence set (one endogenous regressor).

    AR(b0): regress y - b0 x on W and Z and test that the Z coefficients are zero with the
    covariance in `spec`. L = 1: exact set from a quadratic inequality (may be unbounded).
    L > 1: grid inversion; `touches_grid_edge` means the set may be unbounded.
    """
    W, Z = np.asarray(W, float), np.asarray(Z, float)
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    n, L, kw = len(y), Z.shape[1], W.shape[1]
    w = _w(w, n)
    spec = spec or CovSpec()
    if L == 1:
        a, b, qyy, qxy, qxx, d2 = _ar_pieces(W, x, Z, y, w, spec)
        crit = stats.f.ppf(level, 1, d2) if d2 else stats.chi2.ppf(level, 1)
        res = quadratic_set(b * b - crit * qxx, -2 * a * b + 2 * crit * qxy, a * a - crit * qyy)
        stat0 = a * a / qyy
        res.update({"method": "analytic (L = 1)", "stat_beta0": stat0,
                    "pvalue_beta0": f_pvalue(stat0, 1, d2), "critical_value": float(crit)})
        return res
    # L > 1: grid
    WZ = np.hstack([W, Z])
    idx = list(range(kw, kw + L))
    d2 = dof(spec, n, kw + L)
    crit = stats.f.ppf(level, L, d2) if d2 else stats.chi2.ppf(level, L) / L

    def F(b0):
        bb, V, _ = tsls_fit(y - b0 * x, WZ, WZ, w, spec)
        return wald(bb, V, idx) / L
    hw = grid_halfwidth or 10.0
    grid = np.linspace(center - hw, center + hw, 801)
    acc = np.array([F(g) <= crit for g in grid])
    iv, start = [], None
    for g, ok in zip(grid, acc):
        if ok and start is None:
            start = g
        if not ok and start is not None:
            iv.append([start, prev]); start = None
        prev = g
    if start is not None:
        iv.append([start, grid[-1]])
    stat0 = F(0.0)
    return {"intervals": [[float(a), float(b)] for a, b in iv], "bounded": bool(iv) and not (acc[0] or acc[-1]),
            "empty": not iv, "contains_zero": F(0.0) <= crit,
            "touches_grid_edge": bool(acc[0] or acc[-1]), "method": f"grid (L = {L}, 801 points)",
            "stat_beta0": stat0, "pvalue_beta0": f_pvalue(stat0, L, d2), "critical_value": float(crit)}


# ---------------------------------------------------------------- endogeneity


def control_function(W, X, Z, y, w=None, spec: CovSpec | None = None) -> dict:
    """Regression-based (Durbin-Wu-Hausman) test: add first-stage residuals to the
    structural equation and test them jointly with the covariance in `spec`.
    Under `unadjusted` it equals the Wu-Hausman F."""
    W, X, Z = (np.asarray(a, float) for a in (W, X, Z))
    y = np.asarray(y, float).ravel()
    n, kw, K1 = len(y), W.shape[1], X.shape[1]
    w = _w(w, n)
    spec = spec or CovSpec()
    WZ = np.hstack([W, Z])
    V_hat = np.column_stack([tsls_fit(X[:, j], WZ, WZ, w, spec)[2] for j in range(K1)])
    R = np.hstack([W, X, V_hat])
    b, V, _ = tsls_fit(y, R, R, w, spec)
    idx = list(range(kw + K1, kw + 2 * K1))
    F = wald(b, V, idx) / K1
    return {"F": F, "pvalue": f_pvalue(F, K1, dof(spec, n, R.shape[1])), "df": K1,
            "coef_first_stage_resid": b[idx].tolist()}


# ---------------------------------------------------------------- shift-share


def akm(W, x, z, y, S, g, w=None, shock_clusters=None, level: float = 0.95) -> dict:
    """Adao, Kolesar and Morales (2019) inference for a just-identified shift-share IV.

    z = S g is the instrument, S the n x K exposure (shares) matrix and g the K shocks.
      beta     = sum w z~ y~ / sum w z~ x~                       (2SLS)
      R_k      = sum_i w_i s_ik e_i, e the structural residual
      Xhat     = weighted regression of z~ on S (no constant): the shocks net of controls
      V_AKM    = sum_c (sum_{k in c} Xhat_k R_k)^2 / (sum w z~ x~)^2
    AKM0 (null-imposed) confidence set: invert t(b0)^2 <= z_crit^2 using residuals of
    y - b0 x on the controls; it is robust to weak instruments as well.
    The first-stage F uses the same variance with first-stage residuals.
    """
    W, S = np.asarray(W, float), np.asarray(S, float)
    x, z, y, g = (np.asarray(a, float).ravel() for a in (x, z, y, g))
    n = len(y)
    w = _w(w, n)
    yt, xt, zt = (residualize(a, W, w) for a in (y, x, z))
    a, b = float((w * zt * yt).sum()), float((w * zt * xt).sum())
    zz = float((w * zt * zt).sum())
    beta = a / b
    e = yt - beta * xt
    Xhat = wls_coef(zt, S, w)
    rank = int(np.linalg.matrix_rank(S))
    cl = np.arange(len(g)) if shock_clusters is None else np.asarray(shock_clusters)

    def q(r1, r2):
        return float((_cluster_sum(Xhat * r1, cl) * _cluster_sum(Xhat * r2, cl)).sum())

    R_e = S.T @ (w * e)
    se = math.sqrt(q(R_e, R_e)) / abs(b)
    pi = b / zz
    R_v = S.T @ (w * (xt - pi * zt))
    se_pi = math.sqrt(q(R_v, R_v)) / zz
    ry, rx = S.T @ (w * yt), S.T @ (w * xt)
    qyy, qxy, qxx = q(ry, ry), q(rx, ry), q(rx, rx)
    crit = stats.norm.ppf(0.5 + level / 2) ** 2
    ci0 = quadratic_set(b * b - crit * qxx, -2 * a * b + 2 * crit * qxy, a * a - crit * qyy)
    t0 = a * a / qyy
    zcrit = math.sqrt(crit)
    return {
        "beta": beta, "se": se, "pvalue": float(2 * stats.norm.sf(abs(beta / se))),
        "ci": [beta - zcrit * se, beta + zcrit * se],
        "akm0": {**ci0, "stat_beta0": t0, "pvalue_beta0": float(stats.chi2.sf(t0, 1))},
        "first_stage_coef": pi, "first_stage_se": se_pi, "first_stage_F": (pi / se_pi) ** 2,
        "n_shocks": int(len(g)), "n_shock_clusters": int(len(np.unique(cl))),
        "exposure_rank": rank,
        # how well the shares reproduce the residualized instrument, and shocks per
        # observation: with many shocks per observation Xhat is noisy (see interpretation.md)
        "xhat_fit_r2": float(1 - (w * (zt - S @ Xhat) ** 2).sum() / zz),
        "shocks_per_obs": float(len(g) / n),
        "corr_xhat_shock": float(np.corrcoef(Xhat, g)[0, 1]),
        "shock_mean": float(g.mean()), "shock_sd": float(g.std()),
        "note": ("exposure matrix rank < number of shocks: minimum-norm shock estimates used"
                 if rank < S.shape[1] else None),
    }


def rotemberg(W, x, y, S, g, w=None, names=None, top: int = 5) -> dict:
    """Rotemberg weights of Goldsmith-Pinkham, Sorkin and Swift (2020).

    alpha_k = g_k s_k' w x~ / sum_j g_j s_j' w x~ and beta_k = s_k' w y~ / s_k' w x~
    (the just-identified IV that uses only sector k's shares), so beta = sum alpha_k beta_k.
    """
    W, S = np.asarray(W, float), np.asarray(S, float)
    x, y, g = (np.asarray(a, float).ravel() for a in (x, y, g))
    w = _w(w, len(y))
    yt, xt = residualize(y, W, w), residualize(x, W, w)
    sx, sy = S.T @ (w * xt), S.T @ (w * yt)
    num = g * sx
    alpha = num / num.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_k = np.where(sx != 0, sy / sx, np.nan)
    names = list(names) if names is not None else [str(i) for i in range(len(g))]
    order = np.argsort(-np.abs(alpha))
    ok = np.isfinite(beta_k)
    return {
        "sum_alpha": float(alpha.sum()),
        "implied_beta": float(np.nansum(alpha[ok] * beta_k[ok])),
        "sum_negative_alpha": float(alpha[alpha < 0].sum()),
        "share_negative_alpha": float((alpha < 0).mean()),
        "top5_abs_alpha_share": float(np.abs(alpha[order[:top]]).sum() / np.abs(alpha).sum()),
        "top": [{"shock": names[i], "alpha": float(alpha[i]), "g": float(g[i]),
                 "beta_k": None if not ok[i] else float(beta_k[i])} for i in order[:top]],
        "_alpha": alpha, "_beta_k": beta_k,
    }


def shock_summary(S, g, w=None, shock_clusters=None) -> dict:
    """Exposure-weighted shock summary in the spirit of Borusyak, Hull and Jaravel (2022)."""
    S = np.asarray(S, float)
    w = _w(w, S.shape[0])
    s_bar = (w @ S) / w.sum()
    p = s_bar / s_bar.sum()
    out = {"effective_n_shocks": float(1 / (p ** 2).sum()), "largest_exposure_share": float(p.max()),
           "shock_mean_weighted": float(p @ np.asarray(g)),
           "shock_sd_weighted": float(np.sqrt(p @ (np.asarray(g) - p @ np.asarray(g)) ** 2))}
    if shock_clusters is not None:
        pc = _cluster_sum(p, np.asarray(shock_clusters))
        out["effective_n_shock_clusters"] = float(1 / (pc ** 2).sum())
    return out


def share_sums(S) -> dict:
    e = np.asarray(S, float).sum(axis=1)
    return {"min": float(e.min()), "median": float(np.median(e)), "max": float(e.max()),
            "complete": bool(np.all(np.abs(e - 1) < 1e-3)), "_e": e}


def is_spanned(v, W, w=None, tol: float = 1e-6) -> bool:
    """True when v lies in the column space of W (e.g. the sum of shares is a control)."""
    v = np.asarray(v, float)
    w = _w(w, len(v))
    r = residualize(v, W, w)
    return float((w * r * r).sum()) <= tol * max(float((w * (v - v.mean()) ** 2).sum()), 1e-12)


def shock_jackknife(W, x, z, y, S, g, w=None, groups=None, names=None) -> dict:
    """Leave one shock (or shock cluster) out: the instrument drops sum_{k in c} s_k g_k.

    Closed form, because y~ and x~ are orthogonal to W:
      beta_(-c) = (a - sum_{k in c} g_k s_k'w y~) / (b - sum_{k in c} g_k s_k'w x~)
    Controls are held fixed; if the sum of shares is a control, refit instead (see caller).
    """
    W, S = np.asarray(W, float), np.asarray(S, float)
    x, z, y, g = (np.asarray(a, float).ravel() for a in (x, z, y, g))
    w = _w(w, len(y))
    yt, xt, zt = (residualize(a, W, w) for a in (y, x, z))
    St = residualize(S, W, w)
    a, b, zz = (float((w * zt * v).sum()) for v in (yt, xt, zt))
    groups = np.arange(len(g)) if groups is None else np.asarray(groups)
    codes, uniq = pd.factorize(groups)
    gy = _cluster_sum(g * (S.T @ (w * yt)), codes)
    gx = _cluster_sum(g * (S.T @ (w * xt)), codes)
    beta = (a - gy) / (b - gx)
    # first-stage coefficient: (b - gx) / ||z~ - St_c g_c||^2_w
    zc = np.zeros((len(zt), len(uniq)))
    np.add.at(zc.T, codes, (St * g).T)
    zz_c = zz - 2 * (w * zt) @ zc + (w[:, None] * zc * zc).sum(axis=0)
    pi = (b - gx) / zz_c
    return {"labels": [str(u) for u in uniq], "beta": beta, "pi": pi,
            "beta_full": a / b, "pi_full": b / zz}
