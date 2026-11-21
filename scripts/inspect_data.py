"""Step 1 — describe the data structure and flag data problems before any estimation.

Usage:  python inspect_data.py <config.json>
Writes: <output_dir>/inspection.json and prints a short summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import design, load_config, load_data, structure_of, write_json


def panel_shape(df: pd.DataFrame) -> dict:
    ent = df.index.get_level_values(0)
    tim = df.index.get_level_values(1)
    t_i = pd.Series(1, index=df.index).groupby(level=0).size()
    times = np.sort(tim.unique())
    n_times = len(times)
    balanced = bool((t_i == n_times).all())
    # gaps: an entity observed in t and t+2 but not t+1
    gaps = 0
    if np.issubdtype(times.dtype, np.number) and n_times > 1:
        step = np.min(np.diff(times))
        span = pd.Series(tim, index=ent).groupby(level=0).agg(["min", "max", "size"])
        expected = (span["max"] - span["min"]) / step + 1
        gaps = int((expected > span["size"]).sum())
    return {
        "n_entities": int(ent.nunique()),
        "n_periods": int(n_times),
        "first_period": times[0].item() if hasattr(times[0], "item") else times[0],
        "last_period": times[-1].item() if hasattr(times[-1], "item") else times[-1],
        "balanced": balanced,
        "T_i_min": int(t_i.min()),
        "T_i_median": float(t_i.median()),
        "T_i_max": int(t_i.max()),
        "entities_with_gaps": int(gaps),
        "regime": ("large N, small T (micro panel)" if ent.nunique() > 3 * n_times
                   else "large T, small N (macro panel)" if n_times > 3 * ent.nunique()
                   else "N and T of similar size"),
    }


def variable_profile(s: pd.Series) -> dict:
    s = s.astype(float)
    x = s.dropna()
    if len(x) == 0:
        return {"n": 0}
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    z = (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else x * 0
    far = (x < q1 - 5 * iqr) | (x > q3 + 5 * iqr) if iqr > 0 else z.abs() > 5
    top = x[z.abs().sort_values(ascending=False).index[:5]]
    return {
        "n": int(len(x)),
        "missing": int(s.isna().sum()),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "min": float(x.min()),
        "p1": float(np.percentile(x, 1)),
        "median": float(x.median()),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "skewness": float(x.skew()),
        "excess_kurtosis": float(x.kurt()),
        "n_extreme_5iqr": int(far.sum()),
        "most_extreme": [{"index": str(i), "value": float(v), "z": float(z[i])}
                         for i, v in top.items()],
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    df, log = load_data(cfg)
    struct = structure_of(cfg)
    out: dict = {"structure": struct, "sample_log": log, "models": {}}

    if struct == "panel":
        out["panel"] = panel_shape(df)

    variables: set[str] = set()
    for name, m in cfg["models"].items():
        y, X = design(m["formula"], df)
        cols = [y.name] + [c for c in X.columns if c != "Intercept"]
        variables.update(c for c in cols if c in df.columns)
        info = {"formula": m["formula"], "n_used": int(len(y)),
                "n_dropped_missing": int(len(df) - len(y))}
        if struct == "panel":
            info["panel_used"] = panel_shape(df.loc[y.index])
            # within variation of each regressor: share of variance left after entity demeaning
            Xr = X.drop(columns=[c for c in X.columns if c == "Intercept"])
            within = Xr - Xr.groupby(level=0).transform("mean")
            info["within_share_of_variance"] = {
                c: float(within[c].var() / Xr[c].var()) if Xr[c].var() > 0 else 0.0
                for c in Xr.columns
            }
        out["models"][name] = info

    out["variables"] = {v: variable_profile(df[v]) for v in sorted(variables)}

    write_json(out, Path(cfg["output_dir"]) / "inspection.json")

    # --- console summary
    print(f"structure: {struct}")
    print(f"sample: {log}")
    if struct == "panel":
        p = out["panel"]
        print(f"panel: N={p['n_entities']}, T={p['n_periods']} "
              f"({p['first_period']}-{p['last_period']}), balanced={p['balanced']}, "
              f"T_i in [{p['T_i_min']}, {p['T_i_max']}], regime: {p['regime']}")
    for name, info in out["models"].items():
        print(f"model {name}: n_used={info['n_used']} (dropped {info['n_dropped_missing']} NA)")
        if "within_share_of_variance" in info:
            low = {k: round(v, 3) for k, v in info["within_share_of_variance"].items() if v < 0.1}
            if low:
                print(f"  WARNING low within variation (FE will be imprecise): {low}")
    for v, prof in out["variables"].items():
        if prof.get("n_extreme_5iqr", 0):
            print(f"variable {v}: {prof['n_extreme_5iqr']} obs beyond 5*IQR; "
                  f"kurtosis={prof['excess_kurtosis']:.1f}")
    print(f"-> {Path(cfg['output_dir']) / 'inspection.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
