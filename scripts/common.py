"""Shared helpers for the econometric-diagnostics scripts.

Every script reads the same JSON config (see references/config_schema.md), loads the data
with the same filters, and writes machine-readable JSON next to human-readable output.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy

# ---------------------------------------------------------------- config


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_config_dir"] = str(path.parent)
    for key in ("data", "models"):
        if key not in cfg:
            sys.exit(f"config error: missing required key '{key}'")
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = path.parent / data_path
    cfg["data"] = str(data_path.resolve())
    out = Path(cfg.get("output_dir", path.parent))
    if not out.is_absolute():
        out = path.parent / out
    cfg["output_dir"] = str(out.resolve())
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    for name, m in cfg["models"].items():
        if "formula" not in m:
            sys.exit(f"config error: model '{name}' has no 'formula'")
        m.setdefault("effects", "twoway" if cfg.get("entity") else "none")
        m.setdefault("cov", "clustered_entity" if cfg.get("entity")
                     else "HAC" if cfg.get("time") else "HC3")
    return cfg


# ---------------------------------------------------------------- data


def read_any(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".dta",):
        return pd.read_stata(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    sys.exit(f"unsupported data format: {suffix}")


def load_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Load data and apply the config's sample filters. Returns (df, log)."""
    df = read_any(cfg["data"])
    log = {"rows_raw": int(len(df))}
    ent, tim = cfg.get("entity"), cfg.get("time")

    excl = cfg.get("exclude_entities") or []
    if excl:
        if not ent:
            sys.exit("config error: exclude_entities requires 'entity'")
        before = len(df)
        df = df[~df[ent].isin(excl)]
        log["dropped_excluded_entities"] = int(before - len(df))

    if cfg.get("query"):
        before = len(df)
        df = df.query(cfg["query"])
        log["dropped_by_query"] = int(before - len(df))

    num = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[num].to_numpy(dtype=float)).sum())
    if n_inf:
        df = df.copy()
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    log["inf_values_set_to_nan"] = n_inf

    if ent and tim:
        df = df.set_index([ent, tim]).sort_index()
    elif tim:
        df = df.set_index(tim).sort_index()
    log["rows_after_filters"] = int(len(df))
    return df, log


def structure_of(cfg: dict) -> str:
    if cfg.get("entity") and cfg.get("time"):
        return "panel"
    if cfg.get("time"):
        return "time_series"
    return "cross_section"


# ---------------------------------------------------------------- design matrices


def design(formula: str, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """y and X (with Intercept column when the formula has one), rows with NA dropped."""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe", NA_action="drop")
    return y.iloc[:, 0], X


def drop_intercept(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[c for c in X.columns if c == "Intercept"])


def regressor_names(formula: str, df: pd.DataFrame) -> list[str]:
    _, X = design(formula, df.head(50))
    return [c for c in X.columns if c != "Intercept"]


# ---------------------------------------------------------------- output helpers


def clean(obj):
    """Make numpy / pandas objects JSON serialisable; NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Series,)):
        return clean(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def test_result(name: str, h0: str, stat, pvalue, df=None, alpha: float = 0.05,
                note: str | None = None, **extra) -> dict:
    """Uniform record for a hypothesis test."""
    rec = {
        "test": name,
        "h0": h0,
        "statistic": stat,
        "df": df,
        "pvalue": pvalue,
        "reject_h0": None if pvalue is None or (isinstance(pvalue, float) and math.isnan(pvalue))
        else bool(pvalue < alpha),
    }
    if note:
        rec["note"] = note
    rec.update(extra)
    return rec


def failed_test(name: str, h0: str, err: Exception) -> dict:
    return {"test": name, "h0": h0, "statistic": None, "pvalue": None, "reject_h0": None,
            "error": f"{type(err).__name__}: {err}"}


def stars(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_p(p) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
