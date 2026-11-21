"""Step 4 — LaTeX tables (booktabs) from the JSON written by the other scripts.

Usage:  python to_latex.py <config.json> [--lang en|pt]
Writes: <output_dir>/table_diagnostics.tex and <output_dir>/table_robustness.tex
Needs in the preamble: \\usepackage{booktabs} \\usepackage{graphicx}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import load_config, stars

PT = {
    "Specification and diagnostic tests": "Testes de especificação e diagnóstico",
    "Robustness of the key coefficients": "Robustez dos coeficientes de interesse",
    "Test": "Teste", "Variant": "Variante",
    "F test for entity effects": "Teste F de efeitos de unidade",
    "F test for time effects": "Teste F de efeitos de tempo",
    "Wald test for time effects (clustered)": "Wald de efeitos de tempo (cluster)",
    "Breusch-Pagan LM (random effects)": "LM de Breusch-Pagan (efeitos aleatórios)",
    "Hausman (classic)": "Hausman (clássico)",
    "Mundlak / correlated random effects (clustered)": "Mundlak / CRE (cluster)",
    "Wooldridge test for AR(1)": "Wooldridge, AR(1)",
    "Modified Wald (groupwise heteroskedasticity)": "Wald modificado (heteroc. entre grupos)",
    "Pesaran CD": "CD de Pesaran",
    "Baseline": "Base", "A. Inference": "A. Inferência", "B. Estimator": "B. Estimador",
    "C. Sample": "C. Amostra", "Observations": "Observações",
    "Jackknife (leave one entity out): range": "Jackknife (exclui uma unidade): intervalo",
    "notes_diag": ("Estatística de teste e p-valor entre parênteses. Raiz unitária: Z de Choi "
                   "(Fisher-ADF). Testes em resíduos de efeitos fixos quando aplicável."),
    "notes_rob": ("Coeficiente e erro-padrão entre parênteses. * p<0,10; ** p<0,05; *** p<0,01. "
                  "Cada linha altera um único elemento em relação à especificação de base."),
}
EN = {
    "notes_diag": ("Test statistic with p-value in parentheses. Unit root: Choi Z (Fisher-ADF). "
                   "Residual-based tests use fixed-effects residuals where applicable."),
    "notes_rob": ("Coefficient with standard error in parentheses. * p<0.10; ** p<0.05; "
                  "*** p<0.01. Each row changes a single element relative to the baseline."),
}


def tr(s: str, lang: str) -> str:
    if lang == "pt":
        if s in PT:
            return PT[s]
        for a, b in [("Fisher-ADF panel unit root: ", "Raiz unitária (Fisher-ADF): "),
                     ("ADF/KPSS integration order: ", "Ordem de integração: "),
                     ("SE: ", "EP: "), ("clustered by entity", "cluster por unidade"),
                     ("clustered by time", "cluster por tempo"), ("two-way clustered",
                     "cluster em duas vias"), ("unadjusted (iid)", "não ajustado (iid)"),
                     ("heteroskedasticity-robust", "robusto à heterocedasticidade"),
                     ("Pooled OLS", "MQO agrupado"), ("Random effects", "Efeitos aleatórios"),
                     ("FE entity only", "EF só de unidade"), ("FE two-way", "EF de dois sentidos"),
                     ("First differences", "Primeiras diferenças"),
                     ("+ time dummies", "+ dummies de tempo"),
                     ("Between (entity means)", "Between (médias por unidade)"),
                     ("Dynamic FE (+ lagged y)", "EF dinâmico (+ y defasado)"),
                     ("Winsorized", "Winsorizado"), ("Trimmed |residual| > 3 sd",
                     "Exclui |resíduo| > 3 dp"), ("Balanced sub-panel", "Subpainel balanceado"),
                     ("entities", "unidades"), ("Early period", "Período inicial"),
                     ("Late period", "Período final"), ("Baseline", "Base"),
                     ("conflicting", "conflitante"), ("not stationary", "não estacionária"),
                     ("stationary", "estacionária"),
                     ("inconclusive (possibly I(2) or short sample)",
                      "inconclusivo (possível I(2) ou amostra curta)")]:
            s = s.replace(a, b)
        return s
    return EN.get(s, s)


def esc(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
           "$": r"\$", ">=": r"$\geq$", "<=": r"$\leq$", ">": r"$>$", "<": r"$<$", "|": r"$|$"}
    out = s
    for a in ["\\", "&", "%", "#", "_", "$"]:
        out = out.replace(a, rep[a])
    for a in [">=", "<=", ">", "<", "|"]:
        out = out.replace(a, rep[a])
    return out


def num(x, d=3) -> str:
    if x is None:
        return "--"
    if abs(x) >= 1e4:
        return f"{x:.2e}"
    return f"{x:.{d}f}"


def wrap(body: list[str], ncols: int, caption: str, label: str, notes: str) -> str:
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        # shrink to the text width only when the table is wider than it
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"\begin{tabular}{l" + "c" * ncols + "}", r"\toprule", *body, r"\bottomrule",
        r"\end{tabular}}", "",
        r"\vspace{2pt}\parbox{\textwidth}{\footnotesize " + notes + "}",
        r"\end{table}", ""])


def col_head(label: str) -> str:
    """Column header, split over two lines at the first comma to keep columns narrow."""
    parts = [p.strip() for p in label.split(",", 1)]
    if len(parts) == 1:
        return esc(label)
    return r"\begin{tabular}[b]{@{}c@{}}" + esc(parts[0]) + r"\\" + esc(parts[1]) + r"\end{tabular}"


def role_name(test: str, model: dict) -> str:
    """Per-variable tests are relabelled by role (y / x) so models share one row."""
    for prefix in ("Fisher-ADF panel unit root: ", "ADF/KPSS integration order: "):
        if test.startswith(prefix):
            var = test[len(prefix):]
            dep = model["formula"].split("~")[0].strip()
            role = "y" if var == dep else "x" if var in model.get("key", []) else var
            return prefix + role
    return test


def diagnostics_table(models: dict, labels: dict, lang: str) -> str:
    names = list(models)
    order: list[str] = []
    by_role: dict[str, dict[str, dict]] = {n: {} for n in names}
    for n, m in models.items():
        for t in m["tests"]:
            r = role_name(t["test"], m)
            by_role[n][r] = t
            if r not in order:
                order.append(r)
    header = tr("Test", lang) + " & " + " & ".join(col_head(labels[n]) for n in names) + r" \\"
    body = [header, r"\midrule"]
    for test in order:
        cells = []
        for n in names:
            t = by_role[n].get(test)
            if t is None or (t.get("statistic") is None and t.get("pvalue") is None):
                cells.append("--")
            elif t.get("statistic") is None:
                cells.append(esc(tr(str(t.get("order", "--")), lang)))
            else:
                cells.append(f"{num(t['statistic'], 2)} ({num(t['pvalue'])})")
        body.append(esc(tr(test, lang)) + " & " + " & ".join(cells) + r" \\")
    return wrap(body, len(names), tr("Specification and diagnostic tests", lang),
                "tab:diagnostics", tr("notes_diag", lang))


def describe_baseline(m: dict, lang: str) -> str:
    eff = {"twoway": ("efeitos fixos de unidade e tempo", "entity and time fixed effects"),
           "entity": ("efeitos fixos de unidade", "entity fixed effects"),
           "random": ("efeitos aleatórios", "random effects"),
           "none": ("MQO agrupado", "pooled OLS")}
    cov = {"clustered_entity": ("erros-padrão clusterizados por unidade",
                                "standard errors clustered by entity"),
           "clustered_time": ("erros-padrão clusterizados por tempo",
                              "standard errors clustered by time"),
           "clustered_twoway": ("erros-padrão clusterizados em duas vias",
                                "two-way clustered standard errors"),
           "driscoll_kraay": ("erros-padrão de Driscoll-Kraay", "Driscoll-Kraay standard errors"),
           "robust": ("erros-padrão robustos", "robust standard errors"),
           "unadjusted": ("erros-padrão não ajustados", "unadjusted standard errors")}
    i = 0 if lang == "pt" else 1
    e = eff.get(m.get("effects"), (None, None))[i]
    c = cov.get(m.get("cov"), (m.get("cov"), m.get("cov")))[i]
    if e is None:  # OLS (time series / cross section)
        return (f"Base: MQO com {c}." if lang == "pt" else f"Baseline: OLS with {c} SEs.")
    return f"Base: {e}, {c}." if lang == "pt" else f"Baseline: {e}, {c}."


def robustness_table(models: dict, labels: dict, lang: str) -> str:
    cols = [(n, k) for n, m in models.items() for k in m["key"]]
    multi_key = any(len(m["key"]) > 1 for m in models.values())
    head = [col_head(labels[n] + (f" ({k})" if multi_key else "")) for n, k in cols]
    body = [tr("Variant", lang) + " & " + " & ".join(head) + r" \\"]
    stacked = len(cols) > 3  # wide tables: standard error on its own line
    first = models[cols[0][0]]["variants"]
    group = None
    for i, v0 in enumerate(first):
        if v0["group"] != group:
            group = v0["group"]
            body += [r"\midrule", rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{esc(tr(group, lang))}}}}} \\"]
        row = [models[n]["variants"][i] if i < len(models[n]["variants"]) else {}
               for n, _ in cols]
        excluded = [not v.get("counts_for_verdict", True) for v in row]
        whole_row = all(excluded)  # dagger on the label; otherwise on each excluded cell
        cells, se_cells = [], []
        for (n, k), v, exc in zip(cols, row, excluded):
            c = v.get(k)
            mark = r"$^\dagger$" if exc and not whole_row else ""
            if c is None:
                cells.append("--")
                se_cells.append("")
            elif stacked:
                cells.append(f"{c['coef']:.4f}$^{{{stars(c['pvalue'])}}}${mark}")
                se_cells.append(f"({c['se']:.4f})")
            else:
                cells.append(f"{c['coef']:.4f}$^{{{stars(c['pvalue'])}}}${mark} ({c['se']:.4f})")
        name = v0["variant"]
        if v0["group"] == "Baseline":
            name = "Baseline"
        label = tr(name, lang) + (" †" if whole_row else "")
        body.append(esc(label).replace("†", r"$^\dagger$") + " & " + " & ".join(cells)
                    + r" \\")
        if stacked:
            body.append(" & " + " & ".join(se_cells) + r" \\[1pt]")
    if any(models[n].get("jackknife") for n, _ in cols):
        cells = []
        for n, k in cols:
            j = (models[n].get("jackknife") or {}).get(k)
            cells.append(f"[{j['min']:.3f}; {j['max']:.3f}]" if j else "--")
        body += [r"\midrule", esc(tr("Jackknife (leave one entity out): range", lang)) + " & "
                 + " & ".join(cells) + r" \\"]
    base_desc = describe_baseline(models[cols[0][0]], lang)
    notes = (tr("notes_rob", lang) + " " + base_desc + r" $^\dagger$ "
             + ("Informativo; não entra no veredito (estimando diferente ou estimador "
                "inconsistente segundo os testes)." if lang == "pt" else
                "Informative only; not used for the verdict (different estimand, or an "
                "estimator the tests show to be inconsistent)."))
    return wrap(body, len(cols), tr("Robustness of the key coefficients", lang),
                "tab:robustness", notes)


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    cfg = load_config(argv[0])
    lang = argv[argv.index("--lang") + 1] if "--lang" in argv else "en"
    out = Path(cfg["output_dir"])
    labels = {n: m.get("label", n) for n, m in cfg["models"].items()}
    written = []
    diag_file = out / "panel_diagnostics.json"
    rob_file = out / "robustness.json"
    ols_file = out / "ols_diagnostics.json"
    if diag_file.exists():
        d = json.loads(diag_file.read_text(encoding="utf-8"))["models"]
        (out / "table_diagnostics.tex").write_text(diagnostics_table(d, labels, lang), "utf-8")
        written.append("table_diagnostics.tex")
    if rob_file.exists():
        r = json.loads(rob_file.read_text(encoding="utf-8"))["models"]
        (out / "table_robustness.tex").write_text(robustness_table(r, labels, lang), "utf-8")
        written.append("table_robustness.tex")
    if ols_file.exists():
        o = json.loads(ols_file.read_text(encoding="utf-8"))["models"]
        (out / "table_diagnostics.tex").write_text(diagnostics_table(o, labels, lang), "utf-8")
        (out / "table_robustness.tex").write_text(robustness_table(o, labels, lang), "utf-8")
        written += ["table_diagnostics.tex", "table_robustness.tex"]
    if not written:
        sys.exit("no result JSON found in output_dir: run the diagnostics scripts first")
    for w in written:
        print(f"-> {out / w}")


if __name__ == "__main__":
    main(sys.argv[1:])
