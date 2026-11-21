# Diagnostics and robustness — {title}

*Generated with the econometric-diagnostics skill on {date}. Data: `{data}`. Config: `config.json`.*

## 1. Verdict

One short paragraph per key coefficient: ROBUST / MOSTLY ROBUST / FRAGILE, the baseline
value, and the one-sentence reason. Then the one or two changes you recommend for the paper.

| Model | Key coefficient | Baseline (SE) | Verdict | Range across robustness variants |
|---|---|---|---|---|

## 2. Replication

Baseline reproduced: coefficients, SEs and N match `<user's output file>` (yes/no, with
the numbers). Sample: N entities, T periods, unbalanced/balanced, exclusions.

## 3. Diagnostic tests

Table with the test, H0, and the result per model (statistic, p-value). Then, in plain
words, what each block means for this paper — not a definition of the test.

## 4. Decisions: estimator and inference

- Estimator: ... because ...
- Standard errors: ... because ...
- What the current baseline already does right / should change.

## 5. Robustness

Summary of the battery: which variants move the key coefficient, by how much, and why.
Jackknife: range and most influential entities.

## 6. Heterogeneity

Sub-period and sub-panel differences, read as substantive results or as specification
issues. Suggest how to model them if relevant.

## 7. Recommendations for the paper

Numbered, concrete: what to change in the main table, what to add to the appendix
(`table_diagnostics.tex`, `table_robustness.tex`), what to say in the text.

## 8. Methodology paragraph (ready to paste)

One paragraph in the paper's language describing the tests and the robustness checks,
with their results, in the style of an applied economics journal.

## 9. Limitations of these checks

What was not tested and why it matters here (see "Not covered" in
`references/interpretation.md`).

## Files

- `table_diagnostics.tex`, `table_robustness.tex` — `\input{}` them; need `booktabs`, `graphicx`.
- `inspection.json`, `panel_diagnostics.json`, `robustness.json` — full results.
