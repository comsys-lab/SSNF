"""Build the three -full-window .md/.tex paper tables from the driver logs (measure.py, f7_foils.py, and the shared-bit oracle log).
Reads eval_accuracy/logs/ if the drivers were re-run; otherwise falls back to the shipped raw logs in logs_subset/table/, so the tables regenerate without a GPU.
Run: python eval_accuracy/make_tables.py
"""
from __future__ import annotations
import os, re

D = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(D, "logs")
if not os.path.exists(os.path.join(L, "measure_llama31.log")):
    L = os.path.join(D, "..", "logs_subset", "table")
MODELS = [("llama31", "Llama-3.1-8B"), ("mistral", "Mistral-7B"), ("llama2", "Llama-2-7B")]


def parse(key: str) -> dict[str, float | int]:
    """Extract per-format PPL rows (and the window count) from one model's shipped logs."""
    r = {}
    t = open(os.path.join(L, f"measure_{key}.log")).read()
    r["nwin"] = int(re.search(r"tok\), (\d+)w ", t).group(1))
    r["BF16"] = float(re.search(r"BF16 PPL = ([\d.]+)", t).group(1))
    r["SSNF"] = float(re.search(r"SSNF offs=\{0\}\s+([\d.]+)", t).group(1))
    r["SSNF-search"] = float(re.search(r"SSNF offs=\{-1, ?0\}\s+([\d.]+)", t).group(1))
    for n in (8, 7, 6, 5, 4):
        r[f"MXINT{n}"] = float(re.search(rf"MXINT{n} \([\d.]+b\)\s+([\d.]+)", t).group(1))
    t = open(os.path.join(L, f"f7_{key}.log")).read()
    for name, tag in [("MXFP8-E4M3", "MXFP8"), ("MXFP6-E2M3", "MXFP6"),
                      ("MXFP4-E2M1", "MXFP4"), ("NVFP4", "NVFP4")]:
        r[tag] = float(re.search(rf"{re.escape(name)}\s+\([\d.]+b\)\s+([\d.]+)", t).group(1))
    t = open(os.path.join(L, f"oracle_{key}.log")).read()
    r["BF16-oracle"] = float(re.search(r"^BF16\s+([\d.]+)", t, re.M).group(1))
    r["SHARE1"] = float(re.search(r"HESS-share1 gs8 \([\d.]+b\)\s+([\d.]+)", t).group(1))
    return r


# 20-window comparison logs live in the full working tree, not in this package; the diff
# section is skipped when they are absent (it is a working note, not a paper table).
O20 = os.environ.get("SSNF_20W_LOGS", "")  # optional: dir with the 20-window logs (diff table)
HAS20 = os.path.isdir(os.path.join(O20, "logs"))
ORC20 = {"llama31": "Meta-Llama-3.1-8B", "mistral": "Mistral-7B-v0.3", "llama2": "Llama-2-7b-hf"}


def parse20(key: str) -> dict[str, float]:
    """Same rows from the original 20-window logs of the full working tree, for the diff table."""
    r = {}
    t = open(os.path.join(O20, "logs", f"{key}.log")).read()
    r["BF16"] = float(re.search(r"BF16 PPL = ([\d.]+)", t).group(1))
    r["SSNF"] = float(re.search(r"SSNF offs=\{0\}\s+([\d.]+)", t).group(1))
    r["SSNF-search"] = float(re.search(r"SSNF offs=\{-1, ?0\}\s+([\d.]+)", t).group(1))
    for n in (8, 7, 6, 5, 4):
        r[f"MXINT{n}"] = float(re.search(rf"MXINT{n} \([\d.]+b\)\s+([\d.]+)", t).group(1))
    t = open(os.path.join(O20, "logs", f"f7_{key}.log")).read()
    for name, tag in [("MXFP8-E4M3", "MXFP8"), ("MXFP6-E2M3", "MXFP6"),
                      ("MXFP4-E2M1", "MXFP4"), ("NVFP4", "NVFP4")]:
        r[tag] = float(re.search(rf"{re.escape(name)}\s+\([\d.]+b\)\s+([\d.]+)", t).group(1))
    t = open(os.path.join(O20, "oracle20w_logs", f"oracle20w_{ORC20[key]}.log"), errors="ignore").read()
    r["SHARE1"] = float(re.search(r"HESS-share1 gs8 \([\d.]+b\)\s+([\d.]+)", t).group(1))
    return r


R = {k: parse(k) for k, _ in MODELS}
R20 = {k: parse20(k) for k, _ in MODELS} if HAS20 else None


def cell(k: str, row: str) -> str:        # "5.50 (+0.15)"  -- raw PPL, absolute increase over BF16
    p, b = R[k][row], R[k]["BF16"]
    return f"{p:.2f} (+{p - b:.2f})" if row != "BF16" else f"{p:.2f}"


def pct(k: str, row: str) -> float:
    """Relative PPL increase over BF16 in percent (full-window)."""
    return (R[k][row] / R[k]["BF16"] - 1) * 100


def texrows(rows: list[tuple[str, str, str]], wbits: bool = True) -> list[str]:
    """Format (label, tag, bpe) rows as LaTeX table lines."""
    out = []
    for label, tag, bpe in rows:
        b = f" & {bpe} " if wbits else " "
        out.append(f"{label:<37}{b}& " + " & ".join(f"{cell(k, tag):<13}" for k, _ in MODELS) + r" \\")
    return out


# No window statement in the captions: evaluating the whole test split is the unmarked default in the
# quantization literature, and the 2,048/stride-1,024 protocol is already stated in 4_evaluation.tex.
# Window counts stay in the .md files only.

# ---------------------------------------------------------------- F2: MXINT bitwidth
mxint_rows = [(rf"MXINT{n}", f"MXINT{n}", f"{n + 0.25:5.2f}") for n in (8, 7, 6, 5, 4)]
tex_mxint = r"""\begin{table}[t]
\centering
\caption{Perplexity (PPL) results across MXINT formats.}
\label{tab:motivation-mxint-ppl}
\footnotesize
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lrccc@{}}
\toprule
\textbf{Format}
& \textbf{Bits/value}
& \textbf{Llama-3.1-8B}
& \textbf{Mistral-7B}
& \textbf{Llama-2-7B} \\
\midrule
""" + f"BF16{'':<33} & 16.00 & " + " & ".join(f"{cell(k, 'BF16'):<13}" for k, _ in MODELS) + r" \\" + """
\\specialrule{\\lightrulewidth}{0.5pt}{1.5pt}
""" + "\n".join(texrows(mxint_rows)) + r"""
\bottomrule
\end{tabular}}

\vspace{2pt}
\raggedright
\scriptsize
WikiText-2 PPL of MXINT-quantized weights and KV caches across element bitwidths.
The transformer linear-layer weights and the entire KV cache are quantized.
Values in parentheses give the absolute PPL increase over BF16.
\end{table}
"""

# ---------------------------------------------------------------- F3: one shared bit
share_rows = [("MXINT6", "MXINT6", " 6.25"), ("MXINT5", "MXINT5", " 5.25"),
              (r"MXINT5 with a shared bit ($g{=}8$)", "SHARE1", " 5.38")]
tex_share = r"""\begin{table}[t]
\centering
\caption{PPL results for MXINT6, MXINT5, and MXINT5 with a shared bit.}
\label{tab:motivation-sharing-ppl}
\footnotesize
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lrccc@{}}
\toprule
\textbf{Format}
& \textbf{Bits/value}
& \textbf{Llama-3.1-8B}
& \textbf{Mistral-7B}
& \textbf{Llama-2-7B} \\
\midrule
""" + f"BF16{'':<33} & 16.00 & " + " & ".join(f"{cell(k, 'BF16'):<13}" for k, _ in MODELS) + r" \\" + """
\\specialrule{\\lightrulewidth}{0.5pt}{1.5pt}
""" + "\n".join(texrows(share_rows)) + r"""
\bottomrule
\end{tabular}}

\vspace{2pt}
\raggedright
\scriptsize
WikiText-2 PPL of MXINT6, MXINT5, and MXINT5 with one low-order mantissa bit shared across a group of $g$ values.
The transformer linear-layer weights and the entire KV cache are quantized.
Values in parentheses give the absolute PPL increase over BF16.
\end{table}
"""

# ---------------------------------------------------------------- F7: format comparison
eval_rows = [("MXINT8", "MXINT8", None), ("MXFP8 (E4M3)", "MXFP8", None),
             ("MXFP6 (E2M3)", "MXFP6", None), ("NVFP4", "NVFP4", None),
             ("MXFP4 (E2M1)", "MXFP4", None), ("SSNF", "SSNF", None)]
tex_eval = r"""\begin{table}[t]
\centering
\caption{PPL results for existing SNFs and SSNF.}
\label{tab:evaluation-ppl}
\footnotesize
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
\textbf{Format}
& \textbf{Llama-3.1-8B}
& \textbf{Mistral-7B}
& \textbf{Llama-2-7B} \\
\midrule
""" + f"{'BF16':<14}& " + " & ".join(f"{cell(k, 'BF16'):<13}" for k, _ in MODELS) + r" \\" + """
\\specialrule{\\lightrulewidth}{0.5pt}{1.5pt}
""" + "\n".join(f"{lab:<14}& " + " & ".join(f"{cell(k, tag):<13}" for k, _ in MODELS) + r" \\"
                for lab, tag, _ in eval_rows) + r"""
\bottomrule
\end{tabular*}

\vspace{2pt}
\raggedright
\scriptsize
Values in parentheses give the absolute PPL increase over BF16.
\end{table}
"""


def pct20(k: str, row: str) -> float:
    """Relative PPL increase over BF16 in percent (20-window logs)."""
    return (R20[k][row] / R20[k]["BF16"] - 1) * 100


def diff_md(rows: list[tuple[str, str]]) -> str:
    """20-window vs full-window: raw BF16 shift, and per-format Delta-PPL% shift in pp."""
    if not HAS20:
        return ""
    out = ["", "## Difference vs the 20-window tables", "",
           "| Model | BF16 20w | BF16 full | shift |", "|---|---:|---:|---:|"]
    for k, n in MODELS:
        out.append(f"| {n} | {R20[k]['BF16']:.4f} | {R[k]['BF16']:.4f} | "
                   f"{R[k]['BF16'] - R20[k]['BF16']:+.4f} |")
    out += ["", "dPPL% vs BF16 (20w -> full; parentheses give the pp change):", "",
            "| Format |" + "".join(f" {n} |" for _, n in MODELS),
            "|---|" + "---:|" * len(MODELS)]
    for lab, tag in rows:
        cells = "".join(f" {pct20(k, tag):+.2f}% → {pct(k, tag):+.2f}% "
                        f"({pct(k, tag) - pct20(k, tag):+.2f}pp) |" for k, _ in MODELS)
        out.append(f"| {lab} |" + cells)
    return "\n".join(out) + "\n"


def md(title: str, note: str, rows: list[tuple[str, str, str]], bpe_col: bool) -> str:
    """Format one full-window table as Markdown."""
    h = "| Format |" + (" Bits/value |" if bpe_col else "") + "".join(f" {n} |" for _, n in MODELS)
    s = "|---|" + ("---:|" if bpe_col else "") + "---:|---:|---:|"
    lines = [f"# {title}", "", note, "", h, s]
    b16 = "| BF16 |" + (" 16.00 |" if bpe_col else "")
    lines.append(b16 + "".join(f" {R[k]['BF16']:.4f} |" for k, _ in MODELS))
    for lab, tag, bpe in rows:
        cells = "".join(f" {R[k][tag]:.4f} (+{R[k][tag] - R[k]['BF16']:.4f}, {pct(k, tag):+.2f}%) |"
                        for k, _ in MODELS)
        lines.append(f"| {lab} |" + (f" {bpe.strip()} |" if bpe_col else "") + cells)
    return "\n".join(lines) + "\n"


COND = (f"WikiText-2-raw test, **full window** (Llama-3.1-8B {R['llama31']['nwin']}w / "
        f"Mistral-7B {R['mistral']['nwin']}w / Llama-2-7B {R['llama2']['nwin']}w, 2,048 tokens at stride "
        "1,024), sdpa, no rotation. W+K+V all quantized (embeddings / LM head / activations stay BF16).\n"
        "Same drivers, encoders, and calibration as the 20-window working tables; "
        "only the window cap changed (`PPL_WINDOWS`).\n"
        "Cells are `raw PPL (+absolute increase, +relative %)`; the `.tex` files carry raw PPL and the "
        "absolute increase only.")

md_mxint = md("MXINT element bitwidth vs PPL — full window", COND,
              [(f"MXINT{n}", f"MXINT{n}", f"{n + 0.25:.2f}") for n in (8, 7, 6, 5, 4)], True)
md_share = md("One shared mantissa bit over MXINT5 — full window", COND + "\n\nThe shared-bit row is the "
              "per-group oracle (exhaustive argmin over the shared code under importance-weighted SSE) — "
              "the accuracy upper bound of one shared bit, not a deployed encoder.",
              [("MXINT6", "MXINT6", "6.25"), ("MXINT5", "MXINT5", "5.25"),
               ("MXINT5 + shared bit (g=8)", "SHARE1", "5.375")], True)
md_eval = md("Format comparison, W+KV full-quant — full window", COND + "\n\nSSNF = W u3/gs8 (5.63b), "
             "K activation-weighted mean u3/gs16 (5.44b), V arithmetic mean u4/gs32 (4.38b), KV avg 4.91b, "
             "no weight scale search (`offs={0}`). The `SSNF (W scale search)` row is the `offs={-1,0}` "
             "variant, reported for provenance and not used in the paper table.",
             [("MXINT8", "MXINT8", "8.25"), ("MXFP8 (E4M3)", "MXFP8", "8.25"),
              ("MXFP6 (E2M3)", "MXFP6", "6.25"), ("NVFP4", "NVFP4", "4.50"),
              ("MXFP4 (E2M1)", "MXFP4", "4.25"), ("SSNF", "SSNF", "W 5.63 / KV 4.91"),
              ("SSNF (W scale search)", "SSNF-search", "W 5.63 / KV 4.91")], True)

# BF16 cross-check: the oracle driver re-measures BF16 in its own harness.
chk = "\n## BF16 cross-check (independent harness)\n\n| Model | measure.py | oracle driver | delta |\n|---|---:|---:|---:|\n"
for k, n in MODELS:
    chk += f"| {n} | {R[k]['BF16']:.4f} | {R[k]['BF16-oracle']:.4f} | {R[k]['BF16-oracle'] - R[k]['BF16']:+.4f} |\n"

d_mxint = diff_md([(f"MXINT{n}", f"MXINT{n}") for n in (8, 7, 6, 5, 4)])
d_share = diff_md([("MXINT6", "MXINT6"), ("MXINT5", "MXINT5"),
                   ("MXINT5 + shared bit (g=8)", "SHARE1")])
d_eval = diff_md([("MXINT8", "MXINT8"), ("MXFP8 (E4M3)", "MXFP8"), ("MXFP6 (E2M3)", "MXFP6"),
                  ("NVFP4", "NVFP4"), ("MXFP4 (E2M1)", "MXFP4"), ("SSNF", "SSNF"),
                  ("SSNF (W scale search)", "SSNF-search")])

for name, mtxt, ttxt in [("motivation-mxint-ppl", md_mxint + d_mxint, tex_mxint),
                         ("motivation-sharing-ppl", md_share + d_share + chk, tex_share),
                         ("evaluation-ppl", md_eval + d_eval, tex_eval)]:
    open(os.path.join(D, f"{name}-full-window.md"), "w").write(mtxt)
    open(os.path.join(D, f"{name}-full-window.tex"), "w").write(ttxt)
    print(f"wrote {name}-full-window.{{md,tex}}")
