#!/usr/bin/env python3
"""Generate a markdown results report from notebooks/expermint_ppp.ipynb.

This script extracts inline plot outputs from the notebook, pulls key metric
lines (currently OOS R^2), and writes a compact markdown report that can be
regenerated after each run.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _join_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def _infer_caption(source: str, cell_index: int, plot_index: int) -> str:
    source_l = source.lower()
    if "monthly_r2.plot" in source_l or "monthly oos" in source_l:
        return "Monthly OOS R^2"
    if "diag_dfs" in source_l and "grassmann_dist" in source_l:
        return "Intrinsic Dimension and Grassmann Stability"
    return f"Notebook plot {plot_index} (cell {cell_index})"


def _extract_images(nb: dict[str, Any]) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(nb.get("cells", [])):
        source = _join_text(cell.get("source", []))
        outputs = cell.get("outputs", [])
        for output_index, out in enumerate(outputs):
            data = out.get("data", {})
            if not isinstance(data, dict):
                continue
            if "image/png" not in data:
                continue
            image_b64 = _join_text(data["image/png"]).replace("\n", "")
            plots.append(
                {
                    "cell_index": cell_index,
                    "output_index": output_index,
                    "caption": _infer_caption(source, cell_index, len(plots) + 1),
                    "image_b64": image_b64,
                }
            )
    return plots


def _extract_oos_r2_values(nb: dict[str, Any]) -> list[float]:
    values: list[float] = []
    pattern = re.compile(r"OOS\s*R.*?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = ""
            if out.get("output_type") == "stream":
                text = _join_text(out.get("text"))
            else:
                data = out.get("data", {})
                if isinstance(data, dict):
                    text = _join_text(data.get("text/plain"))

            if not text:
                continue

            for match in pattern.finditer(text):
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    continue

    return values


def _extract_errors(nb: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for cell_index, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs", []):
            if out.get("output_type") != "error":
                continue
            errors.append(
                {
                    "cell": str(cell_index),
                    "ename": _join_text(out.get("ename")),
                    "evalue": _join_text(out.get("evalue")),
                }
            )
    return errors


def _write_plot_assets(plots: list[dict[str, Any]], assets_dir: Path) -> list[dict[str, Any]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for idx, plot in enumerate(plots, start=1):
        filename = f"plot_{idx:02d}_cell{plot['cell_index']}_out{plot['output_index']}.png"
        path = assets_dir / filename
        path.write_bytes(base64.b64decode(plot["image_b64"]))
        written.append(
            {
                "caption": plot["caption"],
                "relative_path": f"{assets_dir.name}/{filename}",
                "cell_index": plot["cell_index"],
            }
        )
    return written


def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _build_markdown(
    notebook_path: Path,
    nb: dict[str, Any],
    images: list[dict[str, Any]],
    oos_r2_values: list[float],
    errors: list[dict[str, str]],
) -> str:
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    notebook_modified = _format_timestamp(notebook_path.stat().st_mtime)
    lines: list[str] = []

    lines.append("# expermint_ppp Results")
    lines.append("")
    lines.append(f"- Source notebook: `{notebook_path.as_posix()}`")
    lines.append(f"- Notebook last modified: `{notebook_modified}`")
    lines.append(f"- Report generated: `{generated_at}`")
    lines.append(f"- Notebook cells: `{len(nb.get('cells', []))}`")
    lines.append(f"- Plot outputs extracted: `{len(images)}`")
    lines.append("")

    lines.append("## Key Metrics")
    if oos_r2_values:
        for idx, value in enumerate(oos_r2_values, start=1):
            lines.append(f"- Overall OOS R^2 ({idx}): `{value:.4f}`")
    else:
        lines.append("- No OOS R^2 value found in notebook outputs.")
    lines.append("")

    lines.append("## Plots")
    if images:
        for idx, item in enumerate(images, start=1):
            lines.append(f"### {idx}. {item['caption']}")
            lines.append(f"Cell: `{item['cell_index']}`")
            lines.append("")
            lines.append(f"![{item['caption']}]({item['relative_path']})")
            lines.append("")
    else:
        lines.append("No inline plots were found in notebook outputs.")
        lines.append("")

    lines.append("## Run Issues")
    if errors:
        for err in errors:
            lines.append(
                f"- Cell {err['cell']}: `{err['ename']}` - {err['evalue'] or 'No error message'}"
            )
    else:
        lines.append("- No notebook errors captured in outputs.")
    lines.append("")

    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Escape minimal LaTeX special chars in plain text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _build_latex(
    notebook_path: Path,
    nb: dict[str, Any],
    images: list[dict[str, Any]],
    oos_r2_values: list[float],
    errors: list[dict[str, str]],
    assets_dir: Path,
) -> str:
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    notebook_modified = _format_timestamp(notebook_path.stat().st_mtime)

    lines: list[str] = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{float}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage[T1]{fontenc}")
    lines.append(r"\title{expermint\_ppp Results}")
    lines.append(r"\date{}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\section*{Metadata}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"\item Source notebook: \texttt{{{_latex_escape(notebook_path.as_posix())}}}")
    lines.append(rf"\item Notebook last modified: \texttt{{{_latex_escape(notebook_modified)}}}")
    lines.append(rf"\item Report generated: \texttt{{{_latex_escape(generated_at)}}}")
    lines.append(rf"\item Notebook cells: \texttt{{{len(nb.get('cells', []))}}}")
    lines.append(rf"\item Plot outputs extracted: \texttt{{{len(images)}}}")
    lines.append(r"\end{itemize}")

    lines.append(r"\section*{Key Metrics}")
    if oos_r2_values:
        lines.append(r"\begin{itemize}")
        for idx, value in enumerate(oos_r2_values, start=1):
            lines.append(rf"\item Overall OOS R\^2 ({idx}): \texttt{{{value:.4f}}}")
        lines.append(r"\end{itemize}")
    else:
        lines.append("No OOS R\\^2 value found in notebook outputs.")

    lines.append(r"\section*{Plots}")
    if images:
        for idx, item in enumerate(images, start=1):
            rel_img_path = (assets_dir.name + "/" + Path(item["relative_path"]).name).replace("\\", "/")
            caption = _latex_escape(item["caption"])
            lines.append(r"\begin{figure}[H]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=0.95\textwidth]{{{rel_img_path}}}")
            lines.append(rf"\caption{{{idx}. {caption} (cell {item['cell_index']})}}")
            lines.append(r"\end{figure}")
    else:
        lines.append("No inline plots were found in notebook outputs.")

    lines.append(r"\section*{Run Issues}")
    if errors:
        lines.append(r"\begin{itemize}")
        for err in errors:
            ename = _latex_escape(err["ename"] or "Error")
            evalue = _latex_escape(err["evalue"] or "No error message")
            lines.append(rf"\item Cell {err['cell']}: \texttt{{{ename}}} -- {evalue}")
        lines.append(r"\end{itemize}")
    else:
        lines.append("No notebook errors captured in outputs.")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def _build_pdf_from_latex(latex_path: Path, pdf_path: Path) -> None:
    """Compile LaTeX into PDF using pdflatex, if available."""
    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin is None:
        raise RuntimeError(
            "pdflatex is not installed or not on PATH. Install a LaTeX distribution "
            "(e.g., MacTeX/TeX Live) to enable PDF generation."
        )

    workdir = latex_path.parent
    cmd = [pdflatex_bin, "-interaction=nonstopmode", "-halt-on-error", latex_path.name]
    # Two passes for stable references/figure numbering.
    for _ in range(2):
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-20:])
            raise RuntimeError(f"pdflatex failed for {latex_path.name}.\n{tail}")

    built_pdf = workdir / f"{latex_path.stem}.pdf"
    if not built_pdf.exists():
        raise RuntimeError(f"Expected PDF not found after pdflatex run: {built_pdf}")
    if built_pdf.resolve() != pdf_path.resolve():
        pdf_path.write_bytes(built_pdf.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate markdown results report from notebooks/expermint_ppp.ipynb",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/expermint_ppp.ipynb"),
        help="Path to the source notebook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/expermint_ppp_results.md"),
        help="Path to the markdown report output.",
    )
    parser.add_argument(
        "--latex",
        type=Path,
        default=Path("results/expermint_ppp_results.tex"),
        help="Path to the LaTeX report output.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("results/expermint_ppp_results.pdf"),
        help="Path to the PDF report output.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF build step.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = args.notebook if args.notebook.is_absolute() else repo_root / args.notebook
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    latex_path = args.latex if args.latex.is_absolute() else repo_root / args.latex
    pdf_path = args.pdf if args.pdf.is_absolute() else repo_root / args.pdf
    assets_dir = output_path.parent / "expermint_ppp_assets"

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    plots = _extract_images(nb)
    saved_images = _write_plot_assets(plots, assets_dir)
    oos_r2_values = _extract_oos_r2_values(nb)
    errors = _extract_errors(nb)

    markdown = _build_markdown(
        notebook_path=notebook_path,
        nb=nb,
        images=saved_images,
        oos_r2_values=oos_r2_values,
        errors=errors,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    latex = _build_latex(
        notebook_path=notebook_path,
        nb=nb,
        images=saved_images,
        oos_r2_values=oos_r2_values,
        errors=errors,
        assets_dir=assets_dir,
    )
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex, encoding="utf-8")

    print(f"Wrote report: {output_path}")
    print(f"Wrote LaTeX: {latex_path}")
    print(f"Wrote plot assets: {assets_dir}")
    if args.no_pdf:
        print("Skipped PDF build (--no-pdf).")
    else:
        try:
            _build_pdf_from_latex(latex_path=latex_path, pdf_path=pdf_path)
            print(f"Wrote PDF: {pdf_path}")
        except RuntimeError as exc:
            print(f"PDF build skipped: {exc}")


if __name__ == "__main__":
    main()
