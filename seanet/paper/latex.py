"""
seanet/paper/latex.py - the bridge between the figures and the Overleaf project.

What this writes
----------------
    figures.json      : one record per figure - title, caption, section, label, the scientific
                        question it answers, and the path of every exported file.
    figures.tex       : a ready \\begin{figure} block for every figure, in section order. Copy a
                        block into the paper, or \\input the whole file to see them all.
    tables.tex        : \\input lines for every generated table.
    README.md         : how to wire the folder into Overleaf, and what each figure is for.

Why bother
----------
Writing captions by hand at submission time is where figures and text drift apart - a caption ends
up describing an older version of a plot. Here the caption is written next to the code that draws
the figure, and this file just collects them. Regenerate the figures and the captions follow.
"""
import os
from typing import Dict, List

from seanet.paper import style as S

# the order sections appear in the paper
SECTION_ORDER = ["main", "web", "ablation", "stats", "appendix"]

SECTION_TITLES = {
    "main": "Main paper figures",
    "web": "WebTraffic dataset figures",
    "ablation": "Ablation figures",
    "stats": "Statistical analysis figures",
    "appendix": "Appendix figures",
}


def figure_block(entry: Dict, width: str = "\\linewidth") -> str:
    """
    One ready-to-paste LaTeX figure block.

    The path is written relative to the paper root and WITHOUT the file extension, which is how
    \\includegraphics should always be used: LaTeX then picks the PDF when compiling with pdflatex
    and could pick another format elsewhere, and the same line keeps working if the format changes.
    """
    pdf = entry["files"].get("pdf", "")
    stem = pdf[:-4] if pdf.endswith(".pdf") else pdf         # drop ".pdf"
    return "\n".join([
        f"% {entry['title']}",
        f"% Question answered: {entry['question']}",
        f"\\begin{{figure}}[{entry.get('placement', 't')}]",
        "  \\centering",
        f"  \\includegraphics[width={width}]{{{stem}}}",
        f"  \\caption{{{entry['caption']}}}",
        f"  \\label{{{entry['label']}}}",
        "\\end{figure}",
    ])


def write_latex_snippets(entries: List[Dict], tables: List[Dict],
                         root: str = S.PAPER_ROOT) -> Dict[str, str]:
    """
    Write figures.json, figures.tex, tables.tex and README.md.

    entries : the figure metadata (from style.manifest()).
    tables : the table metadata (from tables.generate_tables()).
    returns : {what: path} for everything written.
    """
    os.makedirs(root, exist_ok=True)
    written = {}

    # ---- the machine-readable record ----
    by_section: Dict[str, List[Dict]] = {}
    for entry in entries:
        by_section.setdefault(entry["section"], []).append(entry)
    payload = {
        "figures": entries,
        "tables": tables,
        "sections": {name: SECTION_TITLES.get(name, name) for name in by_section},
        "counts": {name: len(items) for name, items in by_section.items()},
    }
    written["json"] = S.write_json(payload, os.path.join(root, "figures.json"))

    # ---- one \begin{figure} block per figure, grouped by section ----
    lines = ["% ---------------------------------------------------------------",
             "% Auto-generated figure blocks. Copy the ones you need into the paper,",
             "% or \\input this whole file to see every figure at once.",
             "%",
             "% Requires in the preamble:  \\usepackage{graphicx}",
             "% ---------------------------------------------------------------", ""]
    for section in SECTION_ORDER:
        items = by_section.get(section)
        if not items:
            continue
        lines += [f"% ===== {SECTION_TITLES.get(section, section)} "
                  f"({len(items)} figure{'s' if len(items) != 1 else ''}) =====", ""]
        for entry in items:
            # a full-width figure gets figure*, which spans both columns in a two-column layout
            wide = entry["name"].startswith(("appendix_dataset", "stats_pairwise", "topk"))
            block = figure_block(entry)
            if wide:
                block = block.replace("\\begin{figure}", "\\begin{figure*}") \
                             .replace("\\end{figure}", "\\end{figure*}")
            lines += [block, ""]
    path = os.path.join(root, "figures.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    written["figures_tex"] = path

    # ---- the tables ----
    lines = ["% Auto-generated table inputs.",
             "% Requires in the preamble:  \\usepackage{booktabs}", ""]
    for table in tables:
        rel = table["files"]["tex"]
        lines += [f"% {table['caption'][:90]}...", f"\\input{{{rel[:-4]}}}", ""]
    path = os.path.join(root, "tables.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    written["tables_tex"] = path

    # ---- a human README ----
    written["readme"] = _write_readme(by_section, tables, root)
    return written


def _write_readme(by_section: Dict[str, List[Dict]], tables: List[Dict], root: str) -> str:
    """Write the folder's README: what is here, and how to use it in Overleaf."""
    lines = [
        "# Paper figures and tables",
        "",
        "Everything in this folder is generated by `python main.py paper`. Do not edit by hand -",
        "your edits will be overwritten the next time it runs. Change the code instead:",
        "`seanet/paper/figures.py` (figures), `seanet/paper/tables.py` (tables).",
        "",
        "## Using this in Overleaf",
        "",
        "1. Upload this whole folder to the Overleaf project, keeping the folder names.",
        "2. Add to the preamble:",
        "",
        "```latex",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\graphicspath{{results/paper_figures/}}",
        "```",
        "",
        "3. Copy a block from `figures.tex`, or `\\input{results/paper_figures/figures.tex}`",
        "   to drop every figure in at once and then delete the ones you do not want.",
        "",
        "Every figure is exported three times from the same drawing:",
        "",
        "| format | use it for |",
        "|---|---|",
        "| `.pdf` | LaTeX - vector, stays sharp at any zoom |",
        "| `.svg` | hand-editing later if a reviewer asks for a tweak |",
        "| `.png` | slides and quick viewing (600 dpi) |",
        "",
        "## What is in each folder",
        "",
    ]
    for section in SECTION_ORDER:
        items = by_section.get(section)
        if not items:
            continue
        folder = S.SECTIONS.get(section, section)
        lines += [f"### `{folder}/` - {SECTION_TITLES.get(section, section)}", ""]
        lines += ["| figure | answers |", "|---|---|"]
        for entry in items:
            lines.append(f"| `{entry['name']}` | {entry['question']} |")
        lines.append("")

    if tables:
        lines += ["## Tables", "",
                  "Each table is written as `.tex` (booktabs), `.csv` (raw numbers) and `.md`.", "",
                  "| table | rows | about |", "|---|---|---|"]
        for table in tables:
            short = table["caption"].split(".")[0]
            lines.append(f"| `{table['name']}` | {table['rows']} | {short} |")
        lines.append("")

    lines += [
        "## The rule these figures follow",
        "",
        "Every figure answers exactly one scientific question - that question is stored with the",
        "figure in `figures.json` and repeated as a comment above its LaTeX block. If a figure's",
        "question cannot be written in one line, it does not belong in the paper.",
        "",
    ]
    path = os.path.join(root, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
