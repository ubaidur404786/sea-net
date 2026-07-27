"""
seanet/paper/style.py - one house style for every figure in the paper.

Why a whole file just for style
-------------------------------
A paper looks amateur when its figures disagree with each other: different fonts, different blues,
one chart with a grid and the next without. Reviewers notice. So NO figure in this package sets its
own font, colour or line width - they all come from here. Change a value here and every figure in the
paper changes with it.

The rules this file enforces
----------------------------
1. VECTOR FIRST. Every figure is saved as PDF (what LaTeX should include) and SVG (editable), plus a
   600-dpi PNG for slides and quick viewing. A PDF figure stays sharp at any zoom because it stores
   lines and text, not pixels.

2. FONT MATCHED TO THE PAPER. ICLR / NeurIPS / ICML use a serif body font at 10pt. Figure text one or
   two points smaller reads as part of the page instead of as a pasted-in screenshot.

3. SIZED FOR THE COLUMN. A figure squeezed from 12 inches down to 3.25 makes its 8pt labels
   unreadable. So figures are BUILT at their final printed width (COL_WIDTH or FULL_WIDTH) and never
   scaled in LaTeX.

4. GRAYSCALE-SAFE. Many reviewers print in black and white. Colour alone must never carry meaning,
   so every series also differs by MARKER and LINE STYLE, and the palette's colours differ in
   lightness as well as hue.

5. COLOUR-BLIND SAFE. The palette is Okabe-Ito, the standard for scientific figures - it stays
   distinguishable for the common forms of colour blindness.

6. NO CHART JUNK. No 3-D effects, no heavy borders, no background fill. Only a light horizontal grid
   behind the data, and the top/right frame lines removed.
"""
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                                        # headless: we only ever save files
import matplotlib.pyplot as plt                              # noqa: E402

# --------------------------------------------------------------------------------------
# Where the figures go. One folder per paper section, so the Overleaf project can mirror it.
# --------------------------------------------------------------------------------------
PAPER_ROOT = os.path.join("results", "paper_figures")

SECTIONS = {
    "main": "01_main_figures",      # the few figures that go in the paper body
    "ablation": "02_ablation",      # what each part of the model contributes
    "appendix": "03_appendix",      # everything detailed / every model / every dataset
    "web": "04_web",                # the WebTraffic dataset on its own
    "stats": "05_statistics",       # ranks, significance, correlations
}

# --------------------------------------------------------------------------------------
# Sizes, in INCHES, matching the ICLR two-column style sheet. Build at final size, never rescale.
# --------------------------------------------------------------------------------------
COL_WIDTH = 3.25          # one column
FULL_WIDTH = 5.50         # the full text width (ICLR's \textwidth)
GOLDEN = 0.618            # a pleasant height:width ratio for a single plot

# --------------------------------------------------------------------------------------
# The palette: Okabe-Ito. Colour-blind safe, and the entries differ in LIGHTNESS too, so they stay
# distinct when the page is printed in black and white.
# --------------------------------------------------------------------------------------
C_OURS = "#0072B2"        # blue    - our models (the ones we are proposing)
C_BASELINE = "#E69F00"    # orange  - the paper's own backbones, rerun by us
C_MILLET = "#000000"      # black   - the MILLET paper's published numbers (the bar to beat)
C_GOOD = "#009E73"        # green   - a win / the Pareto front
C_BAD = "#D55E00"         # vermilion - a loss
C_NEUTRAL = "#999999"     # grey    - ties, guides, "everything else"
C_ACCENT = "#CC79A7"      # pink    - a fourth category when one is needed

# an ordered list for when a figure needs N distinguishable series
PALETTE: List[str] = [C_OURS, C_BASELINE, C_GOOD, C_ACCENT, "#56B4E9", C_BAD, "#F0E442", C_NEUTRAL]

# markers and line styles, paired with the palette. Together with colour these give every series
# THREE ways to be told apart - so the figure survives being printed in grayscale.
MARKERS: List[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES: List[str] = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), "-"]

# grayscale fallbacks: when a figure is ONLY about two groups, lightness alone separates them
GRAY_DARK = "#333333"
GRAY_LIGHT = "#BBBBBB"


def apply_style() -> None:
    """
    Install the house style into matplotlib. Call once before drawing anything.

    Every number here is a deliberate choice for a two-column paper:
      - serif fonts to match the body text,
      - 8pt tick labels / 9pt axis labels (small but legible at final size),
      - thin lines (0.8-1.4pt) because thick lines look clumsy when the figure is only 3.25in wide,
      - a light horizontal grid BEHIND the data, and no top/right frame ("despined"), which is the
        modern convention and removes two lines of chart junk from every plot.
    """
    plt.rcParams.update({
        # --- fonts: serif, to match an ICLR/NeurIPS page ---
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Computer Modern Roman"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # --- lines and markers: thin and small, for a small printed figure ---
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.6,

        # --- axes: no chart junk ---
        "axes.spines.top": False,           # remove the top frame line
        "axes.spines.right": False,         # remove the right frame line
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",              # a horizontal grid only - it guides the eye to values
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "grid.color": "#CCCCCC",
        "axes.axisbelow": True,             # grid BEHIND the data, never on top of it

        # --- legend: a light box, never a heavy one ---
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#DDDDDD",
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.8,

        # --- saving: tight margins and real vector text ---
        "figure.dpi": 150,
        "savefig.dpi": 600,                 # the PNG copy is 600 dpi, as asked
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,                 # embed real TrueType text in the PDF, not outlines,
        "ps.fonttype": 42,                  # so the text stays selectable and searchable
        "svg.fonttype": "none",             # SVG keeps text as text, so it stays editable
    })


# --------------------------------------------------------------------------------------
# Saving: one call writes the PDF, the SVG, the 600-dpi PNG, and the LaTeX metadata.
# --------------------------------------------------------------------------------------
_MANIFEST: List[Dict] = []                  # every figure saved in this run, for the metadata json


def figure_path(section: str, name: str, ext: str) -> str:
    """Build the full path of one figure file, e.g. results/paper_figures/01_main_figures/fig1.pdf."""
    folder = SECTIONS.get(section, section)
    return os.path.join(PAPER_ROOT, folder, f"{name}.{ext}")


def save(fig, section: str, name: str, *, title: str, caption: str,
         question: str, placement: str = "t", label: Optional[str] = None,
         formats=("pdf", "svg", "png")) -> Dict:
    """
    Save one figure in every format, and record what it is for the LaTeX metadata.

    Saving three formats from ONE drawing is the point: the PDF goes into LaTeX, the SVG can be
    hand-edited later if a reviewer asks, and the PNG is for slides and for looking at quickly. They
    can never drift apart because they come from the same figure object.

    fig : the matplotlib figure.
    section : which paper section ("main" / "ablation" / "appendix" / "web" / "stats").
    name : the file name without extension (also becomes the LaTeX label).
    title : a short human title (used in the metadata, not drawn on the figure).
    caption : the LaTeX caption - write it as a full sentence, as it will appear in the paper.
    question : the ONE scientific question this figure answers (our own rule: if you cannot write
               this line, the figure should not exist).
    placement : LaTeX float placement, e.g. "t" (top of page) or "h".
    label : LaTeX \\label; defaults to "fig:<name>".
    returns : the metadata entry for this figure.
    """
    entry = {
        "name": name,
        "section": section,
        "folder": SECTIONS.get(section, section),
        "title": title,
        "caption": caption,
        "question": question,
        "label": label or f"fig:{name}",
        "placement": placement,
        "files": {},
    }
    for ext in formats:
        path = figure_path(section, name, ext)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        entry["files"][ext] = path.replace(os.sep, "/")
    plt.close(fig)
    _MANIFEST.append(entry)
    return entry


def manifest() -> List[Dict]:
    """Everything saved so far in this run."""
    return list(_MANIFEST)


def reset_manifest() -> None:
    """Forget what was saved (called at the start of a full run so the metadata is not doubled)."""
    _MANIFEST.clear()


# --------------------------------------------------------------------------------------
# Small drawing helpers every figure shares
# --------------------------------------------------------------------------------------
def shorten(model_id: str, max_len: int = 22) -> str:
    """
    Turn a long model id into a label that fits on an axis.

    A model id looks like "seanet_bottleneck_topk__sea_mstcn_sep_bottleneck__sea_topk_conjunctive".
    Printed in full it is wider than the whole figure. We keep the CONFIG name (the first part,
    which is the one humans use), drop the "seanet_" prefix that every one of our models shares, and
    cut what is still too long in the middle - keeping both ends, because the ends are what differ.

    model_id : the full id.
    max_len : the longest label we allow.
    returns : the short label.
    """
    name = str(model_id).split("__")[0]
    if name.startswith("seanet_"):
        name = name[len("seanet_"):]                # every model of ours starts with this
    elif name == "seanet":
        name = "SEA-Net"
    if len(name) <= max_len:
        return name
    keep = (max_len - 1) // 2
    return f"{name[:keep]}…{name[-keep:]}"


def legend_outside(ax, loc: str = "upper center", ncol: int = 3, below: bool = True):
    """
    Put the legend OUTSIDE the plotting area, so it can never sit on top of the data.

    matplotlib's "best" location still lands on the data when the plot is crowded, and a legend over
    a data point is the single most common figure mistake. Placing it above or below the axes costs a
    little space and removes the problem completely.
    """
    if below:
        return ax.legend(loc=loc, bbox_to_anchor=(0.5, -0.18), ncol=ncol, frameon=False)
    return ax.legend(loc=loc, bbox_to_anchor=(0.5, 1.15), ncol=ncol, frameon=False)


def annotate_bars(ax, bars, values, fmt="{:.3f}", horizontal=True, fontsize=7):
    """Write each bar's value at its end - so a reader can quote the number without a table."""
    for bar, value in zip(bars, values):
        if value != value:                          # NaN check without importing pandas
            continue
        if horizontal:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + fmt.format(value),
                    va="center", ha="left", fontsize=fontsize)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value),
                    va="bottom", ha="center", fontsize=fontsize)


def write_json(obj, path: str) -> str:
    """Write any object as pretty JSON (used for the figure metadata)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path
