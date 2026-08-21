"""
seanet/analysis/style.py - one house style for every figure in the paper.

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
# Where the comparison figures go. One folder per QUESTION, so you can find a figure by what it
# answers rather than by which model produced it.
# --------------------------------------------------------------------------------------
ANALYSIS_ROOT = os.path.join("results", "analysis")
PAPER_ROOT = ANALYSIS_ROOT          # old name, kept so nothing that imports it breaks


def set_analysis_root(path: str) -> str:
    """
    Point the analysis output at a different folder (configs/main.yaml: output.analysis_dir).

    main.py calls this once at start-up, so the folder named in the config is the folder actually
    used. See seanet/results.py: set_results_root for the same idea on the results side.

    path : the new root folder.
    returns : the path that was set.
    """
    global ANALYSIS_ROOT, PAPER_ROOT
    ANALYSIS_ROOT = str(path)
    PAPER_ROOT = ANALYSIS_ROOT
    return ANALYSIS_ROOT

SECTIONS = {
    "main": "01_leaderboard",       # who is strongest overall
    "ablation": "02_ablation",      # what each encoder / pooling choice contributes
    "appendix": "03_detail",        # every model, every dataset
    "web": "04_webtraffic",         # the WebTraffic dataset on its own
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
        "savefig.dpi": 300,                 # 300 dpi is sharp on screen and a sane file size
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
    """Build the full path of one figure file, e.g. results/analysis/01_leaderboard/fig1.png."""
    folder = SECTIONS.get(section, section)
    return os.path.join(PAPER_ROOT, folder, f"{name}.{ext}")


def save(fig, section: str, name: str, *, title: str, caption: str,
         question: str, placement: str = "t", label: Optional[str] = None,
         formats=("png",)) -> Dict:
    """
    Save one figure and record what it is (title, caption, and the question it answers).

    Only PNG is written. The project used to save each figure three times (PDF for LaTeX, SVG to
    hand-edit, PNG to look at); the LaTeX report was removed in seanetv7, so the two extra copies
    were pure duplication.

    fig : the matplotlib figure.
    section : which group it belongs to ("main" / "ablation" / "appendix" / "web" / "stats").
    name : the file name without extension.
    title : a short human title (used in the metadata, not drawn on the figure).
    caption : one full sentence describing the figure.
    question : the ONE scientific question this figure answers (our own rule: if you cannot write
               this line, the figure should not exist).
    placement : kept so older recorded metadata still loads; unused.
    label : an id for the figure; defaults to "fig:<name>".
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
# --------------------------------------------------------------------------------------
# Short display names for figures and tables.
#
# A model id is built for STORAGE - it must be unique and self-describing, so it is long:
#
#     seanet_spiketrend__sea_mstcn_sep_spiketrend__sea_classwise_conjunctive
#
# A figure axis needs something else entirely: short, but still saying which encoder and which
# pooling head, and whose they are. So we abbreviate each half and join them:
#
#     sea_spiketrend__sea_cls_conj
#          encoder        pooling
#
# The "sea_" / "mil_" prefixes survive the abbreviation on purpose - that is the whole point. A
# reader sees "sea_mstcn__mil_conj" and knows immediately: our encoder, MILLET's pooling head. No
# dagger, no legend entry, no cross-reference to a table.
#
# These are DISPLAY names only. Nothing on disk is renamed - the folders keep their full ids.
# --------------------------------------------------------------------------------------
ENCODER_SHORT = {
    "sea_mstcn_sep": "sea_mstcn",
    "sea_mstcn_sep_gated": "sea_gated",
    "sea_mstcn_sep_spiketrend": "sea_spiketrend",
    "sea_mstcn_sep_bottleneck": "sea_bottleneck",
    "sea_mstcn_sep_inputgate": "sea_inputgate",
    "sea_mstcn_sep_recon": "sea_recon",
    "mil_inceptiontime": "mil_incept",
    "mil_fcn": "mil_fcn",
    "mil_resnet": "mil_resnet",
}

POOLING_SHORT = {
    "sea_classwise_conjunctive": "sea_cls_conj",
    "sea_softmax_conjunctive": "sea_sm_conj",
    "sea_adaptive_classwise": "sea_adapt_cls",
    "sea_topk_conjunctive": "sea_topk_conj",
    "sea_attention_max": "sea_attn_max",
    "sea_gated_attention": "sea_gated_attn",
    "sea_dualstream_conjunctive": "sea_dual_conj",
    "mil_additive": "mil_add",
    "mil_conjunctive": "mil_conj",
    "mil_attention": "mil_attn",
    "mil_instance": "mil_inst",
    "mil_gap": "mil_gap",
}

# Filled in by build_label_map(). Kept module-level on purpose: the SAME model must get the SAME
# label in every figure of the paper, so the map is computed once from every known model and then
# reused, rather than each figure inventing its own abbreviation from whatever it happens to hold.
_LABEL_MAP: Dict[str, str] = {}


def _abbreviate(model_id: str) -> str:
    """
    Abbreviate one model id into "<short encoder>__<short pooling>", ignoring collisions.

    Anything not in the two tables above is shortened generically (keep the prefix, keep the last
    word), so a brand-new encoder or head still gets a sensible label the day it is added instead
    of crashing or printing its full name.
    """
    parts = str(model_id).split("__")
    if len(parts) < 3:                              # e.g. the "MILLET (published)" reference row
        return str(model_id)
    encoder, pooling = parts[1], parts[2]
    enc = ENCODER_SHORT.get(encoder)
    pool = POOLING_SHORT.get(pooling)
    if enc is None:                                 # unknown encoder: prefix + last word
        bits = encoder.split("_")
        enc = f"{bits[0]}_{bits[-1]}" if len(bits) > 1 else encoder
    if pool is None:
        bits = pooling.split("_")
        pool = f"{bits[0]}_{bits[-1]}" if len(bits) > 1 else pooling
    return f"{enc}__{pool}"


def build_label_map(model_ids: List[str]) -> Dict[str, str]:
    """
    Build the display label for every model, resolving the collisions the short form creates.

    THE COLLISION PROBLEM. Abbreviating to encoder+pooling loses information that the config name
    carried. gated_last, gated_max and gated_mean are three different experiments that share the
    same encoder (sea_mstcn_sep_gated) AND the same pooling head - they differ only in a `summary`
    setting inside the config file. Abbreviated blindly, all three become "sea_gated__sea_cls_conj"
    and the figure silently shows three different models under one name.

    THE FIX. Abbreviate everything first, then look for labels used more than once. Only those get a
    discriminator appended - the distinctive tail of their config name:

        sea_gated-last__sea_cls_conj
        sea_gated-max__sea_cls_conj
        sea_gated-mean__sea_cls_conj

    So the common case stays short, and only the models that genuinely need more get more. This runs
    ONCE over every known model, so a given model has the same label in every figure of the paper.

    model_ids : every model that might appear in any figure.
    returns : {full id -> display label} (also stored for shorten() to use).
    """
    draft: Dict[str, str] = {m: _abbreviate(m) for m in model_ids}

    # which short labels are claimed by more than one model?
    counts: Dict[str, int] = {}
    for label in draft.values():
        counts[label] = counts.get(label, 0) + 1

    # group the clashing models so each group can be given the smallest discriminator that works
    groups: Dict[str, List[str]] = {}
    for model_id, label in draft.items():
        if counts.get(label, 0) > 1:
            groups.setdefault(label, []).append(model_id)

    final: Dict[str, str] = dict(draft)
    for label, members in groups.items():
        enc, pool = label.split("__", 1)
        for model_id, extra in zip(members, _discriminators(members)):
            final[model_id] = f"{enc}-{extra}__{pool}"

    _LABEL_MAP.clear()
    _LABEL_MAP.update(final)
    return final


def _discriminators(model_ids: List[str]) -> List[str]:
    """
    For a group of models that abbreviate to the SAME label, find the shortest thing that tells
    them apart.

    The idea: take each model's config name, split it into words, and throw away every word the
    whole group shares. What is left is exactly what makes that model different.

        gated_last_topk                       every member has "gated" and "topk",
        gated_max_topk        ->  last / max / mean      so only the middle word survives
        gated_mean_topk

    A member whose words are ALL shared (e.g. plain "softmax" inside a group that also holds
    "slim_softmax") has nothing left over, so it keeps its full config name instead of an empty
    string.

    model_ids : the clashing model ids.
    returns : one discriminator per model, in the same order.
    """
    configs = [str(m).split("__")[0] for m in model_ids]
    configs = [c[len("seanet_"):] if c.startswith("seanet_") else c for c in configs]
    token_sets = [set(c.split("_")) for c in configs]
    shared = set.intersection(*token_sets) if token_sets else set()

    out = []
    for config in configs:
        distinct = [word for word in config.split("_") if word not in shared]
        out.append("_".join(distinct) if distinct else config)
    return out


def shorten(model_id: str, max_len: int = 34, mark_reused: bool = False) -> str:
    """
    The display label for a model: "<short encoder>__<short pooling>".

        seanet_spiketrend__sea_mstcn_sep_spiketrend__sea_classwise_conjunctive
        -> sea_spiketrend__sea_cls_conj

    Reading it needs no legend: "sea_" is ours, "mil_" is MILLET's, reused unchanged. So
    "sea_mstcn__mil_conj" says our encoder with the paper's pooling head, in the label itself. That
    is why the old dagger marker is gone - the name already carries the information.

    model_id : the full id.
    max_len : longest label allowed; longer ones are cut in the MIDDLE, keeping both ends, because
              the ends are the encoder and the pooling and the middle is just the separator.
    mark_reused : kept so older calls do not break; the prefixes make it unnecessary.
    returns : the display label.
    """
    if model_id in _LABEL_MAP:                      # the collision-resolved label, if we built one
        label = _LABEL_MAP[model_id]
    else:
        label = _abbreviate(model_id)
    if len(label) > max_len:
        keep = (max_len - 1) // 2
        label = f"{label[:keep]}…{label[-keep:]}"
    return label


def parts_label(model_id: str) -> str:
    """
    The full "encoder + pooling" description of a model, for a caption or a table - not an axis.

    Example: "sea_mstcn_sep_bottleneck + sea_topk_conjunctive". Too wide for a tick label, which is
    why the axes use shorten(); but exactly what a caption should spell out so the reader knows what
    the nickname stands for.
    """
    parts = str(model_id).split("__")
    if len(parts) < 3:
        return str(model_id)
    return f"{parts[1]} + {parts[2]}"


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
