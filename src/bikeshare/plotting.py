"""Shared chart style, so every figure in the notebooks, README and app looks the same.

Colours come from a colour-blind-checked palette; categorical colours are used in fixed order.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#c3c2b7"  # grey for marks that are context, not the story

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]

SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# for coloured marks on the light surface: starts dark enough to stay visible
ORDINAL = LinearSegmentedColormap.from_list("blue_marks", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#86b6ef", "#f0efec", "#f19a99", "#c23a39"])

thousands = FuncFormatter(lambda v, _: f"{v / 1000:g}K" if abs(v) >= 1000 else f"{v:g}")
percent = FuncFormatter(lambda v, _: f"{v:.0%}")


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, name):
    """Save a figure for the README under reports/figures/."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
