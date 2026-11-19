"""Shared figure style. Importing this module sets the style for every figure."""

import matplotlib as mpl

# Categorical colours, always taken in this order (classes 0, 1, 2 and the three methods of P4).
# Checked for colour-blind separation on a light surface; aqua is below 3:1 contrast, so every
# figure that uses it carries a legend and a marker shape of its own.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    }
)
