"""Shared style for the paper figures: Times-like fonts matching newtx, print-size lettering (>= 6 pt),
TrueType embedding, and one name / colour / line style per family, used in all figures and tables."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
})

# key: (name as in the tables, colour, line style, marker)
FAMILY = {
    "gauss": ("Gauss disks (single)", "0.45", "none", "."),
    "gauss_sup": ("sup-closure of Gauss disks", "k", "-", None),
    "octagon": ("octagons", "C3", "--", None),
    "adams": ("Bresenham/DSS, 8 dir.", "C1", ":", None),
    "periodic": ("periodic lines, 8 dir.", "C4", "-.", None),
    "ours": ("periodic lines, 13 dir.", "C0", "-", None),
    "dss_milp": ("DSS chain, MILP", "C2", "-", "x"),
}
