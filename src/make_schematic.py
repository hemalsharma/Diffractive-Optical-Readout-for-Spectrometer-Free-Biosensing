"""
make_schematic.py -- 3D-style system schematic for the minimal Stage 1 report.

    python src/make_schematic.py   ->  results/fig_schematic.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent.parent / "results"
DX, DY = 1.1, 0.75          # oblique "depth" direction of each transverse plane
H = 1.25                    # half height of the DOE / detector planes


def plane_pt(x0, t, s):
    """Point on a vertical plane at axial position x0.
    t in [0, 1]: position along the plane's depth (transverse x); s in [-1, 1]: height (transverse y)."""
    return np.array([x0 + t * DX, s * H + t * DY])


def plane(ax, x0, fc, ec="#444", alpha=1.0, z=1):
    pts = [plane_pt(x0, 0, -1), plane_pt(x0, 0, 1), plane_pt(x0, 1, 1), plane_pt(x0, 1, -1)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.9, alpha=alpha, zorder=z))


def label(ax, x, y, title, sub, colour="#222"):
    ax.text(x, y, title, ha="center", va="top", fontsize=10.5, weight="bold", color=colour)
    ax.text(x, y - 0.32, sub, ha="center", va="top", fontsize=9.5, color="#333")


fig, ax = plt.subplots(figsize=(13.5, 4.4))
ax.set_xlim(-0.5, 15.4)
ax.set_ylim(-2.75, 2.6)
ax.set_aspect("equal")
ax.axis("off")

yc = 0.5 * DY                                  # beam height at the plane centres
beam_c = "#f2994a"

# --- broadband source ------------------------------------------------------
ax.add_patch(Rectangle((0.0, yc - 0.35), 1.0, 0.7, fc="#e8e8e8", ec="#444", lw=0.9, zorder=3))
ax.text(0.5, yc, "SLED", ha="center", va="center", fontsize=9, zorder=4)
label(ax, 0.5, -1.6, "Broadband source", "S$_{source}$(λ)")

# --- resonant sensor chip ----------------------------------------------------
chip = [(2.45, yc - 0.55), (3.75, yc - 0.55), (4.05, yc - 0.3), (2.75, yc - 0.3)]
ax.add_patch(Polygon(chip, fc="#cfe8df", ec="#2f6f5a", lw=0.9, zorder=3))
for xm in np.linspace(2.7, 3.7, 6):                                  # bound analyte
    ax.add_patch(Ellipse((xm + 0.12, yc - 0.33), 0.1, 0.07, fc="#2f6f5a", zorder=4))
ax.add_patch(Rectangle((2.55, yc + 0.35), 1.35, 0.9, fc="white", ec="#2f6f5a", lw=0.8, zorder=3))
lam = np.linspace(-1, 1, 200)
for shift, col in ((0.0, "#2f6f5a"), (0.22, "#c0392b")):
    T = 1 - 0.8 / (1 + ((lam - shift) / 0.12) ** 2)
    ax.plot(2.65 + 0.575 * (lam + 1), yc + 0.43 + 0.72 * T, color=col, lw=1.1, zorder=4)
ax.text(3.22, yc + 1.33, "λ$_r$ shifts with c", ha="center", fontsize=8.5, color="#2f6f5a")
label(ax, 3.25, -1.6, "Resonant sensor", "analyte c  →  S(λ; c)", "#2f6f5a")

# beam: source -> sensor -> DOE centre (0.5 depth)
x_doe = 6.1
p_doe = plane_pt(x_doe, 0.5, 0)
ax.add_patch(Polygon([(1.0, yc - 0.18), (1.0, yc + 0.18), (p_doe[0], p_doe[1] + 0.3),
                      (p_doe[0], p_doe[1] - 0.3)], fc=beam_c, ec="none", alpha=0.55, zorder=2))

# --- DOE plane with grating stripes ------------------------------------------
plane(ax, x_doe, "#f4f1fb", z=1)
cmap = plt.get_cmap("twilight")
n = 48
for i in range(n):
    t0, t1 = i / n, (i + 1) / n
    pts = [plane_pt(x_doe, t0, -1), plane_pt(x_doe, t0, 1), plane_pt(x_doe, t1, 1), plane_pt(x_doe, t1, -1)]
    ax.add_patch(Polygon(pts, fc=cmap((i % 8) / 8), ec="none", alpha=0.85, zorder=1.5))
plane(ax, x_doe, "none", z=1.6)
ax.add_patch(Ellipse(p_doe, 0.35, 0.62, fc=beam_c, ec="none", alpha=0.7, zorder=2.5))
label(ax, x_doe + DX / 2, -1.6, "Phase-only DOE", "t = exp[iφ(x, y)]", "#4b3f8f")

# --- propagation to the detector plane (deflected +1 order) ------------------
x_det = 10.9
p_spot = plane_pt(x_det, 0.72, 0)
ax.add_patch(Polygon([(p_doe[0], p_doe[1] + 0.3), (p_spot[0], p_spot[1] + 0.36),
                      (p_spot[0], p_spot[1] - 0.36), (p_doe[0], p_doe[1] - 0.3)],
                     fc=beam_c, ec="none", alpha=0.45, zorder=0.5))
ax.annotate("", xy=(x_det - 0.25, -1.05), xytext=(x_doe + DX + 0.25, -1.05),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=0.9))
ax.text((x_doe + DX + x_det) / 2, -0.95, "z = 50 mm", ha="center", va="bottom", fontsize=9, color="#555")
label(ax, (x_doe + DX + x_det) / 2, -1.6, "Propagation", "angular spectrum", "#4b3f8f")

# --- detector plane with A / B -----------------------------------------------
plane(ax, x_det, "#fbfbfb", z=1)
ax.add_patch(Ellipse(p_spot, 0.42, 0.72, fc=beam_c, ec="none", alpha=0.85, zorder=2))
for t0, t1, lab in ((0.56, 0.72, "A"), (0.72, 0.88, "B")):
    pts = [plane_pt(x_det, t0, -0.42), plane_pt(x_det, t0, 0.42),
           plane_pt(x_det, t1, 0.42), plane_pt(x_det, t1, -0.42)]
    ax.add_patch(Polygon(pts, fc="none", ec="#1f6fb2", lw=1.3, zorder=3))
    c = (plane_pt(x_det, t0, 0.42) + plane_pt(x_det, t1, 0.42)) / 2
    ax.text(c[0], c[1] + 0.12, lab, ha="center", va="bottom", fontsize=10, weight="bold", color="#1f6fb2")
label(ax, x_det + DX / 2, -1.6, "Detector plane", "I(x, y; c) → I$_A$, I$_B$", "#1f6fb2")

# --- readout -----------------------------------------------------------------
ax.add_patch(FancyArrowPatch((x_det + DX + 0.15, yc), (x_det + DX + 1.1, yc),
                             arrowstyle="-|>", mutation_scale=14, color="#333"))
ax.text(14.3, yc + 0.02, r"$R=\dfrac{I_A-I_B}{I_A+I_B}$", ha="center", va="center", fontsize=12.5)
label(ax, 14.3, -1.6, "Readout", "no spectrometer")

fig.savefig(OUT / "fig_schematic.png", dpi=170, bbox_inches="tight")
print("saved", OUT / "fig_schematic.png")
