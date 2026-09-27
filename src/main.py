"""
main.py -- complete Stage 1 forward calculation, c -> detector signals.

    c -> S(lambda;c) -> E_in -> t = exp(i phi) -> angular spectrum -> I(x,y;c)
      -> detector powers I_A, I_B -> R

The DOE here is the analytical linear phase grating used for validation;
no optimization is performed in Stage 1. Figures and the detector table
(final_detector_table.csv) go to results/.

    python src/main.py
"""

import csv
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)   # torch.jit notice on Python 3.14

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import CFG
from detector import (broadband_intensity, centroid, cross_section,
                      differential_readout, monochromatic_intensity,
                      rect_mask, region_power)
from doe import linear_grating
from propagation import (apply_doe, embed_phase, gaussian_beam, grid_for_doe,
                         power)

OUT = Path(__file__).resolve().parent.parent / "results"
UM, MM = 1e6, 1e3


def savefig(fig, name, **kw):
    """Save to results/, retrying if another process briefly locks the file."""
    for attempt in range(5):
        try:
            fig.savefig(OUT / name, dpi=160, **kw)
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5)
    plt.close(fig)


def spot_position(lam, cfg=CFG):
    """Grating-equation position of the +1 order at the detector plane."""
    return cfg.z * math.tan(math.asin(lam / cfg.period))


def _trapezoid_weights_nm(lam_m):
    """Quadrature weights for a strictly increasing wavelength grid."""
    if lam_m.numel() < 2:
        raise ValueError("dlam_nm is required when only one wavelength is supplied")
    steps = lam_m[1:] - lam_m[:-1]
    if not bool((steps > 0).all()):
        raise ValueError("lam_m must be strictly increasing")
    weights = torch.empty_like(lam_m)
    weights[0] = steps[0] / 2
    weights[-1] = steps[-1] / 2
    if lam_m.numel() > 2:
        weights[1:-1] = (steps[:-1] + steps[1:]) / 2
    return weights * 1e9


def forward_from_spectrum(lam_m, S, phi_doe, cfg=CFG, dlam_nm=None,
                          upsample=None, pad=None, chunk=None):
    """Propagate supplied spectra through a supplied DOE phase map.

    Parameters
    ----------
    lam_m : (Nl,) tensor-like
        Vacuum wavelengths in metres, in strictly increasing order.
    S : (Nl,) or (Nc, Nl) tensor-like
        Spectral density per nanometre. A leading row represents one
        concentration or experimental condition.
    phi_doe : (cfg.n_doe, cfg.n_doe) tensor-like
        Phase delay in radians on the physical DOE pixel grid.
    dlam_nm : float or (Nl,) tensor-like, optional
        Wavelength-integration step or quadrature weights in nanometres.
        If omitted, composite-trapezoid weights are derived from lam_m.

    Returns the detector-plane intensity ``I``, integrated detector powers
    ``IA`` and ``IB``, total grid power ``P_total``, and normalized readout
    ``R``, together with the intermediate fields and detector masks.
    """
    lam_m = torch.as_tensor(lam_m, dtype=torch.float64)
    S = torch.as_tensor(S, dtype=torch.float64)
    phi_doe = torch.as_tensor(phi_doe, dtype=torch.float64)
    if lam_m.ndim != 1 or lam_m.numel() == 0:
        raise ValueError("lam_m must be a non-empty 1-D wavelength grid")
    if S.ndim not in (1, 2) or S.shape[-1] != lam_m.numel():
        raise ValueError("S must have shape (Nl,) or (Nc, Nl), matching lam_m")
    if not bool(torch.isfinite(lam_m).all()) or not bool((lam_m > 0).all()):
        raise ValueError("lam_m must contain only finite, positive wavelengths")
    if lam_m.numel() > 1 and not bool((lam_m[1:] > lam_m[:-1]).all()):
        raise ValueError("lam_m must be strictly increasing")
    expected = (cfg.n_doe, cfg.n_doe)
    if tuple(phi_doe.shape) != expected:
        raise ValueError(f"phi_doe must have shape {expected}, got {tuple(phi_doe.shape)}")
    if not bool(torch.isfinite(phi_doe).all()):
        raise ValueError("phi_doe must contain only finite values")

    weights_nm = _trapezoid_weights_nm(lam_m) if dlam_nm is None else dlam_nm
    s = cfg.upsample if upsample is None else upsample
    p = cfg.pad if pad is None else pad
    if not isinstance(s, int) or s <= 0 or not isinstance(p, int) or p <= 0:
        raise ValueError("upsample and pad must be positive integers")
    grid = grid_for_doe(cfg.n_doe, cfg.pitch, s, p)
    phi = embed_phase(phi_doe, grid, s)
    E_in = gaussian_beam(grid, cfg.w0)
    E_doe = apply_doe(E_in, phi)
    if chunk is None:
        chunk = max(1, int(16 * (512 / grid.n) ** 2))
    I = broadband_intensity(E_doe, grid, lam_m, cfg.z, S, weights_nm, chunk)

    x_spot = spot_position(cfg.lambda0_nm * 1e-9, cfg)
    mA = rect_mask(grid, x_spot - cfg.det_width / 2, 0, cfg.det_width, cfg.det_height)
    mB = rect_mask(grid, x_spot + cfg.det_width / 2, 0, cfg.det_width, cfg.det_height)
    IA, IB = region_power(I, mA, grid), region_power(I, mB, grid)
    P_total = I.sum((-2, -1)) * grid.cell_area
    return dict(grid=grid, phi=phi, S=S, E_in=E_in, E_doe=E_doe,
                lam_m=lam_m, I=I, IA=IA, IB=IB, P_total=P_total,
                R=differential_readout(IA, IB), xc=centroid(I, grid)[0],
                x_spot=x_spot, masks=(mA, mB))


def forward(c, cfg=CFG, phi_doe=None, n_lambda=None, upsample=None, pad=None):
    """Full forward model. Returns a dict with every intermediate quantity."""
    if phi_doe is None:
        phi_doe = linear_grating(cfg.n_doe, cfg.pitch, cfg.period)

    sensor = cfg.sensor(n_lambda)
    c = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
    S = sensor.spectrum(c)                                  # (Nc, Nl)
    lam_m = cfg.lam_m(sensor)
    result = forward_from_spectrum(lam_m, S, phi_doe, cfg,
                                   dlam_nm=cfg.dlam_nm(sensor),
                                   upsample=upsample, pad=pad)
    result.update(sensor=sensor, c=c)
    return result


# ----------------------------------------------------------------------------
# Report figures
# ----------------------------------------------------------------------------

def fig_concept():
    """Block diagram of the forward model."""
    boxes = [("Concentration", "c"),
             ("Sensor spectrum", "S(λ; c)"),
             ("Phase-only DOE", "t = exp[iφ(x, y)]"),
             ("Propagation", "angular spectrum, per λ"),
             ("Detectors", r"$I_A,\ I_B\ \rightarrow\ R$")]
    fig, ax = plt.subplots(figsize=(13.5, 2.1))
    ax.set(xlim=(0, 13.5), ylim=(0, 2.1))
    ax.axis("off")
    w, h, gap = 2.2, 1.3, 0.5
    for i, (title, sub) in enumerate(boxes):
        x0 = 0.15 + i * (w + gap)
        colour = "#e1f5ee" if i < 2 else "#eeedfe"
        ax.add_patch(FancyBboxPatch((x0, 0.4), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=colour, ec="#555", lw=0.8))
        ax.text(x0 + w / 2, 0.4 + h * 0.64, title, ha="center", va="center", fontsize=11,
                weight="bold")
        ax.text(x0 + w / 2, 0.4 + h * 0.30, sub, ha="center", va="center", fontsize=10)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.04, 0.4 + h / 2),
                                         (x0 + w + gap - 0.04, 0.4 + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#333"))
    savefig(fig, "fig_concept.png", bbox_inches="tight")


def fig_sensor(cfg=CFG):
    sensor = cfg.sensor(n_lambda=2001)
    nm = (sensor.lam * cfg.lambda0_nm).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for c in (0.0, 0.5, 1.0, 1.5, 2.0):
        ax[0].plot(nm, sensor.transmission(c).numpy(), label=f"c = {c}")
    ax[0].set(xlim=(1535, 1570), xlabel="wavelength [nm]", ylabel="T(λ; c)",
              title="(a) Resonance dip shifts with concentration")
    ax[0].legend(fontsize=8)
    ax[1].plot(nm, sensor.source.numpy(), "k--", label="S_source")
    for c in (0.0, 1.0, 2.0):
        ax[1].plot(nm, sensor.spectrum(c).numpy(), label=f"S(λ; c = {c})")
    ax[1].set(xlabel="wavelength [nm]", ylabel="S(λ; c) [a.u.]",
              title="(b) Sensor output spectrum")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_sensor.png")


def fig_doe(cfg=CFG):
    phi = linear_grating(cfg.n_doe, cfg.pitch, cfg.period)
    ext = [-cfg.n_doe / 2 * cfg.pitch * MM, cfg.n_doe / 2 * cfg.pitch * MM] * 2
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(phi.numpy(), extent=ext, origin="lower", cmap="twilight")
    fig.colorbar(im, ax=ax, label="φ [rad]")
    ax.set(xlabel="x [mm]", ylabel="y [mm]",
           title=f"Linear phase grating, d = {cfg.period * UM:.0f} µm")
    fig.tight_layout()
    savefig(fig, "fig_doe.png")


def fig_input(r, cfg=CFG):
    g = r["grid"]
    x = g.x * UM
    z = 600
    sel = (x.abs() <= z)
    ext = [-z, z, -z, z]
    Iin = (r["E_in"].abs() ** 2)[sel][:, sel]
    ph = torch.angle(r["E_doe"])[sel][:, sel]
    ph = torch.where(Iin > 1e-3 * Iin.max(), torch.remainder(ph, 2 * math.pi),
                     torch.tensor(float("nan"), dtype=torch.float64))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(Iin.numpy(), extent=ext, origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax[0], label="|E_in|²")
    ax[0].set(xlabel="x [µm]", ylabel="y [µm]",
              title=f"(a) Gaussian input, w0 = {cfg.w0 * UM:.0f} µm")
    im = ax[1].imshow(ph.numpy(), extent=ext, origin="lower", cmap="twilight")
    fig.colorbar(im, ax=ax[1], label="arg E_0+ [rad]")
    ax[1].set(xlabel="x [µm]", ylabel="y [µm]", title="(b) Phase of the field after the DOE")
    fig.tight_layout()
    savefig(fig, "fig_input.png")


def fig_propagation(r, cfg=CFG):
    g = r["grid"]
    lam0 = cfg.lambda0_nm * 1e-9
    I_free = monochromatic_intensity(r["E_in"], g, lam0, cfg.z)
    I_doe = monochromatic_intensity(r["E_doe"], g, lam0, cfg.z)
    x = g.x * MM
    ext = [float(x[0]), float(x[-1])] * 2
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, I, t in ((ax[0], I_free, "(a) No DOE (φ = 0)"),
                    (ax[1], I_doe, "(b) Linear grating")):
        im = a.imshow(I.numpy(), extent=ext, origin="lower", cmap="inferno")
        fig.colorbar(im, ax=a, label="|E|²")
        a.axvline(r["x_spot"] * MM, color="w", ls=":", lw=0.8)
        a.set(xlabel="x [mm]", ylabel="y [mm]", title=f"{t}, λ = 1550 nm, z = {cfg.z * MM:.0f} mm")
    ax[2].plot(x.numpy(), cross_section(I_free, g).numpy(), label="no DOE")
    ax[2].plot(x.numpy(), cross_section(I_doe, g).numpy(), label="linear grating")
    ax[2].axvline(r["x_spot"] * MM, color="gray", ls=":", label="grating equation")
    ax[2].set(xlabel="x [mm]", ylabel="|E(x, 0)|²", title="(c) Cross-section at y = 0")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_propagation.png")


def _spot_window(r, half=500):
    g = r["grid"]
    x = g.x * UM
    x0 = r["x_spot"] * UM
    return x, x0, (x - x0).abs() <= half, x.abs() <= half, [x0 - half, x0 + half, -half, half]


def fig_broadband(r):
    g = r["grid"]
    x, x0, sel, sely, ext = _spot_window(r)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(r["I"][0][sely][:, sel].numpy(), origin="lower", cmap="inferno", extent=ext)
    fig.colorbar(im, ax=ax[0], label="I(x, y; c = 0)")
    for m, lab in zip(r["masks"], "AB"):
        cols = torch.nonzero(m.any(0)).squeeze()
        rows = torch.nonzero(m.any(1)).squeeze()
        xa, xb = float(x[cols[0]]), float(x[cols[-1]])
        ya, yb = float(x[rows[0]]), float(x[rows[-1]])
        ax[0].add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ec="c", lw=1))
        ax[0].text((xa + xb) / 2, yb + 15, lab, color="c", ha="center")
    ax[0].set(xlabel="x [µm]", ylabel="y [µm]", title="(a) Broadband detector image, c = 0")
    for k in (0, len(r["c"]) // 2, len(r["c"]) - 1):
        ax[1].plot(x[sel].numpy(), cross_section(r["I"][k], g)[sel].numpy(),
                   label=f"c = {float(r['c'][k]):.1f}")
    ax[1].set(xlabel="x [µm]", ylabel="I(x, 0; c)", title="(b) Cross-sections for several c")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_broadband.png")


def fig_difference(r):
    """Appendix: power-normalized difference image, I(c_max) - I(0)."""
    x, x0, sel, sely, ext = _spot_window(r)
    I0, I1 = r["I"][0], r["I"][-1]
    dI = (I1 / I1.sum() - I0 / I0.sum()) / (I0 / I0.sum()).max()
    v = float(dI.abs().max())
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(dI[sely][:, sel].numpy(), origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, extent=ext)
    fig.colorbar(im, ax=ax, label="Δ(I / ΣI) / max")
    ax.set(xlabel="x [µm]", ylabel="y [µm]",
           title=f"Normalized I(c = {float(r['c'][-1]):.0f}) − I(c = 0)")
    fig.tight_layout()
    savefig(fig, "fig_difference.png")


def fig_readout(r):
    c = r["c"].numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ax[0].plot(c, ((r["R"] - r["R"][0]) * 1e3).numpy(), "o-")
    ax[0].set(xlabel="concentration c", ylabel="ΔR = R(c) − R(0)  [×10⁻³]",
              title="(a) Differential detector readout")
    ax[1].plot(c, ((r["xc"] - r["xc"][0]) * 1e9).numpy(), "o-", color="C1")
    ax[1].set(xlabel="concentration c", ylabel="centroid shift [nm]",
              title="(b) Spot-centroid shift")
    fig.tight_layout()
    savefig(fig, "fig_readout.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CFG
    r = forward(torch.linspace(0, cfg.c_max, 9, dtype=torch.float64))
    g = r["grid"]
    print(f"simulation grid {g.n} x {g.n}, dx = {g.dx * UM:.2f} um, "
          f"window {g.length * MM:.2f} mm, {r['lam_m'].numel()} wavelengths "
          f"{r['lam_m'][0] * 1e9:.1f}-{r['lam_m'][-1] * 1e9:.1f} nm")
    print(f"+1 order at x = {r['x_spot'] * UM:.1f} um (grating equation)")
    print(f"input power {float(power(r['E_in'], g)):.6e}, after DOE "
          f"{float(power(r['E_doe'], g)):.6e}")
    print(" c    I_A/I_A0   I_B/I_B0   dR = R - R(0)   centroid shift [nm]")
    for k in range(len(r["c"])):
        print(f"{float(r['c'][k]):4.2f} {float(r['IA'][k] / r['IA'][0]):9.6f} "
              f"{float(r['IB'][k] / r['IB'][0]):9.6f}  {float(r['R'][k] - r['R'][0]):+.4e} "
              f"{float((r['xc'][k] - r['xc'][0]) * 1e9):+10.2f}")
    with open(OUT / "final_detector_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["c", "resonance_shift_nm", "I_A_over_I_A0", "I_B_over_I_B0",
                    "R", "delta_R", "centroid_shift_nm"])
        for k in range(len(r["c"])):
            w.writerow([f"{float(r['c'][k]):.2f}",
                        f"{float(r['c'][k]) * cfg.shift_per_c_nm:.4f}",
                        f"{float(r['IA'][k] / r['IA'][0]):.8f}",
                        f"{float(r['IB'][k] / r['IB'][0]):.8f}",
                        f"{float(r['R'][k]):.8e}", f"{float(r['R'][k] - r['R'][0]):.8e}",
                        f"{float((r['xc'][k] - r['xc'][0]) * 1e9):.4f}"])
    fig_concept()
    fig_sensor()
    fig_doe()
    fig_input(r)
    fig_propagation(r)
    fig_broadband(r)
    fig_difference(r)
    fig_readout(r)
    print(f"Forward model complete: 8 figures and final_detector_table.csv saved to {OUT}")
