"""
validation.py -- Stage 1 validation tests (plan section 2.7).

  1. Free propagation      phi = 0 reproduces analytic Gaussian-beam spreading
  2. Linear grating        +1-order position follows the grating equation;
                           wavelength sweep changes the output smoothly
  3. Fraunhofer model      far-field model converges to the angular spectrum
  4. Diffraction orders    binary and staircase gratings give the analytic
                           order efficiencies
  5. Energy accounting     a lossless phase mask creates no power
  6. Sampling convergence  wavelength samples, simulation grid, detector pixels

Each test is checked against a pass threshold. Figures and validation.json
go to results/.

    python src/validation.py
"""

import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)   # torch.jit notice on Python 3.14

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import CFG
from detector import (bin_pixels, centroid, cross_section,
                      monochromatic_intensity, second_moment_radius)
from doe import binary_grating, linear_grating, transmission, zero_phase
from main import forward, savefig, spot_position
from propagation import (SimGrid, angular_spectrum, apply_doe,
                         band_limited_fraction, embed_phase, fraunhofer,
                         gaussian_beam, grid_for_doe, power)

OUT = Path(__file__).resolve().parent.parent / "results"
UM, MM = 1e6, 1e3
LAM0 = CFG.lambda0_nm * 1e-9
RESULTS = {}


def setup(phi_doe, s=CFG.upsample, pad=CFG.pad):
    g = grid_for_doe(CFG.n_doe, CFG.pitch, s, pad)
    E_in = gaussian_beam(g, CFG.w0)
    return g, E_in, apply_doe(E_in, embed_phase(phi_doe, g, s))


# ----------------------------------------------------------------------------
# 1. Free propagation
# ----------------------------------------------------------------------------

def test_free_propagation():
    g, E_in, E0 = setup(zero_phase(CFG.n_doe))
    zR = math.pi * CFG.w0**2 / LAM0
    zs = torch.linspace(0, 150e-3, 16, dtype=torch.float64)
    w_num, w_th, peak_err = [], [], []
    x = g.x
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    for z in zs:
        I = monochromatic_intensity(E0, g, LAM0, float(z))
        w = CFG.w0 * math.sqrt(1 + (float(z) / zR) ** 2)
        I_th = (CFG.w0 / w) ** 2 * torch.exp(-2 * r2 / w**2)
        w_num.append(float(second_moment_radius(I, g)))
        w_th.append(w)
        peak_err.append(float((I - I_th).abs().max() / I_th.max()))
    rel = max(abs(a - b) / b for a, b in zip(w_num, w_th))
    RESULTS["free_propagation"] = dict(z_R_mm=zR * MM, max_rel_err_w=rel,
                                       max_rel_err_intensity=max(peak_err))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    zz = torch.linspace(0, 150e-3, 200, dtype=torch.float64)
    ax.plot((zz * MM).numpy(), (CFG.w0 * torch.sqrt(1 + (zz / zR) ** 2) * UM).numpy(),
            label="analytic w(z) = w0 √(1 + (z/zR)²)")
    ax.plot((zs * MM).numpy(), [v * UM for v in w_num], "o", mfc="none",
            label="angular-spectrum simulation")
    ax.set(xlabel="z [mm]", ylabel="beam radius w [µm]",
           title=f"Free Gaussian propagation, zR = {zR * MM:.1f} mm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "val_free.png")


# ----------------------------------------------------------------------------
# 2. Linear grating: grating equation and wavelength sweep
# ----------------------------------------------------------------------------

def test_grating_equation():
    g, _, E0 = setup(linear_grating(CFG.n_doe, CFG.pitch, CFG.period))
    lam = torch.linspace(1450e-9, 1650e-9, 21, dtype=torch.float64)
    I = monochromatic_intensity(E0, g, lam, CFG.z)
    xc = centroid(I, g)[0]
    x_th = torch.tensor([spot_position(float(l)) for l in lam], dtype=torch.float64)
    err = (xc - x_th).abs()
    theta = torch.asin(lam / CFG.period)
    slope_th = CFG.z / (CFG.period * torch.cos(theta) ** 3)      # dx/dlambda
    slope_num = torch.gradient(xc, spacing=(lam,))[0]
    RESULTS["grating_equation"] = dict(
        lambda_nm=[1450, 1650], x_range_um=[float(xc[0] * UM), float(xc[-1] * UM)],
        max_abs_err_um=float(err.max() * UM),
        dispersion_um_per_nm_at_1550=float(slope_th[10] * 1e-3),
        dispersion_rel_err=float(((slope_num - slope_th) / slope_th)[1:-1].abs().max()),
        shift_over_5nm_um=float(slope_th[10] * 5e-9 * UM))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot((lam * 1e9).numpy(), (x_th * UM).numpy(), label="grating equation z tan(asin(λ/d))")
    ax.plot((lam * 1e9).numpy(), (xc * UM).numpy(), "o", mfc="none",
            label="simulated +1-order position")
    ax.set(xlabel="λ [nm]", ylabel="spot position at detector [µm]",
           title="Linear grating: wavelength → position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "val_grating.png")


# ----------------------------------------------------------------------------
# 2b. Fraunhofer model: far-field limit of the angular spectrum
# ----------------------------------------------------------------------------

def test_fraunhofer():
    # (a) Gaussian beam, no DOE, on a wide window so the far field fits.
    g = SimGrid(n=1024, dx=20e-6)
    E0 = gaussian_beam(g, CFG.w0)
    zR = math.pi * CFG.w0**2 / LAM0
    zs = [0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    errs, profiles = [], {}
    for z in zs:
        I_as = monochromatic_intensity(E0, g, LAM0, z)
        I_f = monochromatic_intensity(E0, g, LAM0, z, method="fraunhofer")
        # The sampled far field repeats every lambda z / dx; compare only
        # inside the unaliased region.
        xlim = min(g.length / 2, LAM0 * z / (2 * g.dx))
        m = (g.x.abs() < xlim)
        sel = m[:, None] & m[None, :]
        errs.append(float(torch.linalg.norm((I_f - I_as)[sel]) / torch.linalg.norm(I_as[sel])))
        if z in (0.05, 1.0):
            w = CFG.w0 * math.sqrt(1 + (z / zR) ** 2)
            profiles[z] = (w, cross_section(I_as, g), cross_section(I_f, g))
    # (b) linear grating on the design grid: Fraunhofer gives the paraxial
    #     position x = lambda z / d; the exact position is z tan(asin(lambda/d)).
    gd, _, Ed = setup(linear_grating(CFG.n_doe, CFG.pitch, CFG.period))
    lam = torch.linspace(1450e-9, 1650e-9, 21, dtype=torch.float64)
    xc = centroid(monochromatic_intensity(Ed, gd, lam, CFG.z, method="fraunhofer"), gd)[0]
    x_par = lam * CFG.z / CFG.period
    x_ex = torch.tensor([spot_position(float(l)) for l in lam], dtype=torch.float64)
    RESULTS["fraunhofer"] = dict(
        z_over_zR=[z / zR for z in zs], rel_L2_error=errs,
        grating_max_err_vs_paraxial_nm=float((xc - x_par).abs().max() * 1e9),
        grating_max_diff_vs_exact_nm=float((xc - x_ex).abs().max() * 1e9))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    zz = [z / zR for z in zs]
    ax[0].loglog(zz, errs, "o-", label="‖I_Fraunhofer − I_AS‖ / ‖I_AS‖")
    ax[0].axvline(CFG.z / zR, color="gray", ls=":", label=f"z = {CFG.z * MM:.0f} mm (this work)")
    ax[0].set(xlabel="z / z_R", ylabel="relative difference",
              title="(a) Fraunhofer approaches the exact result for z ≫ z_R")
    ax[0].legend(fontsize=8)
    for z, col in ((0.05, "C0"), (1.0, "C1")):
        w, a, f = profiles[z]
        u = (g.x / w).numpy()
        ax[1].plot(u, (a / a.max()).numpy(), col, label=f"angular spectrum, z = {z * MM:.0f} mm")
        ax[1].plot(u, (f / a.max()).numpy(), col, ls="--", label=f"Fraunhofer, z = {z * MM:.0f} mm")
    ax[1].set(xlim=(-2.5, 2.5), xlabel="x / w(z)", ylabel="intensity (normalized to exact peak)",
              title="(b) Beam profiles")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "val_fraunhofer.png")


# ----------------------------------------------------------------------------
# 3. Diffraction-order efficiencies
# ----------------------------------------------------------------------------

def order_powers(E0, g, orders):
    """Fraction of the angular spectrum in each grating order (along kx)."""
    A2 = torch.fft.fft2(E0).abs() ** 2
    fx = torch.fft.fftfreq(g.n, d=g.dx, dtype=torch.float64)
    f_g = 1.0 / CFG.period
    tot = A2.sum()
    out = {}
    for m in orders:
        sel = (fx - m * f_g).abs() < f_g / 2
        out[m] = float(A2[:, sel].sum() / tot)
    return out


def test_orders():
    orders = [-3, -1, 0, 1, 3]
    binary_th = {m: (0.0 if m % 2 == 0 else (2 / (m * math.pi)) ** 2) for m in orders}
    stair_th = (math.sin(math.pi / 16) / (math.pi / 16)) ** 2
    res = {"binary_theory": binary_th, "staircase_+1_theory": stair_th, "by_upsample": {}}
    for s in (1, 2, 4):
        g, _, Eb = setup(binary_grating(CFG.n_doe, CFG.pitch, CFG.period), s=s)
        _, _, El = setup(linear_grating(CFG.n_doe, CFG.pitch, CFG.period), s=s)
        res["by_upsample"][s] = dict(binary=order_powers(Eb, g, orders),
                                     staircase_plus1=order_powers(El, g, [1])[1])
    RESULTS["orders"] = res                     # numbers only; no report figure


# ----------------------------------------------------------------------------
# 4. Energy accounting
# ----------------------------------------------------------------------------

def test_energy():
    rows = {}
    for name, phi in (("no DOE", zero_phase(CFG.n_doe)),
                      ("linear grating", linear_grating(CFG.n_doe, CFG.pitch, CFG.period)),
                      ("binary grating", binary_grating(CFG.n_doe, CFG.pitch, CFG.period))):
        g, E_in, E0 = setup(phi)
        P_in = float(power(E_in, g))
        P_doe = float(power(E0, g))
        P_z = float(power(angular_spectrum(E0, g, LAM0, CFG.z, band_limit=False), g))
        P_bl = float(power(angular_spectrum(E0, g, LAM0, CFG.z, band_limit=True), g))
        lost = float(band_limited_fraction(E0, g, LAM0, CFG.z))
        rows[name] = dict(doe_over_in=P_doe / P_in, prop_over_in=P_z / P_in,
                          bandlimited_over_in=P_bl / P_in, removed_fraction=lost,
                          closure=(P_bl / P_in + lost))
    # Broadband: detector-plane integral equals sum_k S_k dlam x P_in
    r = forward([0.0])
    g = r["grid"]
    P_det = float(r["I"][0].sum() * g.cell_area)
    P_exp = float(r["S"][0].sum()) * CFG.dlam_nm(r["sensor"]) * float(power(r["E_in"], g))
    rows["broadband (c = 0)"] = dict(detector_over_expected=P_det / P_exp)
    RESULTS["energy"] = rows


# ----------------------------------------------------------------------------
# 5. Convergence
# ----------------------------------------------------------------------------

def observables(r):
    """Stage-1 observables for c = [0, 1]."""
    return dict(dR=float(r["R"][1] - r["R"][0]),
                dx_nm=float((r["xc"][1] - r["xc"][0]) * 1e9),
                IA_ratio=float(r["IA"][1] / r["IA"][0]))


def test_convergence():
    c = [0.0, 1.0]
    # (a) wavelength samples
    Ns = [10, 20, 30, 50, 75, 100, 150, 200, 300, 600]
    obs_l = {n: observables(forward(c, n_lambda=n)) for n in Ns}
    ref = obs_l[Ns[-1]]
    # (b) simulation grid: samples per DOE pixel, window padding
    grids = [(1, 1), (2, 1), (4, 1), (2, 2)]
    obs_g = {f"s={s}, pad={p}": observables(forward(c, upsample=s, pad=p)) for s, p in grids}
    # (c) detector pixel size, from the s = 2 image
    r = forward(c)
    g = r["grid"]
    obs_d = {}
    for f in (1, 2, 4, 8, 16):
        Ib = bin_pixels(r["I"], f)
        gb = SimGrid(n=g.n // f, dx=g.dx * f)
        xc = centroid(Ib, gb)[0]
        obs_d[g.dx * f * UM] = float((xc[1] - xc[0]) * 1e9)

    RESULTS["convergence"] = dict(
        wavelength={n: o for n, o in obs_l.items()}, grid=obs_g,
        detector_pixel_um_to_dx_nm=obs_d)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for key, lab in (("dR", "ΔR"), ("dx_nm", "centroid shift")):
        e = [abs(obs_l[n][key] - ref[key]) / abs(ref[key]) for n in Ns[:-1]]
        ax.loglog(Ns[:-1], [max(v, 1e-16) for v in e], "o-", label=lab)
    ax.axvline(CFG.n_lambda, color="gray", ls=":", label=f"chosen Nλ = {CFG.n_lambda}")
    ax.set(xlabel="wavelength samples Nλ", ylabel="relative error vs Nλ = 600",
           title="Convergence with wavelength sampling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "val_convergence.png")


# ----------------------------------------------------------------------------
# Pass/fail thresholds
# ----------------------------------------------------------------------------

def checks():
    """(name, passed, detail) for every validation result."""
    R = RESULTS
    fp, ge, fr = R["free_propagation"], R["grating_equation"], R["fraunhofer"]
    o4 = R["orders"]["by_upsample"][4]
    th = R["orders"]["binary_theory"]
    cv = R["convergence"]
    wl = cv["wavelength"]
    rel150 = abs(wl[CFG.n_lambda]["dR"] / wl[600]["dR"] - 1)
    dRs = [v["dR"] for v in cv["grid"].values()]
    dxs = [v["dx_nm"] for v in cv["grid"].values()]
    pix = list(cv["detector_pixel_um_to_dx_nm"].values())
    en = R["energy"]
    closure = max(abs(v["closure"] - 1) for k, v in en.items() if "closure" in v)
    bb = abs(en["broadband (c = 0)"]["detector_over_expected"]
             - (1 - en["linear grating"]["removed_fraction"]))
    order_err = max(max(abs(o4["binary"][m] - th[m]) for m in (-1, 1, 3)),
                    abs(o4["staircase_plus1"] - R["orders"]["staircase_+1_theory"]))
    return [
        ("Free Gaussian propagation", max(fp["max_rel_err_w"], fp["max_rel_err_intensity"]) < 1e-4,
         f"max rel. error {max(fp['max_rel_err_w'], fp['max_rel_err_intensity']):.1e} (limit 1e-4)"),
        ("Grating equation, 1450-1650 nm", ge["max_abs_err_um"] * 1e3 < 10 and ge["dispersion_rel_err"] < 1e-3,
         f"position error {ge['max_abs_err_um'] * 1e3:.2f} nm (limit 10 nm), "
         f"dispersion error {ge['dispersion_rel_err']:.1e}"),
        ("Fraunhofer -> angular spectrum for z >> zR",
         fr["rel_L2_error"][-1] < 1e-2 and fr["grating_max_err_vs_paraxial_nm"] < 1e-3,
         f"difference {fr['rel_L2_error'][-1]:.1e} at z = {fr['z_over_zR'][-1]:.0f} zR (limit 1e-2)"),
        ("Diffraction-order efficiencies", order_err < 1e-3,
         f"max deviation from theory {order_err:.1e} (limit 1e-3)"),
        ("Energy conservation", closure < 1e-12 and bb < 1e-9,
         f"closure {closure:.1e}, broadband {bb:.1e} (limit 1e-12 / 1e-9)"),
        ("Wavelength-sampling convergence", rel150 < 1e-3,
         f"N_lambda = {CFG.n_lambda}: error {rel150:.1e} vs N_lambda = 600 (limit 1e-3)"),
        ("Spatial-sampling convergence",
         (max(dRs) - min(dRs)) / abs(dRs[-1]) < 1e-2 and (max(dxs) - min(dxs)) / abs(dxs[-1]) < 1e-6
         and (max(pix) - min(pix)) / abs(pix[0]) < 1e-9,
         f"dR spread {(max(dRs) - min(dRs)) / abs(dRs[-1]):.1e}, centroid spread "
         f"{(max(dxs) - min(dxs)) / abs(dxs[-1]):.1e}"),
    ]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for t in (test_free_propagation, test_grating_equation, test_fraunhofer, test_orders,
              test_energy, test_convergence):
        print(f"running {t.__name__} ...", flush=True)
        t()
    with open(OUT / "validation.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    results = checks()
    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    n_ok = sum(ok for _, ok, _ in results)
    print()
    if n_ok == len(results):
        print(f"All {len(results)} validation tests passed. Figures and validation.json saved to {OUT}")
    else:
        print(f"{len(results) - n_ok} of {len(results)} validation tests FAILED.")
        sys.exit(1)
