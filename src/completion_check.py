"""
completion_check.py -- Stage 1 completion criterion.

"Given S(lambda;c) and phi(x,y), the code must reliably and reproducibly
calculate I(x,y;c) and integrated detector signals."

  1. Reproducibility   identical results on repeated runs; independent of
                       how wavelengths and concentrations are batched
  2. Arbitrary phi     random phase masks: energy conservation, and the
                       angular spectrum against an independent direct
                       Rayleigh-Sommerfeld integral
  3. Arbitrary S       the detector image is linear in S (any spectrum,
                       not only Lorentzians); R ignores source power
  4. Gradients         autograd dR/dphi against finite differences
                       (needed for inverse design in Stage 2)

Each check has a pass threshold. Results go to results/completion.json.

    python src/completion_check.py
"""

import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)   # torch.jit notice on Python 3.14

import torch

from config import CFG
from detector import broadband_intensity, differential_readout, region_power
from doe import linear_grating, upsample
from main import forward, forward_from_spectrum
from propagation import (SimGrid, angular_spectrum, apply_doe,
                         band_limited_fraction, embed_phase, gaussian_beam,
                         grid_for_doe, power)

OUT = Path(__file__).resolve().parent.parent / "results"
LAM0 = CFG.lambda0_nm * 1e-9
RES = {}


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


# ----------------------------------------------------------------------------
# 1. Reproducibility
# ----------------------------------------------------------------------------

def check_reproducibility():
    c = torch.tensor([0.0, 1.0], dtype=torch.float64)
    r1, r2 = forward(c), forward(c)
    identical = bool(torch.equal(r1["I"], r2["I"]) and torch.equal(r1["R"], r2["R"]))
    # The explicit public interface must reproduce forward(c) when it receives
    # the same sampled S(lambda;c) and phi(x,y).
    phi_doe = linear_grating(CFG.n_doe, CFG.pitch, CFG.period)
    supplied = forward_from_spectrum(r1["lam_m"], r1["S"], phi_doe,
                                     dlam_nm=CFG.dlam_nm(r1["sensor"]))
    # Wavelength batching: one wavelength at a time vs all at once.
    g = r1["grid"]
    dl = CFG.dlam_nm(r1["sensor"])
    I_1 = broadband_intensity(r1["E_doe"], g, r1["lam_m"], CFG.z, r1["S"], dl, chunk=1)
    I_all = broadband_intensity(r1["E_doe"], g, r1["lam_m"], CFG.z, r1["S"], dl, chunk=150)
    # Concentration batching: [0, 1] together vs c = 1 alone.
    I_single = forward([1.0])["I"][0]
    RES["reproducibility"] = dict(
        repeated_runs_bitwise_identical=identical,
        supplied_S_phi_max_rel_diff=max(rel(supplied[k], r1[k])
                                        for k in ("I", "IA", "IB", "R")),
        wavelength_chunking_max_rel_diff=rel(I_1, I_all),
        concentration_batching_max_rel_diff=rel(I_single, r1["I"][1]))


# ----------------------------------------------------------------------------
# 2. Arbitrary phase masks
# ----------------------------------------------------------------------------

def smooth(phi, sigma_px):
    """Gaussian-smoothed copy of a phase map (periodic FFT convolution)."""
    n = phi.shape[0]
    f = torch.fft.fftfreq(n, dtype=torch.float64)
    G = torch.exp(-2 * (math.pi * sigma_px) ** 2 * (f[None, :] ** 2 + f[:, None] ** 2))
    return torch.fft.ifft2(torch.fft.fft2(phi) * G).real


def rayleigh_sommerfeld(E0, g, lam, z, targets):
    """Direct first Rayleigh-Sommerfeld integral at the target points.

    E(x,y,z) = sum E0(x',y') (z / r) (1/r - i k) exp(ikr) / (2 pi r) dx'^2
    Computed point by point, with no FFT and no periodic window.
    """
    k = 2 * math.pi / lam
    X, Y = torch.meshgrid(g.x, g.x, indexing="xy")
    keep = E0.abs() > 1e-8 * E0.abs().max()
    xs, ys, es = X[keep], Y[keep], E0[keep]
    out = []
    for i in range(0, len(targets), 256):
        t = targets[i:i + 256]
        r = torch.sqrt((t[:, :1] - xs) ** 2 + (t[:, 1:] - ys) ** 2 + z**2)
        h = (z / r) * (1 / r - 1j * k) * torch.exp(1j * k * r) / (2 * math.pi * r)
        out.append((h * es).sum(-1) * g.dx**2)
    return torch.cat(out)


def check_arbitrary_phase():
    torch.manual_seed(0)
    masks = {
        "random phase": 2 * math.pi * torch.rand(CFG.n_doe, CFG.n_doe, dtype=torch.float64),
        "smoothed random phase": 2 * math.pi * smooth(
            torch.rand(CFG.n_doe, CFG.n_doe, dtype=torch.float64) * 8, 4.0),
    }
    energy = {}
    for name, phi_doe in masks.items():
        g = grid_for_doe(CFG.n_doe, CFG.pitch, CFG.upsample, CFG.pad)
        E_in = gaussian_beam(g, CFG.w0)
        E0 = apply_doe(E_in, embed_phase(phi_doe, g, CFG.upsample))
        P_in = power(E_in, g)
        P_free = power(angular_spectrum(E0, g, LAM0, CFG.z, band_limit=False), g)
        P_bl = power(angular_spectrum(E0, g, LAM0, CFG.z), g)
        lost = band_limited_fraction(E0, g, LAM0, CFG.z)
        energy[name] = dict(doe=float(power(E0, g) / P_in) - 1,
                            propagation=float(P_free / P_in) - 1,
                            closure=float(P_bl / P_in + lost) - 1,
                            fraction_leaving_window=float(lost))

    # Independent reference: direct Rayleigh-Sommerfeld integral on a small,
    # finely sampled problem (0.5 um cells, random 4 um phase pixels). The
    # random pixels scatter light to steep angles, which wraps around the
    # periodic FFT window, so the window is enlarged until that is negligible.
    ph = upsample(2 * math.pi * torch.rand(32, 32, dtype=torch.float64), 8)
    z = 20e-6
    rs = {}
    for n in (256, 1024, 4096):
        gs = SimGrid(n=n, dx=0.5e-6)
        phi = torch.zeros(n, n, dtype=torch.float64)
        o = (n - 256) // 2
        phi[o:o + 256, o:o + 256] = ph
        E0 = gaussian_beam(gs, 10e-6) * torch.exp(1j * phi)
        E_as = angular_spectrum(E0, gs, LAM0, z, band_limit=False)
        idx = torch.arange(n // 2 - 32, n // 2 + 32)
        X, Y = torch.meshgrid(gs.x[idx], gs.x[idx], indexing="xy")
        targets = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
        E_rs = rayleigh_sommerfeld(E0, gs, LAM0, z, targets).reshape(64, 64)
        E_as_c = E_as[idx][:, idx]
        rs[f"{n * 0.5:.0f} um window"] = float(torch.linalg.norm(E_as_c - E_rs) / torch.linalg.norm(E_rs))
    RES["arbitrary_phase"] = dict(
        energy_deviation=energy,
        angular_spectrum_vs_rayleigh_sommerfeld_rel_L2=rs)


# ----------------------------------------------------------------------------
# 3. Arbitrary spectra
# ----------------------------------------------------------------------------

def check_arbitrary_spectrum():
    r = forward([0.0])
    g, lam, E0 = r["grid"], r["lam_m"], r["E_doe"]
    dl = CFG.dlam_nm(r["sensor"])
    nm = lam * 1e9
    torch.manual_seed(1)
    S1 = torch.rand(lam.numel(), dtype=torch.float64)                        # random spectrum
    S2 = (torch.exp(-((nm - 1535) / 0.8) ** 2) + 0.5 * torch.exp(-((nm - 1570) / 0.8) ** 2))
    I1 = broadband_intensity(E0, g, lam, CFG.z, S1, dl)
    I2 = broadband_intensity(E0, g, lam, CFG.z, S2, dl)
    I12 = broadband_intensity(E0, g, lam, CFG.z, S1 + S2, dl)
    I3 = broadband_intensity(E0, g, lam, CFG.z, 3.7 * S2, dl)
    mA, mB = r["masks"]
    R2 = differential_readout(region_power(I2, mA, g), region_power(I2, mB, g))
    R3 = differential_readout(region_power(I3, mA, g), region_power(I3, mB, g))
    RES["arbitrary_spectrum"] = dict(
        superposition_max_rel_err=rel(I12, I1 + I2),
        scaling_max_rel_err=rel(I3, 3.7 * I2),
        R_change_when_source_power_x3p7=abs(float(R3 - R2)))


# ----------------------------------------------------------------------------
# 4. Gradients with respect to the DOE phase
# ----------------------------------------------------------------------------

def check_gradients():
    base = linear_grating(CFG.n_doe, CFG.pitch, CFG.period)

    def R_of(phi):
        return forward([0.0], phi_doe=phi, n_lambda=30)["R"][0]

    phi = base.clone().requires_grad_(True)
    R_of(phi).backward()
    grad = phi.grad
    h = 1e-5
    rows = []
    for (i, j) in ((128, 128), (120, 150), (140, 105)):
        p, m = base.clone(), base.clone()
        p[i, j] += h
        m[i, j] -= h
        fd = float((R_of(p) - R_of(m)) / (2 * h))
        rows.append(dict(pixel=[i, j], autograd=float(grad[i, j]), finite_difference=fd,
                         rel_diff=abs(float(grad[i, j]) - fd) / abs(fd)))
    RES["gradients"] = rows


def verdicts():
    """(name, passed, detail) for every completion check."""
    rp, ph, sp, gr = RES["reproducibility"], RES["arbitrary_phase"], RES["arbitrary_spectrum"], RES["gradients"]
    en = max(abs(v) for m in ph["energy_deviation"].values()
             for k, v in m.items() if k != "fraction_leaving_window")
    rs = list(ph["angular_spectrum_vs_rayleigh_sommerfeld_rel_L2"].values())
    g = max(r["rel_diff"] for r in gr)
    batch = max(rp["supplied_S_phi_max_rel_diff"],
                rp["wavelength_chunking_max_rel_diff"],
                rp["concentration_batching_max_rel_diff"])
    lin = max(sp["superposition_max_rel_err"], sp["scaling_max_rel_err"])
    return [
        ("Reproducible: repeated runs", rp["repeated_runs_bitwise_identical"], "bit-for-bit identical"),
        ("Reproducible: batching", batch < 1e-12, f"max difference {batch:.1e} (limit 1e-12)"),
        ("Any phase mask: energy", en < 1e-12, f"max deviation {en:.1e} (limit 1e-12)"),
        ("Any phase mask: vs Rayleigh-Sommerfeld", rs[-1] < 1e-3 and rs == sorted(rs, reverse=True),
         f"difference {rs[-1]:.1e} on the largest window (limit 1e-3), decreasing with window size"),
        ("Any spectrum: linearity", lin < 1e-12, f"max error {lin:.1e} (limit 1e-12)"),
        ("Any spectrum: R independent of source power", sp["R_change_when_source_power_x3p7"] < 1e-12,
         f"change {sp['R_change_when_source_power_x3p7']:.1e} (limit 1e-12)"),
        ("Gradients dR/dphi vs finite differences", g < 1e-5, f"max rel. difference {g:.1e} (limit 1e-5)"),
    ]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (check_reproducibility, check_arbitrary_phase, check_arbitrary_spectrum,
              check_gradients):
        print(f"running {f.__name__} ...", flush=True)
        f()
    with open(OUT / "completion.json", "w") as fh:
        json.dump(RES, fh, indent=2)
    results = verdicts()
    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print()
    if all(ok for _, ok, _ in results):
        print(f"Stage 1 completion criterion met: all {len(results)} checks passed. "
              f"Results saved to {OUT / 'completion.json'}")
    else:
        print("Stage 1 completion criterion NOT met.")
        sys.exit(1)
