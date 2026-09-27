"""
embodiment2.py -- Embodiment 2 with the fixed Stage 1 grating: pixel-wise
spectral coding and direct resonance-shift estimation.

    dn -> d = S_n dn -> s(lambda; d) -> DOE -> camera pixels y = W s -> d_hat -> dn_hat

No inverse design: the DOE is the Stage 1 linear phase grating. Produces
the Section 6 deliverables in results/:

  e2_weighting.png      1. pixel spectral weighting functions W_j(lambda)
  e2_camera.png         2. camera vector y(d) for several known shifts
  e2_sensitivity.png    3. sensitivity map g_j; Eq. (7) against Eq. (8)
  e2_estimate.png       4. estimated vs true shift, noiseless and noisy
  e2_rms_vs_M.png       5. RMS error vs retained pixels M, with the CRB
  e2_delta_n.png        6. recovered dn and sigma_dn (Eq. 12)
  e2_information.png    Fisher information vs operating point, with and
                        without the source amplitude as a nuisance (Eq. 13)
  embodiment2.json, e2_rms_vs_M.csv, e2_delta_n.csv, e2_pixel_ranking.csv

Every check has a pass threshold, as in validation.py.

    python src/embodiment2.py
"""

import csv
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)   # torch.jit notice on Python 3.14

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import main
from config import CFG, E2
from detector import broadband_intensity
from estimator import (add_noise, crb, gauss_newton, gls_estimate,
                       joint_amplitude_estimate, local_fisher, noise_variance,
                       ols_estimate,
                       rank_pixels, sensitivity_exact, sensitivity_fd,
                       to_refractive_index)
from main import spot_position
from propagation import band_limited_fraction
from spectral_coding import (ShiftSensor, bin_power, camera_response,
                             weighting_matrix)

OUT = Path(__file__).resolve().parent.parent / "results"
UM = 1e6
RES = {}


def savefig(fig, name):
    main.savefig(fig, name)


def m_values(n):
    """M = 1, 2, 4, ... and n (all ROI pixels)."""
    ms, m = [], 1
    while m < n:
        ms.append(m)
        m *= 2
    return ms + [n]


# ----------------------------------------------------------------------------
# Model: W, y, g
# ----------------------------------------------------------------------------

def build(cfg=CFG, e2=E2, phi_doe=None):
    """Everything the estimator needs, for a fixed DOE (default: Stage 1 grating)."""
    sensor = ShiftSensor(cfg, e2.photons)
    t0 = time.time()
    m = weighting_matrix(sensor.lam_m, phi_doe, cfg, e2)
    W, cam = m["W"], m["camera"]
    mu = camera_response(W, sensor.s(0.0))
    var = noise_variance(mu, e2.read_var)
    delta = e2.delta_gammas * sensor.gamma

    def response(d):
        return camera_response(W, sensor.s(d))

    g_fd = sensitivity_fd(response, delta)
    g_fd2 = sensitivity_fd(response, delta / 2)
    g_an = sensitivity_exact(W, sensor.ds_analytic(0.0))
    g_ag = sensitivity_exact(W, sensor.ds_autograd(0.0))
    order, F_sorted = rank_pixels(g_fd, var)
    return dict(sensor=sensor, W=W, W_full=m["W_full"], grid=m["grid"], camera=cam,
                E_doe=m["E_doe"], mu=mu, var=var, delta=delta, response=response,
                g=g_fd, g_fd2=g_fd2, g_an=g_an, g_ag=g_ag, order=order,
                F_sorted=F_sorted, build_seconds=time.time() - t0)


def model_fns(W, sensor):
    """y(d) and dy/dd for a batch of shifts d (K,), on the pixels of W."""
    return (lambda d: camera_response(W, sensor.s(d)),
            lambda d: sensitivity_exact(W, sensor.ds_analytic(d)))


# ----------------------------------------------------------------------------
# Checks of the measurement model
# ----------------------------------------------------------------------------

def check_model(b, cfg=CFG):
    sensor, W, grid, cam = b["sensor"], b["W"], b["grid"], b["camera"]
    # (a) Power: each column of the full-camera W holds the transmitted
    #     fraction, 1 minus the band-limit loss at that wavelength.
    lost = torch.cat([band_limited_fraction(b["E_doe"], grid, sensor.lam_m[i:i + 16], cfg.z)
                      for i in range(0, sensor.lam_m.numel(), 16)])
    col = b["W_full"].sum((-2, -1))
    power_err = float((col - (1 - lost)).abs().max())
    roi_fraction = float((W.sum(0) / col).min())
    # (b) y = W s equals the Stage 1 broadband image binned into the same pixels.
    s0 = sensor.s(0.0)
    I = broadband_intensity(b["E_doe"], grid, sensor.lam_m, cfg.z, s0 / sensor.dlam_nm,
                            sensor.dlam_nm)
    y_stage1 = bin_power(I, grid, cam.bin)[cam.rows.start:cam.rows.stop,
                                           cam.cols.start:cam.cols.stop].reshape(-1)
    y_err = float((y_stage1 - b["mu"]).abs().max() / b["mu"].max())
    rel = lambda a, c: float((a - c).abs().max() / c.abs().max())
    RES["model"] = dict(
        n_lambda=sensor.lam_m.numel(), lambda_range_nm=[float(sensor.lam_nm[0]),
                                                         float(sensor.lam_nm[-1])],
        camera_pitch_um=cam.pitch * UM, roi_pixels=list(cam.shape),
        roi_x_um=[float(cam.x[0] * UM), float(cam.x[-1] * UM)],
        roi_y_um=[float(cam.y[0] * UM), float(cam.y[-1] * UM)],
        photons=E2.photons, read_noise_e_per_frame=E2.read_noise_e, n_frames=E2.n_frames,
        peak_pixel_e=float(b["mu"].max()), peak_pixel_e_per_frame=float(b["mu"].max()) / E2.n_frames,
        W_column_power_max_err=power_err, band_limit_loss=[float(lost.min()), float(lost.max())],
        roi_fraction_of_camera_power_min=roi_fraction,
        y_vs_stage1_broadband_max_rel=y_err,
        delta_nm=b["delta"],
        g_fd_vs_fd_half_delta=rel(b["g"], b["g_fd2"]),
        g_fd_vs_analytic=rel(b["g"], b["g_an"]),
        g_analytic_vs_autograd=rel(b["g_ag"], b["g_an"]),
        build_seconds=b["build_seconds"])


def check_noise(b, e2=E2, n=20000):
    """Sampled noise has mean y and variance y + read_var (Eq. 9)."""
    gen = torch.Generator().manual_seed(e2.seed + 1)
    idx = torch.cat([b["order"][:3], torch.argsort(b["mu"])[:3]])     # bright and dark
    y = b["mu"][idx]
    ym = add_noise(y, e2.read_var, n=n, generator=gen)
    RES["noise"] = dict(
        pixels=idx.tolist(), mean_e=y.tolist(),
        max_rel_err_mean=float(((ym.mean(0) - y).abs() / (y + e2.read_var) ** 0.5).max()
                               / (1 / n ** 0.5)),                     # in standard errors
        max_rel_err_var=float(((ym.var(0) - noise_variance(y, e2.read_var)).abs()
                               / noise_variance(y, e2.read_var)).max()))


# ----------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------

def linearity(b, e2=E2):
    """Noiseless linear and Gauss-Newton estimates over +-2 gamma; linear range."""
    sensor, W, mu, var, g = b["sensor"], b["W"], b["mu"], b["var"], b["g"]
    gam = sensor.gamma
    d = torch.linspace(-2 * gam, 2 * gam, 161, dtype=torch.float64)
    y = camera_response(W, sensor.s(d))
    lin, sigma = gls_estimate(y, mu, g, var)
    model, jac = model_fns(W, sensor)
    gn = gauss_newton(y, model, jac, var, iterations=e2.gn_iterations)
    bias = lin - d

    def linear_range(sig):
        ok = bias.abs() < sig
        # contiguous interval around d = 0
        i0 = int(torch.argmin(d.abs()))
        lo = hi = i0
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(d) - 1 and ok[hi + 1]:
            hi += 1
        return [float(d[lo]), float(d[hi])]

    ranges = {str(M): linear_range(crb(g, var, b["order"][:M]))
              for M in m_values(g.numel()) if M in (1, 16, 256, g.numel())}
    RES["linearity"] = dict(
        shift_nm=d.tolist(), linear_estimate_nm=lin.tolist(), gauss_newton_nm=gn.tolist(),
        crb_all_nm=sigma, linear_range_nm_by_M=ranges,
        gauss_newton_max_abs_err_nm=float((gn - d).abs().max()),
        bias_at_0p1_gamma_nm=float(bias[torch.argmin((d - 0.1 * gam).abs())]))
    return dict(d=d, lin=lin, gn=gn, bias=bias, ranges=ranges)


def monte_carlo(b, e2=E2):
    """RMS error vs M for GLS (Eq. 10), OLS (Eq. 11) and Gauss-Newton."""
    sensor, W, mu, var, g, order = (b[k] for k in ("sensor", "W", "mu", "var", "g", "order"))
    gen = torch.Generator().manual_seed(e2.seed)
    shifts = torch.tensor(e2.test_shifts_gammas, dtype=torch.float64) * sensor.gamma
    Y = torch.stack([add_noise(camera_response(W, sensor.s(float(t))), e2.read_var,
                               n=e2.mc_trials, generator=gen) for t in shifts])   # (T, K, N)
    truth = shifts[:, None].expand(-1, e2.mc_trials)
    rows, per_shift = [], {}
    for M in m_values(g.numel()):
        idx = order[:M]
        lin, bound = gls_estimate(Y, mu, g, var, idx)
        ols, _ = ols_estimate(Y, mu, g, idx=idx)
        Wm = W[idx]
        model, jac = model_fns(Wm, sensor)
        gn = gauss_newton(Y[..., idx], model, jac, var[idx], iterations=e2.gn_iterations,
                          max_step=0.5 * sensor.gamma, bounds=(-2 * sensor.gamma, 2 * sensor.gamma))
        # Local bound at the test shifts: the information depends on d, so an
        # estimator that re-linearizes (Gauss-Newton) is bounded by F(d_t), not F(0).
        yt, gt = model(shifts), jac(shifts)
        local = sum(1 / local_fisher(gt[k], noise_variance(yt[k], e2.read_var))
                    for k in range(len(shifts))) / len(shifts)
        # Exact mean-square error of the linear GLS estimate at the test shifts:
        # bias^2 + w^T Sigma(d_t) w / F^2, with w = g / var(0) on the pixels P_M.
        w, F0 = g[idx] / var[idx], bound ** -2
        predicted = sum(((float((yt[k] - mu[idx]) @ w) / F0 - float(shifts[k])) ** 2
                         + float((w * w * noise_variance(yt[k], e2.read_var)).sum()) / F0 ** 2)
                        for k in range(len(shifts))) / len(shifts)
        rms = lambda e: float(((e - truth) ** 2).mean().sqrt())
        rows.append(dict(M=M, crb_nm=bound, crb_local_nm=local ** 0.5,
                         rms_gls_predicted_nm=predicted ** 0.5,
                         rms_gls_nm=rms(lin), rms_ols_nm=rms(ols),
                         rms_gauss_newton_nm=rms(gn),
                         max_abs_bias_gls_nm=float((lin.mean(1) - shifts).abs().max()),
                         crb_dn_riu=bound / e2.Sn_nm_per_riu,
                         rms_gls_dn_riu=rms(lin) / e2.Sn_nm_per_riu))
        per_shift[M] = dict(mean=lin.mean(1), std=lin.std(1), gn_mean=gn.mean(1),
                            gn_std=gn.std(1))
    RES["monte_carlo"] = dict(test_shifts_nm=shifts.tolist(), trials=e2.mc_trials,
                              seed=e2.seed, rows=rows)
    return dict(shifts=shifts, rows=rows, per_shift=per_shift)


def information(b, e2=E2):
    """Fisher information vs operating point d, all ROI pixels.

    With the source centred on the resonance, the total transmitted power is
    stationary at d = 0, so there the information comes only from how the
    spectrum is redistributed over the pixels (spectral coding). Away from 0
    the total power changes with d and adds an intensity channel, which a
    source-power drift would corrupt. Treating the amplitude as a nuisance
    (Eq. 13) removes that channel.
    """
    sensor, W = b["sensor"], b["W"]
    model, jac = model_fns(W, sensor)
    d = torch.linspace(-2 * sensor.gamma, 2 * sensor.gamma, 81, dtype=torch.float64)
    Y, G = model(d), jac(d)
    shift_only, joint, power_slope = [], [], []
    for k in range(len(d)):
        v = noise_variance(Y[k], e2.read_var)
        shift_only.append(local_fisher(G[k], v) ** -0.5)
        joint.append(local_fisher(G[k], v, mu=Y[k]) ** -0.5)
        power_slope.append(float(G[k].sum() / Y[k].sum()))
    RES["information"] = dict(
        shift_nm=d.tolist(), crb_shift_only_nm=shift_only, crb_amplitude_robust_nm=joint,
        relative_power_slope_per_nm=power_slope,
        crb_at_0_nm=[shift_only[40], joint[40]],
        crb_at_1gamma_nm=[shift_only[60], joint[60]])
    return dict(d=d, shift_only=shift_only, joint=joint, power_slope=power_slope)


def stage1_two_region(b, cfg=CFG, e2=E2):
    """CRB of the Stage 1 two-region readout (A, B) built from the same pixels."""
    cam, g, mu = b["camera"], b["g"], b["mu"]
    x_spot = spot_position(cfg.lambda0_nm * 1e-9, cfg)
    X = cam.x[None, :].expand(cam.shape).reshape(-1)
    Y = cam.y[:, None].expand(cam.shape).reshape(-1)
    inside = Y.abs() < cfg.det_height / 2
    A = inside & (X >= x_spot - cfg.det_width) & (X < x_spot)
    B = inside & (X >= x_spot) & (X < x_spot + cfg.det_width)
    F = sum(float(g[m].sum()) ** 2 / (float(mu[m].sum()) + int(m.sum()) * e2.read_var)
            for m in (A, B))
    RES["stage1_two_region"] = dict(pixels_A=int(A.sum()), pixels_B=int(B.sum()),
                                    crb_nm=F ** -0.5)
    return F ** -0.5


def power_drift(b, e2=E2, a=1.01, d_true=0.1):
    """Eq. (13): a 1 % source-power change biases Eq. (10) but not the joint fit."""
    sensor, W, mu, var, g = b["sensor"], b["W"], b["mu"], b["var"], b["g"]
    d_true *= sensor.gamma
    y1 = camera_response(W, sensor.s(d_true))
    lin1, _ = gls_estimate(y1, mu, g, var)
    joint1, _ = joint_amplitude_estimate(y1, mu, g, var)
    y = a * y1
    lin, sigma = gls_estimate(y, mu, g, var)
    joint, a_hat = joint_amplitude_estimate(y, mu, g, var)
    # Cost of the extra parameter: CRB of d in the 2-parameter model.
    H = torch.stack([mu, g], 1)
    Finfo = H.T @ (H / var[:, None])
    crb_joint = float(torch.linalg.inv(Finfo)[1, 1]) ** 0.5
    RES["power_drift"] = dict(a=a, true_shift_nm=d_true, gls_estimate_nm=float(lin),
                              joint_estimate_nm=float(joint), joint_a_hat=float(a_hat),
                              gls_error_from_drift_nm=float(lin - lin1),
                              joint_error_from_drift_nm=float(joint - joint1),
                              crb_shift_only_nm=sigma, crb_joint_nm=crb_joint)


# ----------------------------------------------------------------------------
# Figures (Section 6)
# ----------------------------------------------------------------------------

def _extent(cam):
    h = cam.pitch * UM / 2
    return [float(cam.x[0] * UM) - h, float(cam.x[-1] * UM) + h,
            float(cam.y[0] * UM) - h, float(cam.y[-1] * UM) + h]


def _xy(cam, j):
    r, c = divmod(int(j), cam.shape[1])
    return float(cam.x[c] * UM), float(cam.y[r] * UM)


def fig_weighting(b):
    """1. W_j(lambda) of representative pixels, resonance marked."""
    sensor, W, g, cam = b["sensor"], b["W"], b["g"], b["camera"]
    lam = sensor.lam_nm.numpy()
    pos, neg = int(torch.argmax(g)), int(torch.argmin(g))
    bright = int(torch.argmax(b["mu"]))
    weak = int(b["order"][len(b["order"]) // 3])
    picks = [(pos, "max g > 0"), (neg, "min g < 0"), (bright, "brightest"),
             (weak, "low |g|")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9]))
    for k, (j, lab) in enumerate(picks):
        x, y = _xy(cam, j)
        ax[0].plot(lam, W[j].numpy(), color=f"C{k}",
                   label=f"{lab}: ({x:.0f}, {y:.0f}) µm, g = {float(g[j]):+.3g} e⁻/nm")
    for a in ax[:2]:
        a.axvline(sensor.lambda_r0, color="k", ls=":", lw=1, label="λ_r,0 = 1550 nm")
        a.axvspan(sensor.lambda_r0 - sensor.gamma, sensor.lambda_r0 + sensor.gamma,
                  color="0.85", zorder=0, label="λ_r,0 ± γ")
    ax[0].set(xlabel="λ [nm]", ylabel="W_j(λ)  (fraction of power on pixel j)",
              title="(a) Pixel spectral weighting functions")
    ax[0].legend(fontsize=7, loc="lower left")
    for k, (j, lab) in enumerate(picks):
        w = W[j] / W[j].mean()
        ax[1].plot(lam, w.numpy(), color=f"C{k}", label=lab)
    s0 = sensor.s(0.0)
    ax[1].plot(lam, (s0 / s0.max() * 1.02).numpy(), "k--", lw=1, label="s(λ; 0) (scaled)")
    ax[1].set(xlabel="λ [nm]", ylabel="W_j(λ) / mean", ylim=(0, 1.3),
              title="(b) Normalized: slopes across the resonance")
    ax[1].legend(fontsize=7, loc="lower left")
    im = ax[2].imshow(cam.image(b["mu"]).numpy(), origin="lower", extent=_extent(cam),
                      cmap="inferno")
    fig.colorbar(im, ax=ax[2], label="μ = y(0) [e⁻]")
    for k, (j, lab) in enumerate(picks):
        ax[2].plot(*_xy(cam, j), "o", mfc="none", mec=f"C{k}", ms=9, mew=2)
    ax[2].set(xlabel="x [µm]", ylabel="y [µm]", title="(c) Pixel locations")
    fig.tight_layout()
    savefig(fig, "e2_weighting.png")


def fig_camera(b):
    """2. The complete camera vector y(d) for several known shifts."""
    sensor, W, cam, mu = b["sensor"], b["W"], b["camera"], b["mu"]
    gam = sensor.gamma
    shifts = [-1.0, -0.5, 0.5, 1.0]
    Y = {f: camera_response(W, sensor.s(f * gam)) for f in shifts}
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(cam.image(mu).numpy(), origin="lower", extent=_extent(cam), cmap="inferno")
    fig.colorbar(im, ax=ax, label="y [e⁻]")
    ax.set(xlabel="x [µm]", ylabel="y [µm]", title="(a) y(0), camera ROI")
    v = max(float((Y[f] - mu).abs().max()) for f in shifts)
    for k, f in enumerate((-1.0, 1.0)):
        ax = fig.add_subplot(gs[0, k + 1])
        im = ax.imshow(cam.image(Y[f] - mu).numpy(), origin="lower", extent=_extent(cam),
                       cmap="RdBu_r", vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, label="y(Δλ_r) − y(0) [e⁻]")
        ax.set(xlabel="x [µm]", ylabel="y [µm]", title=f"({'bc'[k]}) Δλ_r = {f * gam:+.2f} nm")
    # The raw change is dominated by the total-power change; normalizing the
    # total power shows the spectral redistribution the grating encodes.
    ax = fig.add_subplot(gs[0, 3])
    dn = Y[1.0] * (mu.sum() / Y[1.0].sum()) - mu
    vn = float(dn.abs().max())
    im = ax.imshow(cam.image(dn).numpy(), origin="lower", extent=_extent(cam),
                   cmap="RdBu_r", vmin=-vn, vmax=vn)
    fig.colorbar(im, ax=ax, label="power-normalized change [e⁻]")
    ax.set(xlabel="x [µm]", ylabel="y [µm]", title=f"(d) {gam:+.2f} nm, power-normalized")
    ax = fig.add_subplot(gs[1, :2])
    j = torch.arange(mu.numel())
    ax.plot(j.numpy(), mu.numpy(), "k", lw=0.6, label="Δλ_r = 0")
    for k, f in enumerate(shifts):
        ax.plot(j.numpy(), Y[f].numpy(), lw=0.5, color=f"C{k}", label=f"Δλ_r = {f * gam:+.2f} nm")
    ax.set(xlabel="pixel index j (row-major over the ROI)", ylabel="y_j [e⁻]",
           title=f"(e) Camera vector y(Δλ_r), {mu.numel()} pixels")
    ax.legend(fontsize=7, ncol=5)
    ax = fig.add_subplot(gs[1, 2:])
    for k, f in enumerate(shifts):
        ax.plot(j.numpy(), (Y[f] - mu).numpy(), lw=0.5, color=f"C{k}", label=f"Δλ_r = {f * gam:+.2f} nm")
    ax.set(xlabel="pixel index j", ylabel="y_j(Δλ_r) − y_j(0) [e⁻]",
           title="(f) Change of the camera vector")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    savefig(fig, "e2_camera.png")


def fig_sensitivity(b):
    """3. Signed sensitivity map; finite difference (Eq. 7) vs exact (Eq. 8)."""
    cam, g, g_an, g_ag = b["camera"], b["g"], b["g_an"], b["g_ag"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    v = float(g.abs().max())
    im = ax[0].imshow(cam.image(g).numpy(), origin="lower", extent=_extent(cam),
                      cmap="RdBu_r", vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax[0], label="g_j = ∂y_j/∂(Δλ_r) [e⁻/nm]")
    top = b["order"][:16]
    for j in top:
        ax[0].plot(*_xy(cam, j), "s", mfc="none", mec="k", ms=4, mew=0.8)
    ax[0].set(xlabel="x [µm]", ylabel="y [µm]",
              title="(a) g_j by Eq. (7); □ = 16 best pixels")
    err = (g - g_an) / v
    im = ax[1].imshow(cam.image(err).numpy(), origin="lower", extent=_extent(cam), cmap="RdBu_r")
    fig.colorbar(im, ax=ax[1], label="(g_FD − g_exact) / max|g|")
    ax[1].set(xlabel="x [µm]", ylabel="y [µm]",
              title=f"(b) Eq. (7) − Eq. (8), δ = {b['delta']:.3f} nm")
    ax[2].plot(g_an.numpy(), g.numpy(), ".", ms=2, label="central difference, δ")
    ax[2].plot(g_an.numpy(), b["g_fd2"].numpy(), ".", ms=1, label="central difference, δ/2")
    ax[2].plot(g_an.numpy(), g_ag.numpy(), ".", ms=0.5, label="autograd")
    ax[2].plot([-v, v], [-v, v], "k", lw=0.5)
    m = RES["model"]
    ax[2].text(0.03, 0.97, f"FD(δ) vs analytic: {m['g_fd_vs_analytic']:.1e}\n"
               f"FD(δ) vs FD(δ/2): {m['g_fd_vs_fd_half_delta']:.1e}\n"
               f"autograd vs analytic: {m['g_analytic_vs_autograd']:.1e}\n(max-normalized)",
               transform=ax[2].transAxes, va="top", fontsize=8)
    ax[2].set(xlabel="g_j, analytic dT/dΔλ_r (Eq. 8) [e⁻/nm]", ylabel="g_j [e⁻/nm]",
              title="(c) Cross-check of the three derivatives")
    ax[2].legend(fontsize=7, loc="lower right", markerscale=6)
    fig.tight_layout()
    savefig(fig, "e2_sensitivity.png")


def fig_estimate(b, lin, mc):
    """4. Estimated vs true shift: noiseless (bias) and noisy (error bars)."""
    gam = b["sensor"].gamma
    d, M_all = lin["d"], str(b["g"].numel())
    lo, hi = lin["ranges"][M_all]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(d.numpy(), d.numpy(), "k", lw=0.6, label="ideal")
    ax[0].plot(d.numpy(), lin["lin"].numpy(), label="linear GLS, Eq. (10)")
    ax[0].plot(d.numpy(), lin["gn"].numpy(), "--", label="Gauss–Newton (re-linearized)")
    ax[0].axvspan(lo, hi, color="C2", alpha=0.15, label="linear range (|bias| < σ_CRB, all px)")
    ax[0].set(xlabel="true Δλ_r [nm]", ylabel="estimated Δλ_r [nm]",
              title="(a) Noiseless estimate")
    ax[0].legend(fontsize=7)
    ax[1].plot(d.numpy(), lin["bias"].numpy(), label="bias of linear estimate")
    for M, ls in ((M_all, "-"), ("256", "--"), ("16", ":")):
        s = crb(b["g"], b["var"], b["order"][:int(M)])
        ax[1].fill_between(d.numpy(), -s, s, color="0.5", alpha=0.12 if M != M_all else 0.25,
                           label=f"±σ_CRB, M = {M}")
    ax[1].axvspan(lo, hi, color="C2", alpha=0.12)
    ax[1].set(xlabel="true Δλ_r [nm]", ylabel="estimate − true [nm]", ylim=(-0.8, 1.2),
              title="(b) Linearization bias vs noise")
    ax[1].legend(fontsize=7)
    shifts = mc["shifts"].numpy()
    for k, M in enumerate((16, 256, b["g"].numel())):
        p = mc["per_shift"][M]
        off = (k - 1) * 0.004
        ax[2].errorbar(shifts + off, (p["mean"] - mc["shifts"]).numpy(), yerr=p["std"].numpy(),
                       fmt="o", ms=4, capsize=3, label=f"M = {M}: mean ± std")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].axvspan(max(lo, shifts.min() - 0.05), min(hi, shifts.max() + 0.05), color="C2", alpha=0.12)
    ax[2].set(xlabel="true Δλ_r [nm]", ylabel="estimate − true [nm]",
              title=f"(c) Noisy, {E2.mc_trials} trials per shift (linear GLS)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "e2_estimate.png")


def fig_information(b, info):
    """Information vs operating point: shift only vs amplitude-robust."""
    d, gam = info["d"].numpy(), b["sensor"].gamma
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogy(d, info["shift_only"], label="shift only (source power known)")
    ax[0].semilogy(d, info["joint"], "--", label="amplitude-robust (Eq. 13 nuisance)")
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set(xlabel="operating point Δλ_r [nm]", ylabel="local CRB of Δλ_r [nm]",
              title="(a) Local Cramér–Rao bound, all ROI pixels")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)
    ax[1].plot(d, info["power_slope"])
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="operating point Δλ_r [nm]", ylabel="(dP/dΔλ_r) / P  [1/nm]",
              title="(b) Total transmitted-power slope")
    fig.tight_layout()
    savefig(fig, "e2_information.png")


def fig_rms(mc, ab_crb):
    """5. RMS error vs M with the Cramer-Rao bound."""
    r = mc["rows"]
    M = [x["M"] for x in r]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.loglog(M, [x["crb_nm"] for x in r], "k-", label="Cramér–Rao bound at Δλ_r = 0, Eq. (14)")
    ax.loglog(M, [x["crb_local_nm"] for x in r], "k:", label="local CRB at the test shifts")
    ax.loglog(M, [x["rms_gls_nm"] for x in r], "o", mfc="none", label="linear GLS, Eq. (10)")
    ax.loglog(M, [x["rms_ols_nm"] for x in r], "s", mfc="none", ms=5, label="equal-variance LS, Eq. (11)")
    ax.loglog(M, [x["rms_gauss_newton_nm"] for x in r], "x",
              label="Gauss–Newton (bounded to ±2γ; biased for small M)")
    ax.axhline(ab_crb, color="C3", ls="--", lw=1, label="Stage 1 two-region (A, B) readout, CRB")
    ax.set(xlabel="number of retained pixels M (ranked by F_j)", ylabel="RMS error of Δλ_r [nm]",
           title="Estimation error vs retained pixels (fixed grating)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    sec = ax.secondary_yaxis("right", functions=(lambda v: v / E2.Sn_nm_per_riu * 1e4,
                                                 lambda v: v * E2.Sn_nm_per_riu / 1e4))
    sec.set_ylabel(f"σ_Δn [10⁻⁴ RIU]  (S_n = {E2.Sn_nm_per_riu:.0f} nm/RIU)")
    fig.tight_layout()
    savefig(fig, "e2_rms_vs_M.png")


def fig_delta_n(b, mc):
    """6. Recovered dn with sigma_dn (Eq. 12)."""
    Sn = E2.Sn_nm_per_riu
    shifts = mc["shifts"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    dn_true = (shifts / Sn).numpy() * 1e4
    ax[0].plot(dn_true, dn_true, "k", lw=0.5)
    for k, M in enumerate((16, b["g"].numel())):
        p = mc["per_shift"][M]
        bound = crb(b["g"], b["var"], b["order"][:M])
        dn, sdn = to_refractive_index(p["mean"], bound, Sn)
        ax[0].errorbar(dn_true + (k - 0.5) * 0.02, (dn * 1e4).numpy(),
                       yerr=(p["std"] / Sn * 1e4).numpy(), fmt="o", capsize=3,
                       label=f"M = {M}: σ_Δn = {sdn:.2e} RIU (CRB)")
    ax[0].set(xlabel="true Δn [10⁻⁴ RIU]", ylabel="estimated Δn [10⁻⁴ RIU]",
              title=f"(a) Recovered Δn, S_n = {Sn:.0f} nm/RIU (mean ± std)")
    ax[0].legend(fontsize=8)
    r = mc["rows"]
    M = [x["M"] for x in r]
    ax[1].loglog(M, [x["crb_dn_riu"] for x in r], "k-", label="σ_Δn from CRB, Eq. (12)")
    ax[1].loglog(M, [x["rms_gls_dn_riu"] for x in r], "o", mfc="none", label="Monte Carlo RMS")
    ax[1].set(xlabel="retained pixels M", ylabel="σ_Δn [RIU]", title="(b) Refractive-index resolution")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "e2_delta_n.png")


# ----------------------------------------------------------------------------
# Tables and pass/fail
# ----------------------------------------------------------------------------

def write_tables(b, mc):
    with open(OUT / "e2_rms_vs_M.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mc["rows"][0]))
        w.writeheader()
        w.writerows(mc["rows"])
    Sn = E2.Sn_nm_per_riu
    with open(OUT / "e2_delta_n.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["M", "true_shift_nm", "true_dn_riu", "mean_est_shift_nm", "std_shift_nm",
                    "mean_est_dn_riu", "std_dn_riu", "crb_dn_riu"])
        for M in (1, 16, 256, b["g"].numel()):
            p = mc["per_shift"][M]
            bound = crb(b["g"], b["var"], b["order"][:M])
            for k, t in enumerate(mc["shifts"]):
                w.writerow([M, f"{float(t):.4f}", f"{float(t) / Sn:.4e}", f"{float(p['mean'][k]):.5f}",
                            f"{float(p['std'][k]):.5f}", f"{float(p['mean'][k]) / Sn:.4e}",
                            f"{float(p['std'][k]) / Sn:.4e}", f"{bound / Sn:.4e}"])
    cam = b["camera"]
    with open(OUT / "e2_pixel_ranking.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "pixel", "x_um", "y_um", "mu_e", "g_e_per_nm", "F_per_nm2",
                    "crb_first_M_nm"])
        cum = torch.cumsum(b["F_sorted"], 0)
        for k in range(64):
            j = int(b["order"][k])
            x, y = _xy(cam, j)
            w.writerow([k + 1, j, f"{x:.1f}", f"{y:.1f}", f"{float(b['mu'][j]):.1f}",
                        f"{float(b['g'][j]):.4f}", f"{float(b['F_sorted'][k]):.4e}",
                        f"{float(cum[k]) ** -0.5:.4f}"])


def checks():
    m, n, lin, mc, pd = (RES[k] for k in ("model", "noise", "linearity", "monte_carlo",
                                          "power_drift"))
    last = mc["rows"][-1]
    rms_ratio = max(abs(x["rms_gls_nm"] / x["crb_nm"] - 1) for x in mc["rows"])
    mc_err = max(abs(x["rms_gls_nm"] / x["rms_gls_predicted_nm"] - 1) for x in mc["rows"])
    drift_lin = abs(pd["gls_error_from_drift_nm"])
    drift_joint = abs(pd["joint_error_from_drift_nm"])
    return [
        ("W: power per wavelength", m["W_column_power_max_err"] < 1e-12,
         f"max |sum_j W_ji - (1 - band-limit loss)| = {m['W_column_power_max_err']:.1e} (limit 1e-12)"),
        ("y = W s equals the Stage 1 broadband image", m["y_vs_stage1_broadband_max_rel"] < 1e-12,
         f"max rel. difference {m['y_vs_stage1_broadband_max_rel']:.1e} (limit 1e-12)"),
        ("g: central difference converged in delta", m["g_fd_vs_fd_half_delta"] < 1e-4,
         f"delta vs delta/2: {m['g_fd_vs_fd_half_delta']:.1e} (limit 1e-4)"),
        ("g: Eq. (7) vs Eq. (8)", m["g_fd_vs_analytic"] < 1e-4 and m["g_analytic_vs_autograd"] < 1e-10,
         f"FD vs analytic {m['g_fd_vs_analytic']:.1e} (limit 1e-4), "
         f"autograd vs analytic {m['g_analytic_vs_autograd']:.1e} (limit 1e-10)"),
        ("Noise model: variance = y + sigma_read^2", n["max_rel_err_var"] < 0.05 and n["max_rel_err_mean"] < 5,
         f"variance error {n['max_rel_err_var']:.1e} (limit 5e-2), mean within "
         f"{n['max_rel_err_mean']:.1f} standard errors (limit 5)"),
        ("Gauss-Newton removes the linearization bias", lin["gauss_newton_max_abs_err_nm"] < 1e-9,
         f"max error {lin['gauss_newton_max_abs_err_nm']:.1e} nm over +-2 gamma (limit 1e-9)"),
        ("Monte Carlo matches the exact linear-estimator error", mc_err < 0.02,
         f"max |RMS_MC / RMS_predicted - 1| = {mc_err:.3f} over all M (limit 0.02, "
         f"{E2.mc_trials * len(E2.test_shifts_gammas)} samples per M)"),
        ("Monte Carlo RMS reaches the Cramer-Rao bound", rms_ratio < 0.05,
         f"max |RMS/CRB - 1| = {rms_ratio:.3f} over all M (limit 0.05); "
         f"M = {last['M']}: RMS {last['rms_gls_nm']:.4f} nm, CRB {last['crb_nm']:.4f} nm"),
        ("Joint amplitude fit removes a 1 % power drift", drift_joint < 1e-6 < drift_lin,
         f"drift-induced shift error {drift_lin:.3f} nm (Eq. 10) -> {drift_joint:.1e} nm (Eq. 13); "
         f"CRB cost {pd['crb_shift_only_nm']:.4f} -> {pd['crb_joint_nm']:.4f} nm"),
    ]


def run(cfg=CFG, e2=E2):
    RES.clear()
    b = build(cfg, e2)
    check_model(b, cfg)
    check_noise(b, e2)
    lin = linearity(b, e2)
    mc = monte_carlo(b, e2)
    info = information(b, e2)
    ab = stage1_two_region(b, cfg, e2)
    power_drift(b, e2)
    RES["config"] = dict(Sn_nm_per_riu=e2.Sn_nm_per_riu, delta_gammas=e2.delta_gammas,
                         gamma_nm=b["sensor"].gamma, test_shifts_gammas=list(e2.test_shifts_gammas),
                         gn_iterations=e2.gn_iterations)
    return b, lin, mc, ab, info


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b, lin, mc, ab, info = run()
    fig_weighting(b)
    fig_camera(b)
    fig_sensitivity(b)
    fig_estimate(b, lin, mc)
    fig_rms(mc, ab)
    fig_delta_n(b, mc)
    fig_information(b, info)
    write_tables(b, mc)
    with open(OUT / "embodiment2.json", "w") as fh:
        slim = {k: v for k, v in RES.items() if k not in ("linearity", "information")}
        slim["information"] = {k: v for k, v in RES["information"].items()
                               if k.startswith("crb_at")}
        slim["linearity"] = {k: v for k, v in RES["linearity"].items()
                             if k not in ("shift_nm", "linear_estimate_nm", "gauss_newton_nm")}
        results = checks()
        slim["checks"] = [dict(name=name, passed=bool(ok), detail=detail)
                          for name, ok, detail in results]
        json.dump(slim, fh, indent=2)
    m, last = RES["model"], mc["rows"][-1]
    print(f"camera {m['roi_pixels'][0]} x {m['roi_pixels'][1]} pixels of {m['camera_pitch_um']:.0f} um, "
          f"{m['n_lambda']} wavelengths; {E2.photons:.1e} e- per measurement, "
          f"peak pixel {m['peak_pixel_e_per_frame']:.3g} e-/frame")
    print(f"{'M':>6} {'CRB(0)':>9} {'RMS GLS':>9} {'CRB loc':>9} {'RMS GN':>9} {'sigma_dn [RIU]':>15}")
    for r in mc["rows"]:
        print(f"{r['M']:6d} {r['crb_nm']:9.4f} {r['rms_gls_nm']:9.4f} {r['crb_local_nm']:9.4f} "
              f"{r['rms_gauss_newton_nm']:9.4f} {r['crb_dn_riu']:15.2e}")
    i0 = RES["information"]
    print(f"local CRB shift-only / amplitude-robust: at 0 {i0['crb_at_0_nm'][0]:.4f} / "
          f"{i0['crb_at_0_nm'][1]:.4f} nm, at +gamma {i0['crb_at_1gamma_nm'][0]:.4f} / "
          f"{i0['crb_at_1gamma_nm'][1]:.4f} nm")
    print(f"Stage 1 two-region (A, B) readout, same photons: CRB {ab:.3f} nm")
    print(f"linear range (|bias| < CRB, all pixels): {RES['linearity']['linear_range_nm_by_M'][str(last['M'])]} nm")
    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print()
    if all(ok for _, ok, _ in results):
        print(f"Embodiment 2: all {len(results)} checks passed in {time.time() - t0:.0f} s. "
              f"Figures and tables saved to {OUT}")
    else:
        print(f"{sum(not ok for _, ok, _ in results)} of {len(results)} Embodiment 2 checks FAILED.")
        sys.exit(1)
