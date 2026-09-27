"""Embodiment 2: measurement model y = W s, sensitivity, noise and estimators."""

import math

import pytest
import torch

from config import CFG, E2
from estimator import (add_noise, crb, fisher_per_pixel, gauss_newton, gls_estimate,
                       joint_amplitude_estimate, local_fisher, noise_variance,
                       ols_estimate, rank_pixels, select_greedy, sensitivity_fd,
                       to_refractive_index)
from propagation import grid_for_doe
from spectral_coding import ShiftSensor, bin_power, camera_response, make_camera

D = torch.float64


# ----------------------------------------------------------------------------
# Sensor in shift form
# ----------------------------------------------------------------------------

def test_sensor_photon_budget_and_shift():
    s = ShiftSensor(photons=1e6)
    assert float(s.s(0.0).sum()) == pytest.approx(1e6, rel=1e-12)
    # The dip minimum moves by the requested shift.
    fine = ShiftSensor(n_lambda=6001)
    for d in (-2.5, 0.0, 1.3):
        lam_min = float(fine.lam_nm[torch.argmin(fine.transmission(d))])
        assert abs(lam_min - (CFG.lambda0_nm + d)) <= float(fine.lam_nm[1] - fine.lam_nm[0])


def test_sensor_derivatives_agree():
    s = ShiftSensor()
    for d in (0.0, 0.7, -1.9):
        an, ag = s.ds_analytic(d), s.ds_autograd(d)
        assert torch.allclose(an, ag, rtol=1e-10, atol=1e-10 * float(an.abs().max()))
        fd = (s.s(d + 1e-4) - s.s(d - 1e-4)) / 2e-4
        assert torch.allclose(fd, an, rtol=0, atol=1e-6 * float(an.abs().max()))
    batch = s.ds_analytic(torch.tensor([0.0, 0.7], dtype=D))
    assert batch.shape == (2, s.lam_nm.numel())
    assert torch.allclose(batch[1], s.ds_analytic(0.7))


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------

def test_camera_binning_and_roi():
    g = grid_for_doe(CFG.n_doe, CFG.pitch, CFG.upsample, CFG.pad)
    cam = make_camera(g)
    assert cam.bin * g.dx == pytest.approx(E2.camera_pitch)
    assert cam.shape == (E2.roi_pixels, E2.roi_pixels)
    I = torch.rand(g.n, g.n, dtype=D)
    P = bin_power(I, g, cam.bin)
    assert float(P.sum()) == pytest.approx(float(I.sum()) * g.cell_area, rel=1e-12)
    assert cam.index(1, 2) == len(cam.cols) + 2


def test_camera_rejects_bad_pitch():
    from config import Embodiment2Config
    g = grid_for_doe(CFG.n_doe, CFG.pitch, CFG.upsample, CFG.pad)
    with pytest.raises(ValueError):
        make_camera(g, e2=Embodiment2Config(camera_pitch=7e-6))
    with pytest.raises(ValueError):
        make_camera(g, e2=Embodiment2Config(roi_pixels=10_000))


# ----------------------------------------------------------------------------
# Estimators on a synthetic linear model (exact answers known)
# ----------------------------------------------------------------------------

@pytest.fixture
def linear_model():
    torch.manual_seed(0)
    mu = 1e4 * torch.rand(50, dtype=D) + 100
    g = 200 * torch.randn(50, dtype=D)
    return mu, g, noise_variance(mu, 100.0)


def test_linear_estimators_exact_without_noise(linear_model):
    mu, g, var = linear_model
    y = mu + 0.37 * g
    assert float(gls_estimate(y, mu, g, var)[0]) == pytest.approx(0.37, rel=1e-12)
    assert float(ols_estimate(y, mu, g)[0]) == pytest.approx(0.37, rel=1e-12)
    idx = torch.tensor([3, 7, 11])
    assert float(gls_estimate(y, mu, g, var, idx)[0]) == pytest.approx(0.37, rel=1e-12)
    # Batched over leading dimensions.
    Y = torch.stack([mu + 0.1 * g, mu - 0.2 * g])
    assert torch.allclose(gls_estimate(Y, mu, g, var)[0], torch.tensor([0.1, -0.2], dtype=D))


def test_fisher_crb_and_ranking(linear_model):
    mu, g, var = linear_model
    F = fisher_per_pixel(g, var)
    order, Fs = rank_pixels(g, var)
    assert torch.all(Fs[:-1] >= Fs[1:])
    assert crb(g, var) == pytest.approx(float(F.sum()) ** -0.5)
    assert crb(g, var, order[:5]) == pytest.approx(float(Fs[:5].sum()) ** -0.5)
    assert gls_estimate(mu, mu, g, var)[1] == pytest.approx(crb(g, var))
    # Equal noise: ranking by F is ranking by |g|.
    o2, _ = rank_pixels(g, torch.ones_like(g))
    assert torch.equal(o2, torch.argsort(g.abs(), descending=True))
    # Greedy selection with a diagonal covariance picks the same set.
    greedy = select_greedy(g[:12], torch.diag(var[:12]), 4)
    assert set(greedy.tolist()) == set(rank_pixels(g[:12], var[:12])[0][:4].tolist())


def test_gls_reaches_crb_in_monte_carlo(linear_model):
    mu, g, var = linear_model
    gen = torch.Generator().manual_seed(3)
    y = add_noise(mu + 0.05 * g, 100.0, n=40000, generator=gen)
    d, sigma = gls_estimate(y, mu, g, var)
    assert float(d.mean()) == pytest.approx(0.05, abs=4 * sigma / 200)
    assert float(d.std()) == pytest.approx(sigma, rel=0.03)


def test_noise_statistics():
    gen = torch.Generator().manual_seed(1)
    y = torch.tensor([0.0, 50.0, 1e4, 1e6], dtype=D)
    ym = add_noise(y, 400.0, n=200000, generator=gen)
    assert torch.allclose(ym.mean(0), y, atol=0.1, rtol=1e-3)
    assert torch.allclose(ym.var(0), y + 400.0, rtol=0.02)


def test_joint_amplitude_is_power_invariant(linear_model):
    mu, g, var = linear_model
    y = mu + 0.3 * g
    d1, a1 = joint_amplitude_estimate(y, mu, g, var)
    d2, a2 = joint_amplitude_estimate(1.07 * y, mu, g, var)
    assert float(d1) == pytest.approx(0.3, rel=1e-10) and float(a1) == pytest.approx(1.0)
    assert float(d2) == pytest.approx(float(d1), rel=1e-12) and float(a2) == pytest.approx(1.07)
    # The shift-only estimator reads the power change as a false shift.
    assert abs(float(gls_estimate(1.07 * y, mu, g, var)[0]) - 0.3) > 0.01


def test_local_fisher_nuisance(linear_model):
    mu, g, var = linear_model
    assert local_fisher(g, var) == pytest.approx(1 / crb(g, var) ** 2)
    # If g is proportional to mu, the shift is indistinguishable from a power change.
    assert local_fisher(2e-3 * mu, var, mu=mu) == pytest.approx(0.0, abs=1e-6)
    # Nuisance can only remove information.
    assert local_fisher(g, var, mu=mu) <= local_fisher(g, var) * (1 + 1e-12)


def test_gauss_newton_and_fd_on_nonlinear_model():
    lam = torch.linspace(-10, 10, 81, dtype=D)
    W = torch.stack([torch.exp(-((lam - c) / 6) ** 2) for c in (-6, -2, 2, 6)])

    def s(d):
        d = torch.as_tensor(d, dtype=D)
        return 1 - 0.8 / (1 + ((lam - (d.unsqueeze(-1) if d.ndim else d)) / 2) ** 2)

    def ds(d):
        d = torch.as_tensor(d, dtype=D)
        x = (lam - (d.unsqueeze(-1) if d.ndim else d)) / 2
        return -0.8 * 2 * x / (2 * (1 + x ** 2) ** 2)

    model = lambda d: camera_response(W, s(d))
    jac = lambda d: camera_response(W, ds(d))
    g = sensitivity_fd(model, 1e-3)
    assert torch.allclose(g, jac(torch.tensor([0.0], dtype=D))[0], rtol=1e-5, atol=1e-9)
    truth = torch.tensor([-1.5, 0.4, 1.2], dtype=D)
    est = gauss_newton(model(truth), model, jac, torch.ones(4, dtype=D), iterations=30)
    assert torch.allclose(est, truth, atol=1e-9)
    # Bounds and step limits are honoured.
    est_b = gauss_newton(model(truth), model, jac, torch.ones(4, dtype=D), iterations=2,
                         max_step=0.1, bounds=(-1.0, 1.0))
    assert float(est_b.abs().max()) <= 0.2 + 1e-12


def test_refractive_index_conversion():
    dn, sdn = to_refractive_index(torch.tensor(0.5, dtype=D), 0.02, 200.0)
    assert float(dn) == pytest.approx(2.5e-3) and sdn == pytest.approx(1e-4)


# ----------------------------------------------------------------------------
# Real grating model (one build of W, about 10 s)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grating():
    import embodiment2
    b = embodiment2.build()
    embodiment2.RES.clear()
    embodiment2.check_model(b)
    return b, embodiment2.RES["model"]


def test_W_conserves_power_and_matches_stage1(grating):
    b, m = grating
    assert b["W"].shape == (E2.roi_pixels ** 2, CFG.n_lambda)
    assert bool((b["W"] >= 0).all())
    assert m["W_column_power_max_err"] < 1e-12
    assert m["y_vs_stage1_broadband_max_rel"] < 1e-12
    assert m["roi_fraction_of_camera_power_min"] > 0.999


def test_sensitivity_cross_checks(grating):
    b, m = grating
    assert m["g_fd_vs_fd_half_delta"] < 1e-4
    assert m["g_fd_vs_analytic"] < 1e-4
    assert m["g_analytic_vs_autograd"] < 1e-10
    # Source centred on the resonance: total power is (nearly) stationary at
    # d = 0; the residual comes from the asymmetric wavelength window.
    assert abs(float(b["g"].sum())) < 1e-3 * float(b["g"].abs().sum())


def test_linear_response_is_W_times_s(grating):
    b, _ = grating
    s = b["sensor"]
    y = b["response"](0.8)
    assert torch.allclose(y, b["W"] @ s.s(0.8))
    # Superposition over spectra.
    assert torch.allclose(camera_response(b["W"], s.s(0.8) + s.s(-0.3)),
                          y + b["response"](-0.3))


@pytest.mark.slow
def test_embodiment2_all_checks(tmp_path, monkeypatch):
    import embodiment2
    import main
    monkeypatch.setattr(main, "OUT", tmp_path)
    monkeypatch.setattr(embodiment2, "OUT", tmp_path)
    embodiment2.run()
    failed = [(name, detail) for name, ok, detail in embodiment2.checks() if not ok]
    assert not failed, failed
    rows = embodiment2.RES["monte_carlo"]["rows"]
    assert rows[-1]["M"] == E2.roi_pixels ** 2
    assert all(r1["crb_nm"] >= r2["crb_nm"] for r1, r2 in zip(rows, rows[1:]))
    assert math.isfinite(embodiment2.RES["stage1_two_region"]["crb_nm"])
