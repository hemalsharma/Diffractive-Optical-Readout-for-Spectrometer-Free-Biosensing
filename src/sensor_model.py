"""
sensor_model.py -- resonance spectrum and concentration-to-wavelength mapping.

Stage 1, Step 1 of the diffractive-biosensing project. Maps an analyte
concentration c to the optical spectrum that leaves the sensor:

    lambda_r(c)  = lambda0 + alpha * c                          (sensor law)
    T(lambda; c) = 1 - A / (1 + ((lambda - lambda_r(c)) / gamma)^2)
    S(lambda; c) = S_source(lambda) * T(lambda; c)

Conventions
-----------
* Normalized units: lambda0 = 1 by default and every wavelength is in units
  of lambda0. PhysicalUnits maps them to nm (e.g. lambda0 = 1550 nm) only
  for reporting; the physics never depends on the choice.
* gamma is the Lorentzian half-width at half-maximum (HWHM).
* Difficulty of a sensing task: eta = delta_lambda / gamma.
* Shapes broadcast: c of shape (Nc,) with lam of shape (Nl,) gives (Nc, Nl);
  scalar c gives (Nl,). Everything is differentiable in c.
* This module knows nothing about the DOE. Downstream code only needs
  S(lambda; c), so the sensor can be swapped without touching the optics.

Layout
------
  ResonanceParams        sensor parameters
  PhysicalUnits          normalized <-> nm conversion and figures of merit
  resonance_wavelength   c -> lambda_r
  transmission           T(lambda; c)
  flat_source, gaussian_source
  sensor_spectrum        S(lambda; c)
  spectral_sensitivity   dS/dc (autograd, forward mode)
  wavelength_grid        sampling window covering the resonance
  eta, shift_for_eta, concentration_for_shift
  SensorModel            bundles params + grid + source; used downstream
"""

from dataclasses import dataclass

import torch
from torch.func import jvp

DTYPE = torch.float64


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

@dataclass
class ResonanceParams:
    """Parameters of the Lorentzian transmission dip (normalized units)."""
    lambda0: float = 1.0   # zero-analyte resonance wavelength
    gamma: float = 1e-3    # linewidth (HWHM)
    depth: float = 0.8     # resonance depth A, 0 <= A <= 1
    alpha: float = 1e-3    # intrinsic sensitivity d(lambda_r)/dc

    def __post_init__(self):
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if not 0.0 <= self.depth <= 1.0:
            raise ValueError("depth A must lie in [0, 1] for a passive sensor")

    @property
    def q_factor(self):
        """Quality factor Q = lambda0 / FWHM = lambda0 / (2 gamma)."""
        return self.lambda0 / (2.0 * self.gamma)


@dataclass
class PhysicalUnits:
    """Maps normalized wavelengths to nm. Reporting only."""
    lambda0_nm: float = 1550.0

    def to_nm(self, lam, p: ResonanceParams):
        """Absolute wavelength in nm."""
        return self.lambda0_nm * _as_tensor(lam) / p.lambda0

    def length_nm(self, dlam, p: ResonanceParams):
        """Wavelength interval (shift, linewidth) in nm."""
        return self.lambda0_nm * dlam / p.lambda0

    def summary(self, p: ResonanceParams):
        return {
            "lambda0 [nm]": self.lambda0_nm,
            "HWHM gamma [nm]": self.length_nm(p.gamma, p),
            "FWHM [nm]": self.length_nm(2 * p.gamma, p),
            "Q factor": p.q_factor,
            "alpha [nm per unit c]": self.length_nm(p.alpha, p),
            "depth A": p.depth,
        }


# ----------------------------------------------------------------------------
# Tensor helpers
# ----------------------------------------------------------------------------

def _as_tensor(x):
    return torch.as_tensor(x, dtype=DTYPE)


def _pair(c, lam):
    """Broadcast c against lam: (Nc,) x (Nl,) -> (Nc, 1) and (Nl,)."""
    c, lam = _as_tensor(c), _as_tensor(lam)
    if c.ndim >= 1:
        c = c.unsqueeze(-1)
    return c, lam


# ----------------------------------------------------------------------------
# Concentration -> resonance -> transmission
# ----------------------------------------------------------------------------

def resonance_wavelength(c, p: ResonanceParams):
    """lambda_r(c) = lambda0 + alpha * c."""
    return p.lambda0 + p.alpha * _as_tensor(c)


def transmission(lam, c, p: ResonanceParams):
    """Lorentzian transmission dip T(lambda; c)."""
    c, lam = _pair(c, lam)
    detuning = (lam - resonance_wavelength(c, p)) / p.gamma
    return 1.0 - p.depth / (1.0 + detuning**2)


# ----------------------------------------------------------------------------
# Source spectra
# ----------------------------------------------------------------------------

def flat_source(lam, amplitude=1.0):
    """Spectrally flat source over the sampled window."""
    return amplitude * torch.ones_like(_as_tensor(lam))


def gaussian_source(lam, center, fwhm, amplitude=1.0):
    """Gaussian source spectrum, e.g. a superluminescent diode."""
    lam = _as_tensor(lam)
    sigma = fwhm / (2.0 * (2.0 * torch.log(_as_tensor(2.0))) ** 0.5)
    return amplitude * torch.exp(-0.5 * ((lam - center) / sigma) ** 2)


# ----------------------------------------------------------------------------
# Sensor output spectrum and its sensitivity
# ----------------------------------------------------------------------------

def sensor_spectrum(lam, c, p: ResonanceParams, source=None):
    """S(lambda; c) = S_source(lambda) * T(lambda; c).

    source: tensor of shape (Nl,) sampled on lam. Defaults to a flat source.
    """
    if source is None:
        source = flat_source(lam)
    return _as_tensor(source) * transmission(lam, c, p)


def spectral_sensitivity(lam, c, p: ResonanceParams, source=None):
    """dS(lambda; c)/dc, same shape as sensor_spectrum.

    Forward-mode autograd: each spectrum depends only on its own c, so one
    jvp with unit tangent gives the derivative for every c at once. This is
    where all the information about c lives; later stages turn it into
    detector signals.
    """
    c = _as_tensor(c)
    _, dS = jvp(lambda cc: sensor_spectrum(lam, cc, p, source),
                (c,), (torch.ones_like(c),))
    return dS


# ----------------------------------------------------------------------------
# Wavelength grid and difficulty parameter
# ----------------------------------------------------------------------------

def wavelength_grid(p: ResonanceParams, n=201, half_width_gammas=10.0,
                    c_range=(0.0, 0.0)):
    """Uniform wavelength grid covering the resonance for all c in c_range.

    The window spans half_width_gammas linewidths beyond the extreme
    resonance positions. Returns (lam, d_lam). n is a starting value; the
    final choice must come from the Stage 1 convergence study.
    """
    lr = resonance_wavelength(torch.tensor(c_range, dtype=DTYPE), p)
    lo = lr.min() - half_width_gammas * p.gamma
    hi = lr.max() + half_width_gammas * p.gamma
    lam = torch.linspace(float(lo), float(hi), n, dtype=DTYPE)
    return lam, lam[1] - lam[0]


def eta(delta_lambda, p: ResonanceParams):
    """Normalized difficulty eta = delta_lambda / gamma."""
    return delta_lambda / p.gamma


def shift_for_eta(eta_value, p: ResonanceParams):
    """Resonance shift delta_lambda that gives the requested eta."""
    return eta_value * p.gamma


def concentration_for_shift(delta_lambda, p: ResonanceParams):
    """Concentration that produces a resonance shift delta_lambda."""
    return delta_lambda / p.alpha


# ----------------------------------------------------------------------------
# Bundled model used by the rest of the project
# ----------------------------------------------------------------------------

class SensorModel:
    """Sensor parameters + wavelength grid + source spectrum.

    Downstream modules (propagation, detector, optimization) take a
    SensorModel and call .spectrum(c) on its .lam grid.
    """

    def __init__(self, params: ResonanceParams, lam, source=None):
        self.params = params
        self.lam = _as_tensor(lam)
        self.dlam = self.lam[1] - self.lam[0]
        self.source = flat_source(self.lam) if source is None else _as_tensor(source)

    @classmethod
    def build(cls, params=None, c_range=(0.0, 1.0), n=201,
              half_width_gammas=10.0, source="flat", source_fwhm_gammas=50.0):
        """Convenience constructor. source: 'flat' or 'gaussian'."""
        p = params or ResonanceParams()
        lam, _ = wavelength_grid(p, n, half_width_gammas, c_range)
        if source == "flat":
            src = flat_source(lam)
        elif source == "gaussian":
            src = gaussian_source(lam, p.lambda0, source_fwhm_gammas * p.gamma)
        else:
            raise ValueError("source must be 'flat' or 'gaussian'")
        return cls(p, lam, src)

    def resonance(self, c):
        return resonance_wavelength(c, self.params)

    def transmission(self, c):
        return transmission(self.lam, c, self.params)

    def spectrum(self, c):
        return sensor_spectrum(self.lam, c, self.params, self.source)

    def sensitivity(self, c):
        return spectral_sensitivity(self.lam, c, self.params, self.source)

    def binary_spectra(self, eta_value):
        """(S0, S1) for H0: no analyte and H1: shift of eta * gamma."""
        c1 = concentration_for_shift(shift_for_eta(eta_value, self.params),
                                     self.params)
        return self.spectrum(0.0), self.spectrum(c1)

    def concentration_sweep(self, c_min, c_max, n):
        """(c, S) with c of shape (n,) and S of shape (n, Nl)."""
        c = torch.linspace(c_min, c_max, n, dtype=DTYPE)
        return c, self.spectrum(c)

    def total_power(self, c):
        """Integral of S over the grid, i.e. power reaching the DOE."""
        return (self.spectrum(c) * self.dlam).sum(-1)


# ----------------------------------------------------------------------------
# Self-check:  python sensor_model.py
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    p = ResonanceParams()
    lam, dlam = wavelength_grid(p, n=2001, c_range=(0.0, 1.0))

    # 1. Dip at lambda_r with depth A; half depth at lambda_r +/- gamma.
    for c0 in (0.0, 0.5, 1.0):
        lr = float(resonance_wavelength(c0, p))
        T = transmission(torch.tensor([lr, lr + p.gamma, lr - p.gamma],
                                      dtype=DTYPE), c0, p)
        assert torch.allclose(T, torch.tensor([1 - p.depth, 1 - p.depth / 2,
                                               1 - p.depth / 2], dtype=DTYPE))
        assert abs(float(lam[transmission(lam, c0, p).argmin()]) - lr) <= dlam

    # 2. Passivity: 0 <= S <= S_source, and batching shape.
    cs = torch.linspace(0, 1, 11, dtype=DTYPE)
    src = gaussian_source(lam, p.lambda0, fwhm=50 * p.gamma)
    S = sensor_spectrum(lam, cs, p, src)
    assert S.shape == (11, lam.numel())
    assert (S >= 0).all() and (S <= src + 1e-15).all()

    # 3. eta -> shift -> concentration round-trip.
    for e in (1, 0.5, 0.2, 0.1, 0.05, 0.02):
        dl = shift_for_eta(e, p)
        assert abs(eta(dl, p) - e) < 1e-12
        c1 = concentration_for_shift(dl, p)
        assert abs(float(resonance_wavelength(c1, p)) - p.lambda0 - dl) < 1e-15

    # 4. Autograd dS/dc matches the analytic derivative, for a batch of c.
    x = (lam - resonance_wavelength(cs, p).unsqueeze(-1)) / p.gamma
    analytic = src * (-2 * p.depth * x / (1 + x**2) ** 2) * (p.alpha / p.gamma)
    assert torch.allclose(spectral_sensitivity(lam, cs, p, src), analytic)

    # 5. SensorModel reproduces the functional API.
    m = SensorModel(p, lam, src)
    assert torch.equal(m.spectrum(cs), S)
    S0, S1 = m.binary_spectra(0.1)
    assert torch.allclose(S0, sensor_spectrum(lam, 0.0, p, src))
    assert torch.allclose(S1, sensor_spectrum(lam, 0.1 * p.gamma / p.alpha, p, src))

    print("sensor_model.py: all self-checks passed")
    for k, v in PhysicalUnits().summary(p).items():
        print(f"  {k:24s} {v:g}")
