"""
spectral_coding.py -- Embodiment 2 pixel-wise measurement model, y = W s.

    lambda_r = lambda_r0 + d,  d = S_n dn                          (Eq. 1)
    T(lambda; d) = 1 - A / (1 + ((lambda - lambda_r0 - d) / gamma)^2)   (Eq. 2)
    s_i(d) = s_src(lambda_i) T(lambda_i; d) dlambda                (Eq. 3, 5)
    W_ji = integral over pixel j of |E(x, y; lambda_i)|^2,
           with a unit-power input beam                            (Eq. 4)
    y(d) = W s(d)                                                  (Eq. 5)

Here d is the resonance shift in nm. Everything reuses the Stage 1 pieces:
the Lorentzian SensorModel and its wavelength grid (config.CFG), the
band-limited angular-spectrum propagation and the phase-only DOE. Camera
pixels are blocks of `bin` x `bin` simulation cells; only a region of
interest (ROI) around the +1 order is kept as measurements.

Conventions: wavelengths in nm unless the name ends in _m; y and s in
photoelectrons; W is dimensionless (fraction of the power at lambda_i
that lands on pixel j). Arrays are [y, x] like the rest of the project.
"""

from dataclasses import dataclass

import torch

from config import CFG, E2
from doe import linear_grating
from main import spot_position
from propagation import (DTYPE, angular_spectrum, apply_doe, embed_phase,
                         gaussian_beam, grid_for_doe, power)


# ----------------------------------------------------------------------------
# Sensor, parameterized by the resonance shift d [nm]
# ----------------------------------------------------------------------------

class ShiftSensor:
    """Stage 1 Lorentzian sensor on the Stage 1 wavelength grid, in terms of d.

    Wraps config.CFG.sensor(): the concentration c of Stage 1 maps to the
    shift by d = c * shift_per_c_nm, so the existing spectrum and its
    autograd derivative are reused unchanged. `photons` scales the source so
    that sum_i s_i(0) = photons (photoelectrons at zero shift).
    """

    def __init__(self, cfg=CFG, photons=E2.photons, n_lambda=None):
        self.cfg = cfg
        self.model = cfg.sensor(n_lambda)
        self.lam_nm = self.model.lam * cfg.lambda0_nm
        self.lam_m = cfg.lam_m(self.model)
        self.dlam_nm = cfg.dlam_nm(self.model)
        self.lambda_r0 = cfg.lambda0_nm
        self.gamma = 0.5 * cfg.fwhm_nm
        self.depth = cfg.depth
        self._c_per_nm = 1.0 / cfg.shift_per_c_nm
        s0 = self.model.spectrum(0.0) * self.dlam_nm
        self.scale = photons / float(s0.sum())

    def transmission(self, d):
        return self.model.transmission(torch.as_tensor(d, dtype=DTYPE) * self._c_per_nm)

    def s(self, d):
        """Measurement vector s_i(d) = s_src T dlambda [e-]; (Nl,) or (Nd, Nl)."""
        c = torch.as_tensor(d, dtype=DTYPE) * self._c_per_nm
        return self.scale * self.model.spectrum(c) * self.dlam_nm

    def ds_analytic(self, d=0.0):
        """ds/dd from the analytic derivative of Eq. (2)  (Eq. 8).

        d scalar gives (Nl,); d of shape (Nd,) gives (Nd, Nl).
        """
        d = torch.as_tensor(d, dtype=DTYPE)
        if d.ndim:
            d = d.unsqueeze(-1)
        x = (self.lam_nm - self.lambda_r0 - d) / self.gamma
        dT = -2 * self.depth * x / (self.gamma * (1 + x ** 2) ** 2)
        return self.scale * self.model.source * dT * self.dlam_nm

    def ds_autograd(self, d=0.0):
        """ds/dd by forward-mode autograd through the Stage 1 sensor model."""
        c = torch.as_tensor(d, dtype=DTYPE) * self._c_per_nm
        return self.scale * self.model.sensitivity(c) * self.dlam_nm * self._c_per_nm


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------

@dataclass
class Camera:
    """Camera pixels = bin x bin blocks of the simulation grid.

    rows, cols: index ranges of the ROI on the binned (full-camera) grid.
    x, y:       pixel-centre coordinates of the ROI [m].
    """
    bin: int
    pitch: float
    n_full: int
    rows: range
    cols: range
    x: torch.Tensor
    y: torch.Tensor

    @property
    def shape(self):
        return (len(self.rows), len(self.cols))

    @property
    def n_pixels(self):
        return len(self.rows) * len(self.cols)

    def image(self, v):
        """Reshape an ROI vector (..., Npix) to an image (..., ny, nx)."""
        return v.reshape(*v.shape[:-1], *self.shape)

    def index(self, row, col):
        """ROI vector index of the camera pixel at ROI (row, col)."""
        return row * len(self.cols) + col


def make_camera(grid, cfg=CFG, e2=E2):
    """Camera with pitch e2.camera_pitch and a square ROI on the +1 order."""
    f = e2.camera_pitch / grid.dx
    if abs(f - round(f)) > 1e-9 or grid.n % round(f):
        raise ValueError("camera_pitch must be an integer multiple of the grid "
                         "spacing that divides the grid")
    f = round(f)
    n = grid.n // f
    # Pixel k spans simulation cells [k f, (k+1) f), centred at mean of their x.
    xc = grid.x.reshape(n, f).mean(1)
    m = e2.roi_pixels
    if m > n:
        raise ValueError("ROI larger than the camera")
    x_spot = spot_position(cfg.lambda0_nm * 1e-9, cfg)
    c0 = int(torch.argmin((xc - x_spot).abs())) - m // 2
    r0 = int(torch.argmin(xc.abs())) - m // 2
    c0, r0 = max(0, min(c0, n - m)), max(0, min(r0, n - m))
    rows, cols = range(r0, r0 + m), range(c0, c0 + m)
    return Camera(bin=f, pitch=e2.camera_pitch, n_full=n, rows=rows, cols=cols,
                  x=xc[c0:c0 + m], y=xc[r0:r0 + m])


def bin_power(I, grid, f):
    """Power per camera pixel: sum of I dA over f x f blocks; (..., n/f, n/f)."""
    n = I.shape[-1] // f
    return I.reshape(*I.shape[:-2], n, f, n, f).sum((-3, -1)) * grid.cell_area


# ----------------------------------------------------------------------------
# Spectral weighting matrix W (Eq. 4)
# ----------------------------------------------------------------------------

def unit_power_field(phi_doe, cfg=CFG):
    """Field just after the DOE for a unit-power Gaussian input."""
    grid = grid_for_doe(cfg.n_doe, cfg.pitch, cfg.upsample, cfg.pad)
    E_in = gaussian_beam(grid, cfg.w0)
    E_in = E_in / power(E_in, grid).sqrt()
    return grid, E_in, apply_doe(E_in, embed_phase(phi_doe, grid, cfg.upsample))


def weighting_matrix(lam_m, phi_doe=None, cfg=CFG, e2=E2, chunk=16):
    """W (Npix, Nl) on the ROI, plus the power reaching the whole camera.

    Each wavelength is propagated independently with the band-limited
    angular spectrum, exactly as in Stage 1. Returns a dict with
    W, W_full (Nl, n_full, n_full) for the whole camera, grid, camera, E_doe.
    """
    if phi_doe is None:
        phi_doe = linear_grating(cfg.n_doe, cfg.pitch, cfg.period)
    lam_m = torch.as_tensor(lam_m, dtype=DTYPE)
    grid, E_in, E_doe = unit_power_field(phi_doe, cfg)
    cam = make_camera(grid, cfg, e2)
    full = []
    for i in range(0, lam_m.numel(), chunk):
        I = angular_spectrum(E_doe, grid, lam_m[i:i + chunk], cfg.z).abs() ** 2
        full.append(bin_power(I, grid, cam.bin))
    W_full = torch.cat(full)                                    # (Nl, n, n)
    roi = W_full[:, cam.rows.start:cam.rows.stop, cam.cols.start:cam.cols.stop]
    W = roi.reshape(lam_m.numel(), -1).T.contiguous()           # (Npix, Nl)
    return dict(W=W, W_full=W_full, grid=grid, camera=cam, E_in=E_in, E_doe=E_doe)


def camera_response(W, s):
    """y = W s (Eq. 5) for s of shape (Nl,) or (Nd, Nl); returns (Npix,) or (Nd, Npix)."""
    return s @ W.T
