"""
propagation.py -- input field and wavelength-dependent Fraunhofer and
angular-spectrum propagation.

Stage 1, Steps 3 and 4.

    E_in(x,y)      = E0 exp[-(x^2 + y^2) / w0^2]              (Step 3)
    E_0+(x,y)      = E_in(x,y) exp[i phi(x,y)]
    E~(kx,ky,z;l)  = E~(kx,ky,0;l) exp[i z sqrt(k^2 - kx^2 - ky^2)],  k = 2 pi / l
                                                   (angular spectrum, exact)
    E(x,y;z)       ~ F{E_0+}(x / (l z), y / (l z)) / (i l z)
                                                   (Fraunhofer, z >> z_R)

The same spatial grid is used at every wavelength, so the wavelength only
enters through k = 2 pi / lambda in the transfer function. That is what
turns spectral information into spatial information: identical spatial
frequencies travel at different angles for different wavelengths.

Conventions
-----------
* SI units (metres). Time dependence exp(-i omega t), so a field with
  spatial frequency +kx travels towards +x.
* Grid index i sits at x_i = (i - N/2) dx. Arrays are indexed [y, x].
* The FFT window is periodic. The optional band limit (Matsushima &
  Shimobaba, Opt. Express 17, 19662, 2009) removes spatial frequencies
  whose transfer-function phase is undersampled; that light would leave
  the window at z anyway. Energy accounting reports how much is removed.
"""

from dataclasses import dataclass

import torch

from doe import upsample

DTYPE = torch.float64
CDTYPE = torch.complex128


@dataclass
class SimGrid:
    """Square simulation grid: n x n samples of spacing dx."""
    n: int
    dx: float

    @property
    def length(self):
        return self.n * self.dx

    @property
    def x(self):
        return (torch.arange(self.n, dtype=DTYPE) - self.n // 2) * self.dx

    @property
    def kx(self):
        """Angular spatial frequencies in FFT order, shape (n,)."""
        return 2 * torch.pi * torch.fft.fftfreq(self.n, d=self.dx, dtype=DTYPE)

    @property
    def cell_area(self):
        return self.dx**2


def grid_for_doe(n_doe, pitch, upsample_factor=1, pad_factor=1):
    """Simulation grid for an n_doe x n_doe DOE of the given pixel pitch.

    upsample_factor: simulation samples per DOE pixel (per axis).
    pad_factor:      window size / DOE size.
    """
    return SimGrid(n=n_doe * upsample_factor * pad_factor,
                   dx=pitch / upsample_factor)


def embed_phase(phi_doe, grid, upsample_factor):
    """Place a DOE phase map on the simulation grid (phase 0 outside)."""
    phi = upsample(phi_doe, upsample_factor)
    m = phi.shape[0]
    if m == grid.n:
        return phi
    out = torch.zeros(grid.n, grid.n, dtype=DTYPE)
    o = (grid.n - m) // 2
    out[o:o + m, o:o + m] = phi
    return out


# ----------------------------------------------------------------------------
# Step 3: input field and the DOE
# ----------------------------------------------------------------------------

def gaussian_beam(grid, w0, E0=1.0, x0=0.0, y0=0.0):
    """E_in = E0 exp[-((x-x0)^2 + (y-y0)^2) / w0^2]; w0 is the 1/e^2
    intensity radius at the waist (the DOE plane)."""
    x = grid.x
    r2 = (x[None, :] - x0) ** 2 + (x[:, None] - y0) ** 2
    return (E0 * torch.exp(-r2 / w0**2)).to(CDTYPE)


def apply_doe(E_in, phi):
    """Field just after the DOE, E_0+ = E_in exp(i phi)."""
    return E_in * torch.exp(1j * phi)


# ----------------------------------------------------------------------------
# Step 4: angular-spectrum propagation
# ----------------------------------------------------------------------------

def transfer_function(grid, lam, z, band_limit=True):
    """H(kx, ky; lambda) = exp(i kz z), shape (Nl, n, n) for lam of shape (Nl,).

    Evanescent components (kx^2 + ky^2 > k^2) get imaginary kz and decay.
    """
    lam = torch.as_tensor(lam, dtype=DTYPE).reshape(-1, 1, 1)
    kx = grid.kx
    kr2 = kx[None, :] ** 2 + kx[:, None] ** 2
    k = 2 * torch.pi / lam
    kz = torch.sqrt((k**2 - kr2).to(CDTYPE))
    H = torch.exp(1j * kz * z)
    if band_limit and z != 0:
        du = 1.0 / grid.length
        f_lim = 1.0 / (lam * ((2 * du * abs(z)) ** 2 + 1) ** 0.5)
        k_lim = 2 * torch.pi * f_lim
        keep = (kx[None, None, :].abs() <= k_lim) & (kx[None, :, None].abs() <= k_lim)
        H = H * keep
    return H


def angular_spectrum(E0, grid, lam, z, band_limit=True):
    """Propagate E0 (n, n) by distance z at each wavelength in lam.

    Returns (Nl, n, n) for array lam, or (n, n) for scalar lam.
    """
    scalar = torch.as_tensor(lam).ndim == 0
    A = torch.fft.fft2(E0)
    E = torch.fft.ifft2(A * transfer_function(grid, lam, z, band_limit))
    return E[0] if scalar else E


def fraunhofer(E0, grid, lam, z, out_grid=None):
    """Fraunhofer (far-field) propagation onto a fixed detector grid.

        E(x,y;z) = exp(ikz) exp[ik(x^2+y^2)/2z] / (i lambda z)
                   * sum E0(x',y') exp[-i 2 pi (x x' + y y') / (lambda z)] dx'^2

    The transform is evaluated as a matrix Fourier transform, so the
    detector coordinates (out_grid, default: the input grid) are the same
    at every wavelength; the wavelength sets which spatial frequency lands
    on each detector point, f = x / (lambda z). Valid only in the far
    field, z >> z_R = pi w0^2 / lambda.

    Returns (Nl, n_out, n_out) for array lam, or (n_out, n_out) for scalar lam.
    """
    out = out_grid or grid
    scalar = torch.as_tensor(lam).ndim == 0
    lam = torch.as_tensor(lam, dtype=DTYPE).reshape(-1, 1, 1)
    xs, xd = grid.x, out.x
    W = torch.exp(-2j * torch.pi * xd[None, :, None] * xs[None, None, :] / (lam * z)) * grid.dx
    F = W @ E0.to(CDTYPE) @ W.transpose(-1, -2)          # [y, x] on the detector grid
    k = 2 * torch.pi / lam
    r2 = xd[None, :] ** 2 + xd[:, None] ** 2
    E = torch.exp(1j * k * z) * torch.exp(1j * k * r2 / (2 * z)) / (1j * lam * z) * F
    return E[0] if scalar else E


def power(E, grid):
    """Optical power sum |E|^2 dA over the last two axes."""
    return (E.abs() ** 2).sum((-2, -1)) * grid.cell_area


def band_limited_fraction(E0, grid, lam, z):
    """Fraction of the power of E0 removed by the band limit at (lam, z)."""
    A2 = torch.fft.fft2(E0).abs() ** 2
    H = transfer_function(grid, lam, z, band_limit=True)
    kept = ((H.abs() > 0) * A2).sum((-2, -1)) / A2.sum()
    return 1.0 - kept


if __name__ == "__main__":
    g = grid_for_doe(256, 10e-6, upsample_factor=2)
    E = gaussian_beam(g, 200e-6)
    lam = torch.tensor([1.50e-6, 1.55e-6, 1.60e-6], dtype=DTYPE)
    Ez = angular_spectrum(E, g, lam, 50e-3, band_limit=False)
    assert Ez.shape == (3, g.n, g.n)
    # Unit-modulus transfer function conserves power (Parseval).
    assert torch.allclose(power(Ez, g), power(E, g).expand(3), rtol=1e-12)
    # z = 0 is the identity.
    assert torch.allclose(angular_spectrum(E, g, 1.55e-6, 0.0), E)
    # Fraunhofer conserves power when the far-field pattern fits the window,
    # and gives the far-field Gaussian radius w = lambda z / (pi w0).
    gf = grid_for_doe(256, 10e-6)
    Eg = gaussian_beam(gf, 200e-6)
    If = fraunhofer(Eg, gf, 1.55e-6, 0.1).abs() ** 2
    assert abs(float(If.sum() * gf.cell_area / power(Eg, gf)) - 1) < 1e-12
    prof = If.sum(0)
    w_far = 2 * ((prof * gf.x**2).sum() / prof.sum()).sqrt()
    assert abs(float(w_far) / (1.55e-6 * 0.1 / (torch.pi * 200e-6)) - 1) < 1e-9
    print("propagation.py: all self-checks passed")
