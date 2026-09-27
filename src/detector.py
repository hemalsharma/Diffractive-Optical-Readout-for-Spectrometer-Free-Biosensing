"""
detector.py -- broadband detector integration, detector regions and readout.

Stage 1, Step 5. The spectral components are mutually incoherent, so the
detector adds intensities, not fields:

    I(x,y;c) = sum_k S(lambda_k; c) |E(x,y; lambda_k)|^2 d_lambda

|E(x,y;lambda_k)|^2 does not depend on c, so the propagation is done once
per wavelength and every concentration is a weighted sum of the same
monochromatic patterns.

Region powers and the normalized differential readout
    R = (I_A - I_B) / (I_A + I_B)
are provided for the two-detector architecture of later stages.
"""

import torch

from propagation import DTYPE, angular_spectrum, fraunhofer


# ----------------------------------------------------------------------------
# Intensities
# ----------------------------------------------------------------------------

def monochromatic_intensity(E0, grid, lam, z, band_limit=True, method="angular_spectrum"):
    """|E(x,y; lambda)|^2 at distance z; (Nl, n, n) or (n, n).

    method: 'angular_spectrum' (exact, any z) or 'fraunhofer' (far field only).
    """
    if method == "fraunhofer":
        return fraunhofer(E0, grid, lam, z).abs() ** 2
    if method != "angular_spectrum":
        raise ValueError("method must be 'angular_spectrum' or 'fraunhofer'")
    return angular_spectrum(E0, grid, lam, z, band_limit).abs() ** 2


def broadband_intensity(E0, grid, lam, z, S, dlam, chunk=16, band_limit=True,
                        method="angular_spectrum"):
    """I(x,y;c) = sum_k S(lambda_k;c) |E(x,y;lambda_k)|^2 d_lambda.

    lam:  wavelengths in metres, shape (Nl,)
    S:    spectra sampled on lam, shape (Nl,) or (Nc, Nl)
    dlam: scalar wavelength step, or one quadrature weight per wavelength,
          in the units S is defined per (here nm)
    Wavelengths are processed in chunks to bound memory.
    Returns (n, n) for 1-D S, else (Nc, n, n).
    """
    lam = torch.as_tensor(lam, dtype=DTYPE)
    S = torch.as_tensor(S, dtype=DTYPE)
    if lam.ndim != 1 or lam.numel() == 0:
        raise ValueError("lam must be a non-empty 1-D wavelength grid")
    if S.ndim not in (1, 2) or S.shape[-1] != lam.numel():
        raise ValueError("S must have shape (Nl,) or (Nc, Nl), matching lam")
    if not bool(torch.isfinite(lam).all()) or not bool(torch.isfinite(S).all()):
        raise ValueError("lam and S must contain only finite values")
    if not bool((lam > 0).all()):
        raise ValueError("wavelengths must be positive")
    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError("chunk must be a positive integer")

    weights = torch.as_tensor(dlam, dtype=DTYPE)
    if weights.ndim > 1 or (weights.ndim == 1 and weights.shape != lam.shape):
        raise ValueError("dlam must be a scalar or have shape (Nl,)")
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("dlam weights must be finite and positive")

    single = S.ndim == 1
    S = S.reshape(-1, lam.numel())
    out = torch.zeros(S.shape[0], grid.n, grid.n, dtype=DTYPE)
    for i in range(0, lam.numel(), chunk):
        P = monochromatic_intensity(E0, grid, lam[i:i + chunk], z, band_limit, method)
        spectral_slice = S[:, i:i + chunk]
        if weights.ndim == 1:
            spectral_slice = spectral_slice * weights[i:i + chunk]
        out += torch.einsum("cl,lyx->cyx", spectral_slice, P)
    if weights.ndim == 0:
        out *= weights
    return out[0] if single else out


# ----------------------------------------------------------------------------
# Detector geometry and signals
# ----------------------------------------------------------------------------

def rect_mask(grid, x0, y0, wx, wy):
    """Boolean mask of a wx x wy detector centred at (x0, y0)."""
    x = grid.x
    mx = (x >= x0 - wx / 2) & (x < x0 + wx / 2)
    my = (x >= y0 - wy / 2) & (x < y0 + wy / 2)
    return my[:, None] & mx[None, :]


def region_power(I, mask, grid):
    """Power collected by a detector region: integral of I over the mask."""
    return (I * mask).sum((-2, -1)) * grid.cell_area


def differential_readout(IA, IB):
    """R = (I_A - I_B) / (I_A + I_B); insensitive to common source power."""
    return (IA - IB) / (IA + IB)


def centroid(I, grid):
    """Intensity-weighted centroid (x_c, y_c) over the last two axes."""
    x = grid.x
    tot = I.sum((-2, -1))
    xc = (I.sum(-2) * x).sum(-1) / tot
    yc = (I.sum(-1) * x).sum(-1) / tot
    return xc, yc


def second_moment_radius(I, grid, axis="x"):
    """1/e^2 radius w = 2 sigma of the intensity profile along one axis."""
    x = grid.x
    prof = I.sum(-2) if axis == "x" else I.sum(-1)
    tot = prof.sum(-1, keepdim=True)
    mean = (prof * x).sum(-1, keepdim=True) / tot
    var = (prof * (x - mean) ** 2).sum(-1) / tot.squeeze(-1)
    return 2 * var.sqrt()


def bin_pixels(I, factor):
    """Detector array with pixels `factor` x `factor` simulation cells.

    Returns the power per detector pixel divided by the cell count, i.e. the
    mean intensity per pixel, on a grid `factor` times coarser.
    """
    if factor == 1:
        return I
    lead = I.shape[:-2]
    n = I.shape[-1] // factor
    return I.reshape(*lead, n, factor, n, factor).mean((-3, -1))


def cross_section(I, grid, axis="x", offset=0.0):
    """Line profile through the pattern at y = offset (axis 'x') or x = offset."""
    j = int(torch.argmin((grid.x - offset).abs()))
    return I[..., j, :] if axis == "x" else I[..., :, j]


if __name__ == "__main__":
    from propagation import gaussian_beam, grid_for_doe, power

    g = grid_for_doe(256, 10e-6)
    E = gaussian_beam(g, 200e-6)
    lam = torch.linspace(1.54e-6, 1.56e-6, 21, dtype=DTYPE)
    # Flat unit spectrum over 20 nm: broadband power = 20 nm x single-lambda power
    # (with 21 samples and dlam = 1 nm the rectangle sum covers 21 nm).
    I = broadband_intensity(E, g, lam, 20e-3, torch.ones(21), dlam=1.0, chunk=8)
    P1 = power(E, g)
    assert torch.allclose(I.sum() * g.cell_area, 21 * P1, rtol=1e-9)
    # Chunking does not change the result.
    I2 = broadband_intensity(E, g, lam, 20e-3, torch.ones(21), dlam=1.0, chunk=21)
    assert torch.allclose(I, I2)
    # Beam centred half a cell off axis so the sample grid is mirror
    # symmetric about it: centroid there, and R = 0 for mirror-image detectors.
    h = g.dx / 2
    Eh = gaussian_beam(g, 200e-6, x0=h)
    Ih = broadband_intensity(Eh, g, lam, 20e-3, torch.ones(21), dlam=1.0)
    xc, yc = centroid(Ih, g)
    assert abs(float(xc) - h) < 1e-12 and abs(float(yc)) < 1e-12
    A = region_power(Ih, rect_mask(g, h - 150e-6, 0, 300e-6, 600e-6), g)
    B = region_power(Ih, rect_mask(g, h + 150e-6, 0, 300e-6, 600e-6), g)
    assert abs(float(differential_readout(A, B))) < 1e-9
    assert bin_pixels(I, 4).shape == (64, 64)
    print("detector.py: all self-checks passed")
