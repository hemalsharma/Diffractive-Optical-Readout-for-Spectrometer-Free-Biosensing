"""
doe.py -- phase-only diffractive optical element, t(x,y) = exp[i phi(x,y)].

Stage 1, Step 2. The DOE is a square array of n x n pixels with pitch p.
Phase maps live on this pixel grid; propagation.py resamples them onto the
(finer, possibly padded) simulation grid.

Conventions
-----------
* Pixel j spans x in [(j - n/2) p, (j - n/2 + 1) p], so the DOE is centred
  on the optical axis x = y = 0. Arrays are indexed [y, x].
* The phase is wavelength independent (ideal thin element). Material
  dispersion of a real etched DOE is a Stage 4 refinement.
* Phases are torch tensors so the same maps can later be made trainable.
"""

import math

import torch

DTYPE = torch.float64
TWO_PI = 2.0 * math.pi


def pixel_centres(n, pitch):
    """x (or y) coordinate of each pixel centre, shape (n,)."""
    return (torch.arange(n, dtype=DTYPE) - n / 2 + 0.5) * pitch


def wrap(phi):
    """Wrap phase into [0, 2 pi)."""
    return torch.remainder(phi, TWO_PI)


# ----------------------------------------------------------------------------
# Phase maps (n x n, on the DOE pixel grid)
# ----------------------------------------------------------------------------

def zero_phase(n):
    """No DOE: phi = 0 everywhere."""
    return torch.zeros(n, n, dtype=DTYPE)


def linear_grating(n, pitch, period, axis="x"):
    """Blazed linear phase grating phi = 2 pi x / d, wrapped to [0, 2 pi).

    Sends light into the +1 order at sin(theta) = lambda / d. Sampled on
    pixels, it is a staircase of period/pitch phase levels.
    """
    x = pixel_centres(n, pitch)
    ramp = wrap(TWO_PI * x / period)
    phi = ramp.expand(n, n) if axis == "x" else ramp.unsqueeze(1).expand(n, n)
    return phi.clone()


def binary_grating(n, pitch, period, duty=0.5, depth=math.pi, axis="x"):
    """Two-level grating: phase `depth` for the first `duty` of each period.

    With depth = pi and duty = 0.5 the zero order vanishes and each of the
    +/-1 orders carries 4/pi^2 = 40.5 % of the power.
    """
    x = pixel_centres(n, pitch)
    frac = torch.remainder(x / period, 1.0)
    line = depth * (frac < duty).to(DTYPE)
    phi = line.expand(n, n) if axis == "x" else line.unsqueeze(1).expand(n, n)
    return phi.clone()


# ----------------------------------------------------------------------------
# Transmission and resampling
# ----------------------------------------------------------------------------

def transmission(phi):
    """Complex transmission t = exp(i phi). |t| = 1: the mask is lossless."""
    return torch.exp(1j * phi)


def upsample(phi, factor):
    """Resample onto a grid `factor` times finer, holding each pixel constant.

    This represents the physical pixelated DOE: every pixel is a flat
    phase step of width `pitch`, not a point sample.
    """
    if factor == 1:
        return phi
    return phi.repeat_interleave(factor, 0).repeat_interleave(factor, 1)


if __name__ == "__main__":
    n, p = 256, 10e-6
    phi = linear_grating(n, p, period=160e-6)
    assert phi.shape == (n, n) and float(phi.min()) >= 0 and float(phi.max()) < TWO_PI
    # 16 pixels per period -> 16 distinct phase levels, repeating every 16 px.
    assert torch.allclose(phi[:, :16], phi[:, 16:32])
    assert len(torch.unique(torch.round(phi[0], decimals=9))) == 16
    b = binary_grating(n, p, period=160e-6)
    assert torch.allclose(b[0, :16], torch.tensor([math.pi] * 8 + [0.0] * 8, dtype=DTYPE)) \
        or torch.allclose(b[0, :16], torch.tensor([0.0] * 8 + [math.pi] * 8, dtype=DTYPE))
    assert torch.allclose(transmission(phi).abs(), torch.ones(n, n, dtype=DTYPE))
    assert upsample(phi, 2).shape == (2 * n, 2 * n)
    print("doe.py: all self-checks passed")
