"""
estimator.py -- resonance-shift sensitivity, noise model and estimators.

Embodiment 2, Sections 4-5. Works on the linear model y(d) = W s(d) of
spectral_coding.py, but only needs vectors, so it applies to any fixed DOE.

    g_j = dy_j/dd at d = 0                          (Eq. 6)
        ~ [y_j(+delta) - y_j(-delta)] / (2 delta)   (Eq. 7)
        = W ds/dd                                   (Eq. 8)
    y_meas = y(d) + eps,  var_j = y_j + sigma_read^2  (Eq. 9)
    d_hat = g^T S^-1 (y_meas - mu) / g^T S^-1 g,  Var = 1/F     (Eq. 10)
    d_hat = g^T (y_meas - mu) / g^T g,  sigma = sigma / ||g||   (Eq. 11)
    dn_hat = d_hat / S_n                            (Eq. 12)
    y ~ a mu + (a d) g = H theta  (joint amplitude) (Eq. 13)
    F_j = g_j^2 / var_j,  F = sum F_j,  sigma(M) >= (sum_{P_M} F_j)^-1/2  (Eq. 14)

All estimators take an optional index set `idx` (the retained pixels P_M)
and broadcast over leading dimensions of y_meas (e.g. Monte Carlo trials).
"""

import torch

DTYPE = torch.float64


# ----------------------------------------------------------------------------
# Sensitivity
# ----------------------------------------------------------------------------

def sensitivity_fd(response, delta):
    """Central difference g = [y(+delta) - y(-delta)] / (2 delta)  (Eq. 7).

    response: callable d -> y(d).
    """
    return (response(delta) - response(-delta)) / (2 * delta)


def sensitivity_exact(W, ds):
    """g = W ds/dd (Eq. 8), with ds from the analytic or autograd derivative."""
    return ds @ W.T


# ----------------------------------------------------------------------------
# Noise (Eq. 9)
# ----------------------------------------------------------------------------

def noise_variance(y, read_var):
    """Per-pixel variance sigma_j^2 = y_j + sigma_read^2 (y in photoelectrons)."""
    return y + read_var


def add_noise(y, read_var, n=None, generator=None):
    """Independent shot (Poisson) and read (Gaussian) noise.

    y: noiseless camera vector(s) [e-]. n: number of realizations, prepended
    as a leading dimension. Returns y_meas with mean y and variance y + read_var.
    """
    y = torch.as_tensor(y, dtype=DTYPE).clamp(min=0)
    if n is not None:
        y = y.expand(n, *y.shape)
    shot = torch.poisson(y, generator=generator)
    read = torch.randn(y.shape, dtype=DTYPE, generator=generator) * read_var ** 0.5
    return shot + read


# ----------------------------------------------------------------------------
# Fisher information and pixel selection (Eq. 14)
# ----------------------------------------------------------------------------

def fisher_per_pixel(g, var):
    """F_j = g_j^2 / sigma_j^2."""
    return g ** 2 / var


def rank_pixels(g, var):
    """Pixel indices sorted by decreasing F_j, and the sorted F_j."""
    F = fisher_per_pixel(g, var)
    order = torch.argsort(F, descending=True)
    return order, F[order]


def crb(g, var, idx=None):
    """Cramer-Rao bound sigma = (sum_{j in idx} F_j)^-1/2  (Eq. 10, 14)."""
    F = fisher_per_pixel(g, var)
    if idx is not None:
        F = F[idx]
    return float(F.sum()) ** -0.5


def select_greedy(g, cov, M):
    """Greedy selection maximizing g^T S^-1 g for a full covariance S.

    For correlated noise ranking by F_j is not optimal (Section 5). Exact but
    O(M N) solves; intended for small problems.
    """
    chosen, rest = [], list(range(g.numel()))
    for _ in range(M):
        best, best_F = None, -1.0
        for j in rest:
            s = chosen + [j]
            F = float(g[s] @ torch.linalg.solve(cov[s][:, s], g[s]))
            if F > best_F:
                best, best_F = j, F
        chosen.append(best)
        rest.remove(best)
    return torch.tensor(chosen)


# ----------------------------------------------------------------------------
# Linear estimators
# ----------------------------------------------------------------------------

def _sub(v, idx):
    return v if idx is None else v[..., idx]


def gls_estimate(y_meas, mu, g, var, idx=None):
    """Generalized least squares with diagonal S (Eq. 10). Returns (d_hat, sigma)."""
    y, mu, g, var = _sub(y_meas, idx), _sub(mu, idx), _sub(g, idx), _sub(var, idx)
    w = g / var
    F = float(w @ g)
    return ((y - mu) @ w) / F, F ** -0.5


def ols_estimate(y_meas, mu, g, sigma=1.0, idx=None):
    """Equal-variance least squares, S = sigma^2 I (Eq. 11). Returns (d_hat, sigma_d)."""
    y, mu, g = _sub(y_meas, idx), _sub(mu, idx), _sub(g, idx)
    gg = float(g @ g)
    return ((y - mu) @ g) / gg, sigma / gg ** 0.5


def joint_amplitude_estimate(y_meas, mu, g, var, idx=None):
    """Joint source amplitude a and shift (Eq. 13).

    y ~ a mu + (a d) g = H theta, H = [mu g]; theta = (H^T S^-1 H)^-1 H^T S^-1 y.
    Returns (d_hat, a_hat).
    """
    y, mu, g, var = _sub(y_meas, idx), _sub(mu, idx), _sub(g, idx), _sub(var, idx)
    H = torch.stack([mu, g], 1)                                # (M, 2)
    Hw = H / var[:, None]
    theta = torch.linalg.solve(H.T @ Hw, (y @ Hw).unsqueeze(-1)).squeeze(-1)
    return theta[..., 1] / theta[..., 0], theta[..., 0]


def gauss_newton(y_meas, model, jacobian, var, idx=None, d0=None, iterations=10,
                 max_step=None, bounds=None):
    """Re-linearize about the current estimate and iterate (Section 4, limit i).

    model(d) -> y(d) and jacobian(d) -> dy/dd for d of shape (K,) return
    (K, Npix). Weighted by the fixed diagonal variance var. Returns d_hat (K,).

    max_step limits each update and bounds = (lo, hi) clips the estimate to
    the calibrated range. Both only matter when the noise is comparable to the
    linewidth (few pixels), where the undamped iteration can run away to
    shifts at which the model carries no information.
    """
    y = _sub(torch.as_tensor(y_meas, dtype=DTYPE), idx)
    lead = y.shape[:-1]
    y = y.reshape(-1, y.shape[-1])
    var = _sub(var, idx)
    d = torch.zeros(y.shape[0], dtype=DTYPE) if d0 is None else \
        torch.as_tensor(d0, dtype=DTYPE).reshape(-1).expand(y.shape[0]).clone()
    for _ in range(iterations):
        r = y - _sub(model(d), idx)
        J = _sub(jacobian(d), idx)
        step = ((r * J) / var).sum(-1) / ((J * J) / var).sum(-1)
        if max_step is not None:
            step = step.clamp(-max_step, max_step)
        d = d + step
        if bounds is not None:
            d = d.clamp(*bounds)
    return d.reshape(lead)


def local_fisher(g, var, mu=None, idx=None):
    """Fisher information on the shift at one operating point.

    Without mu: shift only, F = g^T S^-1 g. With mu: the source amplitude is
    a nuisance parameter (Eq. 13) and the information left for the shift is
    the Schur complement F_dd - F_da^2 / F_aa of the 2 x 2 Fisher matrix.
    """
    g, var = _sub(g, idx), _sub(var, idx)
    F = float((g * g / var).sum())
    if mu is None:
        return F
    mu = _sub(mu, idx)
    Fa, Fad = float((mu * mu / var).sum()), float((mu * g / var).sum())
    return F - Fad ** 2 / Fa


def to_refractive_index(d_hat, sigma_d, Sn):
    """dn_hat = d_hat / S_n, sigma_dn = sigma_d / S_n  (Eq. 12)."""
    return d_hat / Sn, sigma_d / Sn
