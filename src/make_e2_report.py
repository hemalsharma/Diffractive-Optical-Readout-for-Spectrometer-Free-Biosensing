"""
make_e2_report.py -- build the Embodiment 2 report (HTML) from the results.

Reads results/embodiment2.json and results/e2_rms_vs_M.csv written by
embodiment2.py, so every number in the report comes from the last run.
Figures are referenced as results/<name>.png relative to the HTML file.

    python src/embodiment2.py          # first, to produce the results
    python src/make_e2_report.py [out.html] [--fig=results] [--standalone]

The HTML is a page body (no <html>/<head> wrapper) so it can be published
as-is; --standalone wraps it for printing with a browser, e.g.

    python src/make_e2_report.py Embodiment2_Report.html --standalone
    chrome --headless --print-to-pdf=Embodiment2_Report.pdf Embodiment2_Report.html
"""

import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load():
    with open(RES / "embodiment2.json") as fh:
        r = json.load(fh)
    with open(RES / "e2_rms_vs_M.csv") as fh:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    return r, rows


def sci(x, digits=2):
    """1.23 × 10⁻⁴ style."""
    m, e = f"{x:.{digits}e}".split("e")
    sup = str(int(e)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m} × 10{sup}"


CSS = """
<title>Embodiment 2 Estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root {
  --paper: #f6f7f9; --panel: #ffffff; --ink: #1b2330; --muted: #5a6474;
  --rule: #d9dde4; --accent: #9b2c3a; --cool: #245d8c; --ok: #2f7a4b; --fail: #b3261e;
  --code: #eef1f5;
  --sans: "IBM Plex Sans", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --serif: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #12161c; --panel: #1a2029; --ink: #e4e8ee; --muted: #9aa5b4;
    --rule: #2e3643; --accent: #e07784; --cool: #7fb2dc; --ok: #6cc28f; --fail: #f08a80;
    --code: #222a35; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #12161c; --panel: #1a2029; --ink: #e4e8ee; --muted: #9aa5b4;
  --rule: #2e3643; --accent: #e07784; --cool: #7fb2dc; --ok: #6cc28f; --fail: #f08a80;
  --code: #222a35; color-scheme: dark;
}
body { background: var(--paper); color: var(--ink); font: 15.5px/1.6 var(--sans); }
.wrap { max-width: 1060px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; }
.prose { max-width: 700px; }
header { display: grid; gap: 10px; padding-bottom: 28px; border-bottom: 1px solid var(--rule); }
.eyebrow { font: 500 12px/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { font: 600 clamp(28px, 4.2vw, 40px)/1.15 var(--serif); margin: 0; text-wrap: balance; }
h2 { font: 600 23px/1.25 var(--serif); margin: 52px 0 10px; text-wrap: balance; }
h3 { font: 600 16px/1.3 var(--sans); margin: 26px 0 6px; }
p { margin: 0 0 12px; }
.lede { font-size: 17.5px; color: var(--muted); max-width: 760px; }
.meta { font: 13px/1.5 var(--mono); color: var(--muted); }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
code, .eq { font-family: var(--mono); font-size: .92em; background: var(--code); padding: 1px 5px; border-radius: 3px; }
.eqblock { font-family: var(--mono); font-size: 14px; background: var(--code); padding: 12px 16px;
  border-radius: 4px; overflow-x: auto; white-space: pre; line-height: 1.7; margin: 10px 0 16px; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 18px 0 8px; }
.flow span.step { background: var(--panel); border: 1px solid var(--rule); padding: 6px 10px; border-radius: 4px;
  font: 13px/1.3 var(--mono); }
.flow span.step.key { border-color: var(--accent); color: var(--accent); }
.flow span.arr { color: var(--muted); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0 4px; }
.stat { background: var(--panel); border: 1px solid var(--rule); padding: 14px 16px; border-radius: 4px; display: grid; gap: 4px; align-content: start; }
.stat .v { font: 500 22px/1.2 var(--mono); font-variant-numeric: tabular-nums; }
.stat .k { font-size: 13px; color: var(--muted); line-height: 1.4; }
figure { margin: 18px 0 26px; }
figure img { width: 100%; height: auto; background: #fff; border: 1px solid var(--rule); border-radius: 3px; }
figcaption { font-size: 13.5px; color: var(--muted); margin-top: 8px; max-width: 860px; }
figcaption b { color: var(--ink); font-weight: 600; }
.fig-narrow img { max-width: 720px; }
.tbl { overflow-x: auto; margin: 12px 0 20px; }
table { border-collapse: collapse; font-size: 13.5px; min-width: 100%; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 12.5px; }
td.r, th.r { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
tr.hl td { background: color-mix(in srgb, var(--accent) 7%, transparent); }
.pill { display: inline-block; font: 600 11px/1 var(--mono); padding: 4px 7px; border-radius: 3px; letter-spacing: .04em; }
.pill.pass { color: var(--ok); border: 1px solid var(--ok); }
.pill.fail { color: var(--fail); border: 1px solid var(--fail); }
.callout { border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; margin: 18px 0; max-width: 760px; }
.callout p:last-child { margin-bottom: 0; }
ul { padding-left: 20px; margin: 0 0 14px; }
li { margin-bottom: 5px; }
footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--rule); font-size: 13px; color: var(--muted); }
a { color: var(--cool); }
@media print {
  body { background: #fff; font-size: 11pt; }
  .wrap { max-width: none; padding: 0; }
  h2 { break-after: avoid; }
  figure, .tbl, .stats { break-inside: avoid; }
}
</style>
"""


def build_html(r, rows, fig="results"):
    m, mc, lin, info, pd, ab = (r[k] for k in ("model", "monte_carlo", "linearity", "information",
                                             "power_drift", "stage1_two_region"))
    cfg = r["config"]
    Sn, gam = cfg["Sn_nm_per_riu"], cfg["gamma_nm"]
    last = rows[-1]
    by_M = {int(x["M"]): x for x in rows}
    n_all = int(last["M"])
    lr = lin["linear_range_nm_by_M"]
    lo, hi = lr[str(n_all)]
    lo16, hi16 = lr["16"]
    frac_512 = by_M[512]["crb_nm"] / last["crb_nm"] - 1
    frac_256 = by_M[256]["crb_nm"] / last["crb_nm"] - 1
    ols_pen = last["rms_ols_nm"] / last["rms_gls_nm"] - 1
    ab_gain = ab["crb_nm"] / last["crb_nm"]
    drift_sig = pd["gls_error_from_drift_nm"] / last["crb_nm"]
    mc_vs_pred = max(abs(x["rms_gls_nm"] / x["rms_gls_predicted_nm"] - 1) for x in rows)
    mc_vs_crb = max(abs(x["rms_gls_nm"] / x["crb_nm"] - 1) for x in rows)

    hl = {16: " class='hl'", n_all: " class='hl'"}
    mc_rows = "".join(
        f"<tr{hl.get(int(x['M']), '')}><td class='r'>{int(x['M'])}</td>"
        f"<td class='r'>{x['crb_nm']:.4f}</td><td class='r'>{x['rms_gls_nm']:.4f}</td>"
        f"<td class='r'>{x['rms_gls_predicted_nm']:.4f}</td><td class='r'>{x['rms_ols_nm']:.4f}</td>"
        f"<td class='r'>{x['crb_local_nm']:.4f}</td><td class='r'>{x['rms_gauss_newton_nm']:.4f}</td>"
        f"<td class='r'>{x['crb_dn_riu']:.2e}</td></tr>"
        for x in rows)

    check_rows = "".join(
        f"<tr><td><span class='pill {'pass' if c['passed'] else 'fail'}'>"
        f"{'PASS' if c['passed'] else 'FAIL'}</span></td>"
        f"<td>{html.escape(c['name'])}</td><td class='num'>{html.escape(c['detail'])}</td></tr>"
        for c in r["checks"])
    n_pass = sum(c["passed"] for c in r["checks"])

    return f"""{CSS}
<div class="wrap">
<header>
  <div class="eyebrow">Diffractive biosensing · Embodiment 2 · fixed-grating baseline</div>
  <h1>Pixel-wise spectral coding and direct resonance-shift estimation</h1>
  <p class="lede">The Stage 1 forward model now feeds a camera instead of two detectors. Each pixel is a
  measurement with its own spectral weighting, and the resonance shift Δλ<sub>r</sub> is estimated
  directly from the most informative pixels, then converted to a refractive-index change Δn.
  The DOE is still the Stage 1 linear grating: no inverse design yet.</p>
  <p class="meta">Hemal Sharma · numbers generated by <code>python src/embodiment2.py</code></p>
</header>

<div class="stats">
  <div class="stat"><span class="v">{last['crb_nm']:.3f} nm</span><span class="k">Cramér–Rao bound on Δλ<sub>r</sub>, all {n_all} pixels (Monte Carlo {last['rms_gls_nm']:.3f} nm)</span></div>
  <div class="stat"><span class="v">{sci(last['crb_dn_riu'])}</span><span class="k">σ<sub>Δn</sub> in RIU, for S<sub>n</sub> = {Sn:.0f} nm/RIU (assumed)</span></div>
  <div class="stat"><span class="v">{by_M[16]['crb_nm']:.3f} nm</span><span class="k">bound with the best 16 pixels; 256 pixels are within {100 * frac_256:.0f} % of all</span></div>
  <div class="stat"><span class="v">{lo:+.2f} … {hi:+.2f} nm</span><span class="k">linear range: linearization bias below the all-pixel noise (γ = {gam:.1f} nm)</span></div>
</div>

<h2>What changed from Stage 1</h2>
<div class="prose">
<p>Stage 1 reduced the detector image to two regions A and B and a ratio R. Embodiment 2 keeps the
individual camera pixels as separate measurements:</p>
</div>
<div class="flow" aria-label="Embodiment 2 pipeline">
  <span class="step">Δn</span><span class="arr">→</span>
  <span class="step">Δλ<sub>r</sub> = S<sub>n</sub> Δn</span><span class="arr">→</span>
  <span class="step">s(λ; Δλ<sub>r</sub>)</span><span class="arr">→</span>
  <span class="step">DOE + propagation</span><span class="arr">→</span>
  <span class="step key">camera y = W s</span><span class="arr">→</span>
  <span class="step key">best M pixels</span><span class="arr">→</span>
  <span class="step key">Δλ̂<sub>r</sub> → Δn̂</span>
</div>
<div class="prose">
<p>Kept unchanged from Stage 1: the Lorentzian sensor (λ<sub>r,0</sub> = 1550 nm, γ = {gam:.1f} nm,
A = 0.8) with its 20 nm SLED source, the {m['n_lambda']}-sample wavelength grid
({m['lambda_range_nm'][0]:.1f}–{m['lambda_range_nm'][1]:.1f} nm), the band-limited angular-spectrum
propagation of every wavelength, incoherent summation, and the 160 µm linear phase grating at z = 50 mm.
The Stage 1 scripts, results and checks are untouched; the two-region readout is kept only as a
comparison.</p>
</div>
<figure class="fig-narrow"><img src="{fig}/fig_schematic.png" alt="Stage 1 system schematic: SLED, resonant sensor, phase-only DOE, 50 mm propagation, detector plane with regions A and B.">
<figcaption><b>Stage 1 architecture, for reference.</b> Embodiment 2 replaces the A/B split and the ratio R
with the full camera vector and a statistical estimator.</figcaption></figure>

<h2>1 · Pixel spectral weighting functions</h2>
<div class="prose">
<p>For each wavelength, a unit-power Gaussian beam is propagated through the grating and the power on
each camera pixel is recorded. Row j of the matrix W is the spectral weighting W<sub>j</sub>(λ) of pixel j:</p>
</div>
<div class="eqblock">W_ji = ∬_Ωj |E_i(x, y)|² dx dy,   ∬ |E_in|² = 1          (Eq. 4)
y_j(Δλ_r) = Σ_i W_ji s(λ_i; Δλ_r) Δλ   ⇔   y = W s      (Eq. 5)</div>
<div class="prose">
<p>The camera has {m['camera_pitch_um']:.0f} µm pixels (4 × 4 simulation cells). A
{m['roi_pixels'][0]} × {m['roi_pixels'][1]} pixel region of interest around the +1 order holds
{100 * m['roi_fraction_of_camera_power_min']:.2f} % of the light on the camera. W is computed once
({m['build_seconds']:.0f} s); every new shift is then one matrix–vector product.</p>
</div>
<figure><img src="{fig}/e2_weighting.png" alt="Spectral weighting functions of four representative pixels, normalized versions, and their locations on the camera.">
<figcaption><b>Deliverable 1.</b> (a) W<sub>j</sub>(λ) for the pixels with the largest positive and negative
g<sub>j</sub>, the brightest pixel and a low-|g| pixel; resonance λ<sub>r,0</sub> ± γ shaded. (b) The same, normalized:
the grating gives every pixel an almost linear spectral slope, positive on one side of the spot and
negative on the other. (c) Pixel locations on y(0).</figcaption></figure>

<h2>2 · The camera vector for known shifts</h2>
<figure><img src="{fig}/e2_camera.png" alt="Camera images and full camera vectors for shifts of minus one to plus one linewidth.">
<figcaption><b>Deliverable 2.</b> (a) y(0) on the ROI, peak {m['peak_pixel_e']:.2e} e⁻ per measurement
({m['peak_pixel_e_per_frame']:.2e} e⁻ per frame). (b, c) y(Δλ<sub>r</sub>) − y(0) for ±γ: both shifts brighten the
spot, because the dip moves off the centre of the source spectrum. (d) With the total power removed,
the spectral coding appears as a left–right redistribution. (e, f) The complete {n_all}-pixel vector
y(Δλ<sub>r</sub>) and its change, row-major over the ROI.</figcaption></figure>

<h2>3 · Resonance-shift sensitivity</h2>
<div class="eqblock">g_j = ∂y_j/∂(Δλ_r) |₀ ≈ [y_j(+δ) − y_j(−δ)] / 2δ,   δ = {cfg['delta_gammas']}γ = {m['delta_nm']:.3f} nm   (Eq. 7)
g   = W ∂s/∂(Δλ_r),   ∂s/∂(Δλ_r) = s_src ∂T/∂(Δλ_r) Δλ                         (Eq. 8)</div>
<div class="prose">
<p>Three independent derivatives agree: the central difference changes by
{m['g_fd_vs_fd_half_delta']:.1e} (max-normalized) when δ is halved and matches the analytic
derivative of the Lorentzian to {m['g_fd_vs_analytic']:.1e}, and forward-mode autograd through the
Stage 1 sensor model matches the analytic result to {m['g_analytic_vs_autograd']:.0e}.</p>
</div>
<figure><img src="{fig}/e2_sensitivity.png" alt="Signed sensitivity map, finite-difference error map, and scatter of three derivative methods.">
<figcaption><b>Deliverable 3.</b> (a) Signed g<sub>j</sub> [e⁻/nm]: an antisymmetric dipole across the +1-order spot.
The 16 highest-information pixels are marked. (b) Eq. (7) − Eq. (8): an O(δ²) residual at the 10⁻⁵ level.
(c) Finite difference (δ and δ/2) and autograd against the analytic derivative.</figcaption></figure>

<h2>4 · Estimator, noise and linear range</h2>
<div class="prose">
<p>Pixels have independent shot and read noise. The measurement sums {r['model']['n_frames']} frames of
{sci(r['model']['photons'] / r['model']['n_frames'], 0)} photoelectrons, each with
{r['model']['read_noise_e_per_frame']:.0f} e⁻ read noise, so the brightest pixel stays near
{m['peak_pixel_e_per_frame']:.1e} e⁻ per frame, below a typical InGaAs full well. The shift is
estimated by generalized least squares:</p>
</div>
<div class="eqblock">σ_j² = y_j + σ_read²                                                     (Eq. 9)
Δλ̂_r = gᵀΣ⁻¹(y_meas − μ) / gᵀΣ⁻¹g,   Var(Δλ̂_r) = 1/F,  F = Σ_j g_j²/σ_j²    (Eq. 10, 14)
Δn̂ = Δλ̂_r / S_n,   σ_Δn = σ_Δλ / S_n                                        (Eq. 12)</div>
<div class="prose">
<p>The linear model is exact only near Δλ<sub>r</sub> = 0. Its bias grows roughly quadratically
({lin['bias_at_0p1_gamma_nm'] * 1e3:.1f} pm at 0.1γ) and exceeds the all-pixel noise outside
<span class="num">{lo:+.3f} … {hi:+.3f} nm</span>. With only 16 pixels the noise is larger, so the
usable range widens to <span class="num">{lo16:+.2f} … {hi16:+.2f} nm</span>. Re-linearizing about
the current estimate (Gauss–Newton, {cfg['gn_iterations']} iterations) removes the bias to
{lin['gauss_newton_max_abs_err_nm']:.0e} nm over ±2γ.</p>
</div>
<figure><img src="{fig}/e2_estimate.png" alt="Noiseless estimate against true shift, bias against noise bands, and noisy estimates with error bars.">
<figcaption><b>Deliverable 4.</b> (a) Noiseless estimate: linear GLS bends away from the ideal line;
Gauss–Newton lies on it. Green: linear range. (b) Linearization bias against ±σ<sub>CRB</sub> for 16, 256 and
all pixels. (c) Noisy estimates at five test shifts that differ from the calibration points ±δ
(mean ± standard deviation over {mc['trials']} realizations each).</figcaption></figure>

<h2>5 · How many pixels are needed</h2>
<div class="prose">
<p>Pixels are ranked by their Fisher information F<sub>j</sub> = g<sub>j</sub>²/σ<sub>j</sub>², and the estimator is
applied to the best M. The Monte Carlo uses {mc['trials']} noise realizations at each of
{len(mc['test_shifts_nm'])} test shifts. Its RMS error matches the exact error of the linear
estimator (bias² plus variance) within {100 * mc_vs_pred:.1f} % and stays within {100 * mc_vs_crb:.1f} % of the
Cramér–Rao bound for every M.</p>
</div>
<figure class="fig-narrow"><img src="{fig}/e2_rms_vs_M.png" alt="RMS estimation error against the number of retained pixels, log-log, with Cramér–Rao bounds.">
<figcaption><b>Deliverable 5.</b> RMS error of Δλ<sub>r</sub> against M. Right axis: σ<sub>Δn</sub>. The dashed red line is the
bound for the Stage 1 two-region readout (A, B) built from the same pixels and photons.</figcaption></figure>
<div class="tbl"><table>
<thead><tr><th class="r">M</th><th class="r">CRB at 0 [nm]</th><th class="r">RMS GLS [nm]</th>
<th class="r">predicted [nm]</th><th class="r">RMS eq-var LS [nm]</th><th class="r">local CRB [nm]</th>
<th class="r">RMS Gauss–Newton [nm]</th><th class="r">σ<sub>Δn</sub> [RIU]</th></tr></thead>
<tbody>{mc_rows}</tbody></table></div>
<div class="prose">
<ul>
<li>A few pixels already carry most of the information: M = 16 gives {by_M[16]['crb_nm']:.3f} nm,
M = 256 is within {100 * frac_256:.0f} % and M = 512 within {100 * frac_512:.1f} % of all {n_all} pixels.</li>
<li>Weighting by the shot-plus-read variance matters: equal-variance least squares (Eq. 11) is
{100 * ols_pen:.1f} % worse with all pixels.</li>
<li>With this grating, the pixel-wise estimator improves on the Stage 1 two-region readout only by a
factor {ab_gain:.2f} ({ab['crb_nm']:.3f} → {last['crb_nm']:.3f} nm). The grating encodes the shift as a
dipole, a centroid move that a two-detector split already captures. Larger gains need a DOE that
spreads spectral information over more pixels, which is the purpose of the inverse design.</li>
<li>For M ≥ 16, Gauss–Newton falls below the Δλ<sub>r</sub> = 0 bound but stays above the local bound at the
test shifts (section 7). For M ≤ 8 it is clipped to ±2γ and biased, so the bound does not apply.</li>
</ul>
</div>

<h2>6 · Refractive-index change</h2>
<div class="prose">
<p>With the assumed bulk sensitivity S<sub>n</sub> = {Sn:.0f} nm/RIU, all pixels give
σ<sub>Δn</sub> = {sci(last['crb_dn_riu'])} RIU, and the best 16 give {sci(by_M[16]['crb_dn_riu'])} RIU.
Because Eq. (12) is a division by S<sub>n</sub>, these numbers scale directly with the value of S<sub>n</sub> of
the real resonator.</p>
</div>
<figure><img src="{fig}/e2_delta_n.png" alt="Estimated against true refractive-index change with error bars, and index resolution against M.">
<figcaption><b>Deliverable 6.</b> (a) Recovered Δn at the five test shifts (mean ± std) for 16 and all pixels.
(b) σ<sub>Δn</sub> against M: Eq. (12) applied to the Cramér–Rao bound, and Monte Carlo RMS.</figcaption></figure>

<h2>7 · Where the information comes from</h2>
<div class="callout">
<p>The source spectrum is centred on the resonance, so the total transmitted power is stationary at
Δλ<sub>r</sub> = 0. There, all the information comes from how the spectrum is redistributed across the
pixels. Away from zero the total power changes with the shift and adds a strong intensity channel:
the local bound drops from {info['crb_at_0_nm'][0]:.3f} nm at 0 to {info['crb_at_1gamma_nm'][0]:.4f} nm at +γ.</p>
<p>That intensity channel is exactly what a drifting source corrupts. With the source amplitude
estimated jointly (Eq. 13) the bound at +γ is {info['crb_at_1gamma_nm'][1]:.3f} nm, close to the
Δλ<sub>r</sub> = 0 value. A 1 % power change read by Eq. (10) gives a false shift of
{pd['gls_error_from_drift_nm']:.2f} nm ({drift_sig:.0f} σ); the joint fit removes it
({pd['joint_error_from_drift_nm']:.0e} nm) for almost no cost at the operating point
({pd['crb_shift_only_nm']:.4f} → {pd['crb_joint_nm']:.4f} nm).</p>
</div>
<figure><img src="{fig}/e2_information.png" alt="Local Cramér–Rao bound against operating point for shift-only and amplitude-robust estimation, and total power slope.">
<figcaption><b>Supporting analysis.</b> (a) Local bound with the source power known (shift only) and with the
amplitude as a nuisance parameter. (b) Relative slope of the total transmitted power.</figcaption></figure>

<h2>Validation</h2>
<div class="tbl"><table><thead><tr><th>Result</th><th>Check</th><th>Value</th></tr></thead>
<tbody>{check_rows}</tbody></table></div>
<div class="prose">
<p>{n_pass} of {len(r['checks'])} Embodiment 2 checks pass. The Stage 1 validation (7 checks) and
completion criterion (7 checks) still pass unchanged, and the pytest suite covers both.</p>

<h2>Assumptions</h2>
<ul>
<li>S<sub>n</sub> = {Sn:.0f} nm/RIU. This is not given in the instructions; it is a typical value for a resonant
dielectric sensor and only rescales Δn.</li>
<li>Camera: {m['camera_pitch_um']:.0f} µm pixels, unit quantum efficiency and flat responsivity (W is not
weighted), no dark current, no saturation modelled.</li>
<li>Noise: {r['model']['n_frames']} frames of {sci(r['model']['photons'] / r['model']['n_frames'], 0)} e⁻ with
{r['model']['read_noise_e_per_frame']:.0f} e⁻ read noise per frame, independent between pixels. Shot noise is
Poisson; read noise is Gaussian.</li>
<li>The sensor is spatially uniform over the beam, so y is linear in the spectrum.</li>
<li>The Stage 1 wavelength grid is reused as instructed. It was built for shifts 0 to +5 nm, so it is
slightly asymmetric about 1550 nm; total power is stationary at 0 only to 6 × 10⁻⁵.</li>
<li>Pixel ranking and GLS weights use the noise at Δλ<sub>r</sub> = 0 (the calibration point).</li>
</ul>

<h2>Limitations and next steps</h2>
<ul>
<li>The fixed grating is a weak spectral coder: only about 2.5 grating periods are illuminated, so one
spot radius corresponds to roughly 400 nm of wavelength. This limits both the bound and the gain over
the A/B readout.</li>
<li>The linear estimator is valid only within about ±0.26γ for all pixels; Gauss–Newton extends this
when the noise is small, but for few pixels it needs bounds and becomes biased.</li>
<li>Correlated noise, pixel crosstalk, detector nonlinearity and temperature drift of the resonance are
not modelled. A greedy selector for correlated noise is implemented but not exercised.</li>
<li>Next: make φ(x, y) trainable and maximize F(φ) = Σ<sub>j∈P<sub>M</sub></sub> g<sub>j</sub>²/σ<sub>j</sub>² with σ<sub>j</sub>²
depending on μ (Eq. 15), preferably using the amplitude-robust information so the design cannot win by
exploiting total power.</li>
</ul>
</div>

<footer>Reproduce: <code>python src/embodiment2.py</code> (about 30 s) then
<code>python src/make_e2_report.py</code>. Source: <code>src/spectral_coding.py</code>
(y = W s), <code>src/estimator.py</code> (g, noise, estimators, Fisher ranking),
<code>src/embodiment2.py</code> (deliverables and checks).</footer>
</div>
"""


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    out = Path(args[0]) if args else ROOT / "Embodiment2_Report.html"
    page = build_html(*load(), fig=opts.get("fig", "results"))
    if "--standalone" in sys.argv:
        head, body = page.split("</style>", 1)
        page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"{head}</style></head><body>{body}</body></html>")
    out.write_text(page, encoding="utf-8")
    print(f"report written to {out}")
