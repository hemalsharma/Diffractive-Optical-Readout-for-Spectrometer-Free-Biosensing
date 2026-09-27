# Inverse-Designed Diffractive Optical Biosensing
## Stage 1: Forward Model Development and Validation

**Author:** Hemal Sharma

## Project overview

This package implements and validates a physics-based forward model for a
spectrometer-free diffractive optical biosensor. It computes what two
photodetectors measure for a given analyte concentration:

```
Concentration c
      |
      v
Sensor spectrum S(lambda; c)          Lorentzian resonance shifted by c        sensor_model.py
      |
      v
Phase-only DOE  t = exp[i phi(x,y)]   linear phase grating (validation DOE)    doe.py
      |
      v
Optical propagation                   angular spectrum (+ Fraunhofer check)    propagation.py
      |
      v
Detector intensity I(x,y; c)          incoherent sum over wavelengths          detector.py
      |
      v
Detector signals I_A, I_B, R = (I_A - I_B) / (I_A + I_B)
```

`main.py` only connects these modules. All parameters (wavelength, linewidth,
DOE size, pixel pitch, beam radius, distance, detector size, ...) are in
**`src/config.py`**. No inverse design or optimization is performed in Stage 1.

For an externally supplied sampled spectrum and DOE phase, call
`main.forward_from_spectrum(lam_m, S, phi_doe, dlam_nm=...)`. It returns the
detector-plane image `I`, detector powers `IA` and `IB`, total grid power
`P_total`, and normalized differential signal `R`. If `dlam_nm` is omitted,
the function derives trapezoidal integration weights from `lam_m`. Note that
`main.forward(c)` uses the rectangle rule instead (uniform step
`cfg.dlam_nm(sensor)` at every sample, as in `results/`); the two differ only in
the end-point weights, and passing `dlam_nm=cfg.dlam_nm(sensor)` reproduces
`forward` exactly. If no light reaches either detector, `R` is returned as 0.

The full description, results and validation are in **`Stage1_Report.pdf`**.

## Contents

```
./
|-- README.md                  this file
|-- requirements.txt
|-- requirements-dev.txt       pytest (for the test suite)
|-- Stage1_Report.pdf          Stage 1 report
|-- src/                       source code (run from the package folder, see below)
|-- notebooks/
|   `-- Stage1_Forward_Model_Demo.ipynb    step-by-step demonstration, outputs included
|-- tests/                     pytest wrappers around the checks below
`-- results/                   expected outputs, already generated
```

## Requirements

- Python 3.10 or newer
- numpy, torch (CPU is enough), matplotlib; jupyter for the notebook
- About 2 GB of free memory (for the largest validation check)

Tested with Python 3.14.0, torch 2.14.0 (CPU), numpy 2.4.4, matplotlib 3.10.8 on Windows 11.

## Installation

```
pip install -r requirements.txt
```

## Running the code

Run all commands from the repository root (the folder containing `README.md`).

### Step 1: Validate the optical model  (about 35 s)

```
python src/validation.py
```

Expected ending:

```
All 7 validation tests passed. Figures and validation.json saved to .../results
```

### Step 2: Run the complete forward model  (about 6 s)

```
python src/main.py
```

Prints the detector signals for nine concentrations and writes the report
figures and `results/final_detector_table.csv`. Expected ending:

```
Forward model complete: 8 figures and final_detector_table.csv saved to .../results
```

### Step 3: Stage 1 completion criterion  (about 25 s)

```
python src/completion_check.py
```

Checks reproducibility, arbitrary phase masks, arbitrary spectra and gradients.
Expected ending:

```
Stage 1 completion criterion met: all 7 checks passed. ...
```

### Automated tests

```
pip install -r requirements-dev.txt
pytest -m "not slow"     # module self-checks and quick checks, about 1 min
pytest                   # also runs the full validation and completion checks
```

The tests call the same functions as the scripts but do not write to `results/`.
The full suite runs in CI (GitHub Actions, CPU only) on every push and pull request.

### Step 4: Interactive demonstration

```
jupyter notebook notebooks/Stage1_Forward_Model_Demo.ipynb
```

The notebook walks through the calculation one step at a time. It is saved
with its outputs, so it can be read without running it.

The system schematic (report Fig. 1) is drawn by `python src/make_schematic.py`.

The calculation is deterministic: re-running the scripts reproduces the files in
`results/` (numbers identical; image files may differ only in metadata).

## Results

| File in `results/` | Report | Content |
|---|---|---|
| `fig_schematic.png` | Fig. 1 | System schematic (`python src/make_schematic.py`) |
| `fig_sensor.png` | Fig. 2 | Sensor spectra S(lambda; c) |
| `fig_doe.png` | Fig. 3 | DOE phase map phi(x, y) |
| `fig_input.png` | Fig. 4 | Gaussian input beam and phase after the DOE |
| `fig_broadband.png` | Fig. 5 | Broadband detector image I(x, y; c) with detectors A, B |
| `fig_readout.png` | Fig. 6 | Detector response: Delta R(c) and spot shift |
| `final_detector_table.csv` | Table 2 | Detector signals for c = 0 ... 2 |
| `validation.json`, `completion.json` | Table 3 | All validation numbers |

Supporting plots that are not in the report but are produced by the scripts:
`fig_concept.png` (block diagram), `fig_propagation.png` (propagation with and without
the grating), `fig_difference.png` (normalized difference image), and the validation
plots `val_free.png`, `val_grating.png`, `val_convergence.png`, `val_fraunhofer.png`.

## Main results

The forward model reproduces:

1. Gaussian beam propagation (error 4e-6)
2. Linear grating diffraction and the grating equation (1.8 nm error in a 480 um spot position)
3. Fraunhofer far-field limit of the angular spectrum
4. Energy conservation (1e-15)
5. Convergence with wavelength, spatial and detector sampling

and meets the Stage 1 completion criterion: reproducible results for arbitrary
spectra S(lambda; c) and phase masks phi(x, y), with correct gradients for the
inverse design of Stage 2.

With the conventional grating, a resonance shift of one linewidth changes the
differential readout R by only 1.2e-3. This is the baseline that the
inverse-designed DOE has to improve on in Stage 2.
