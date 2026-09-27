"""Full Stage 1 validation (python src/validation.py) and completion criterion
(python src/completion_check.py), with every pass/fail threshold as a test."""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def validation_results(tmp_path_factory):
    import main
    import validation

    out = tmp_path_factory.mktemp("results")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main, "OUT", out)
        mp.setattr(validation, "OUT", out)
        validation.RESULTS.clear()
        for t in (validation.test_free_propagation, validation.test_grating_equation,
                  validation.test_fraunhofer, validation.test_orders,
                  validation.test_energy, validation.test_convergence):
            t()
        return {name: (ok, detail) for name, ok, detail in validation.checks()}


@pytest.fixture(scope="module")
def completion_results():
    import completion_check as cc

    cc.RES.clear()
    for f in (cc.check_reproducibility, cc.check_arbitrary_phase,
              cc.check_arbitrary_spectrum, cc.check_gradients):
        f()
    return {name: (ok, detail) for name, ok, detail in cc.verdicts()}


VALIDATION = ["Free Gaussian propagation", "Grating equation, 1450-1650 nm",
              "Fraunhofer -> angular spectrum for z >> zR", "Diffraction-order efficiencies",
              "Energy conservation", "Wavelength-sampling convergence",
              "Spatial-sampling convergence"]
COMPLETION = ["Reproducible: repeated runs", "Reproducible: batching",
              "Any phase mask: energy", "Any phase mask: vs Rayleigh-Sommerfeld",
              "Any spectrum: linearity", "Any spectrum: R independent of source power",
              "Gradients dR/dphi vs finite differences"]


@pytest.mark.parametrize("name", VALIDATION)
def test_validation(validation_results, name):
    ok, detail = validation_results[name]
    assert ok, detail


@pytest.mark.parametrize("name", COMPLETION)
def test_completion(completion_results, name):
    ok, detail = completion_results[name]
    assert ok, detail
