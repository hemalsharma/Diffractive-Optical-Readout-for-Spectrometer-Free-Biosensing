"""Quick checks of the public forward-model interface."""

import csv

import pytest
import torch

from conftest import ROOT
from config import CFG
from doe import linear_grating
from main import forward, forward_from_spectrum


@pytest.fixture(scope="module")
def table_run():
    return forward(torch.linspace(0, CFG.c_max, 9, dtype=torch.float64))


def test_matches_committed_detector_table(table_run):
    """forward() reproduces results/final_detector_table.csv."""
    r = table_run
    with open(ROOT / "results" / "final_detector_table.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(r["c"])
    for k, row in enumerate(rows):
        assert float(row["c"]) == pytest.approx(float(r["c"][k]))
        assert float(row["R"]) == pytest.approx(float(r["R"][k]), rel=1e-6)
        assert float(row["delta_R"]) == pytest.approx(float(r["R"][k] - r["R"][0]), abs=1e-10)
        assert float(row["I_A_over_I_A0"]) == pytest.approx(float(r["IA"][k] / r["IA"][0]), rel=1e-7)
        assert float(row["I_B_over_I_B0"]) == pytest.approx(float(r["IB"][k] / r["IB"][0]), rel=1e-7)


def test_supplied_spectrum_reproduces_forward(table_run):
    r = table_run
    phi = linear_grating(CFG.n_doe, CFG.pitch, CFG.period)
    s = forward_from_spectrum(r["lam_m"], r["S"], phi, dlam_nm=CFG.dlam_nm(r["sensor"]))
    for key in ("I", "IA", "IB", "R"):
        assert torch.equal(s[key], r[key])


@pytest.mark.parametrize("bad", [
    dict(lam_m=[1.55e-6, 1.54e-6], S=[1.0, 1.0]),              # not increasing
    dict(lam_m=[1.55e-6, -1.0], S=[1.0, 1.0]),                 # negative wavelength
    dict(lam_m=[1.54e-6, 1.55e-6], S=[1.0, 1.0, 1.0]),         # shape mismatch
    dict(lam_m=[1.54e-6, 1.55e-6], S=[1.0, 1.0], phi=torch.zeros(10, 10)),
    dict(lam_m=[1.55e-6], S=[1.0]),                            # one sample, no dlam_nm
])
def test_forward_from_spectrum_rejects_bad_input(bad):
    phi = bad.pop("phi", torch.zeros(CFG.n_doe, CFG.n_doe, dtype=torch.float64))
    with pytest.raises(ValueError):
        forward_from_spectrum(bad["lam_m"], bad["S"], phi)
