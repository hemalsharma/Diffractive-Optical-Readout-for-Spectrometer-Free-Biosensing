"""Each module's own `python src/<module>.py` self-check."""

import runpy

import pytest

from conftest import SRC


@pytest.mark.parametrize("module", ["sensor_model", "doe", "propagation", "detector"])
def test_module_self_check(module, capsys):
    runpy.run_path(str(SRC / f"{module}.py"), run_name="__main__")
    assert "all self-checks passed" in capsys.readouterr().out
