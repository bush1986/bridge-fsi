import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
main = importlib.import_module('main')
generate_ccd_samples = main.generate_ccd_samples


def test_generate_ccd_samples():
    mu = {"U10": 30.0, "alpha": 0.0}
    std = {"U10": 5.0, "alpha": 1.0}
    df = generate_ccd_samples(mu, std, n_center=3)
    assert len(df) == 11
    assert set(df["type"]) == {"center", "corner", "axial"}
    assert df.attrs["center"] == mu
    delta = df.attrs["delta"]
    assert delta["U10"] == 5.0
    assert delta["alpha"] == 1.0