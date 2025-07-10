import importlib
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
mod = importlib.import_module('bridge_wind_fragility')


def test_multi_rsm_fit():
    mu, std, dists = mod.define_random_variables()
    samples = mod.generate_ccd_samples(mu, std, n_center=1)
    sims = [mod.run_coupled_simulation(None, s, seed=i) for i, s in enumerate(samples.to_dict('records'))]
    df = pd.DataFrame(sims)
    rsm = mod.MultiRSM(pop_size=0, n_gen=0)
    rsm.fit_all(samples, df)
    preds = rsm.predict('Ucr', samples)
    assert preds.shape[0] == len(samples)


def test_iterate_convergence():
    mu, std, dists = mod.define_random_variables()
    g = mod.create_limit_state("Ucr - U10")
    rsm, history = mod.iterate_until_convergence(
        None,
        mu,
        std,
        dists,
        g_func=g,
        max_iter=3,
        pop_size=0,
        n_gen=0,
        use_fsi=False,
    )
    assert len(history) <= 3
