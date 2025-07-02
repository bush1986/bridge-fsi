import importlib
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
main = importlib.import_module('main')


def test_multi_rsm_fit():
    mu, std, dists = main.define_random_variables()
    samples = main.generate_ccd_samples(mu, std, n_center=1)
    sims = [main.run_coupled_simulation(s, seed=i) for i, s in enumerate(samples.to_dict('records'))]
    df = pd.DataFrame(sims)
    rsm = main.MultiRSM(pop_size=0, n_gen=0)
    rsm.fit_all(samples, df)
    preds = rsm.predict('lambda', samples)
    mse = ((preds - df['lambda'].values)**2).mean()
    assert mse < 1e-3


def test_iterate_convergence():
    mu, std, dists = main.define_random_variables()
    rsm, history = main.iterate_until_convergence(mu, std, dists, max_iter=5, pop_size=0, n_gen=0)
    assert len(history) <= 5