import pickle
from multiprocessing import Pool
import copy
import numpy as np
from utils.utils import load_config, get_parameter_range
from src.backtest.backtest import Backtest
from plotting.plot import plot_single_parameter_tuning_results, plot_heatmap_tuning_results

from pathlib import Path


OUTPUT_PATH = Path.joinpath(Path(__file__).parent.parent.parent.resolve(), "output")


class ParallelBacktest(object):
    """
    A class to run backtests in parallel
    """

    def __init__(self):
        self.base_seed = None
        self.config = None

    def run_bt_single(self, indices):
        bt = Backtest()
        seed = self.base_seed + indices[0]  # everything is the same except the seed we feed to the backtest run method
        bt.run(config=self.config, store_output=False, random_seed=seed)
        return bt.algorithm_loss_holder

    def run_bt_parallel(self, config, base_seed):
        self.base_seed = base_seed
        self.config = config
        n_runs = self.config['global']['n_runs']
        indices, keys = get_indices_and_keys(n_runs=n_runs)

        # use the n_processes value set in the tuning parameters
        n_processes = self.config['tuning_parameters']['n_processes']
        with Pool(processes=n_processes) as pool:
            return_values = pool.starmap(self.run_bt_single, indices)

        return dict(zip(keys, return_values))


def get_indices_and_keys(n_runs):
    indices = [[(i, 0)] for i in range(n_runs)]
    keys = [(i, 0) for i in range(n_runs)]
    return indices, keys


def run_parallel_backtest(config=None, from_config=None, base_seed=12345, save=True):
    if config is None:
        assert (from_config is not None)
        config_name = from_config
        config = load_config(config_name=config_name)

    algorithms = config.get('algorithms')
    all_algorithms = algorithms.copy()
    all_algorithms.append('BEST_EXPERT')

    n_runs = config['global']['n_runs']

    total_trials = config['global']['total_trials_global']
    # Prepare the result dictionary to hold the output of each run
    result_dict = {a: np.zeros((n_runs, total_trials)) for a in all_algorithms}

    pbt = ParallelBacktest()
    results = pbt.run_bt_parallel(config=config, base_seed=base_seed)

    for run, algo_loss_holder in results.items():
        for algo, loss in algo_loss_holder.items():
            result_dict[algo][run[0], :] = loss

    if save:
        save_to_path = Path.joinpath(OUTPUT_PATH, config_name, "run_output")
        save_to_path.mkdir(parents=True, exist_ok=True)

        with open(Path.joinpath(save_to_path, f'results_dict_multi.pickle'), 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(Path.joinpath(save_to_path, f'results_dict_multi_{n_runs}.pickle'), 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  # create a back up

    else:
        return result_dict
