import pickle
from multiprocessing import Pool
import copy
import numpy as np
from utils.utils import load_config, get_parameter_range
from backtest.backtest import Backtest
from plotting.plot import plot_single_parameter_tuning_results, plot_heatmap_tuning_results

from pathlib import Path


OUTPUT_PATH = Path.joinpath(Path(__file__).parent.parent.parent.resolve(), "output")


class Tuning(object):
    """
    Class for parameter optimization
    """

    def __init__(self, params_to_tune: [str]):

        self.parameter_ranges = {param: None for param in params_to_tune}
        self.parameters_to_tune = params_to_tune
        self.seed = None
        self.config = None

    def run_opt_single(self, indices):

        bt = Backtest()
        this_config = copy.deepcopy(self.config)

        # Override the config with new values of eta for each algorithm
        for algorithm in self.config['algorithms']:
            for index, param in enumerate(self.parameters_to_tune):
                if this_config['local'][algorithm].get(param) is not None:
                    # The algorithm uses this parameter, so make sure we use the specified value for this run
                    this_config['local'][algorithm][f'{param}_optimal'] = False
                    this_config['local'][algorithm][param] = self.parameter_ranges[param][indices[index]]

        bt.run(config=this_config, store_output=False, random_seed=self.seed)

        return {
            a.name: a.losses.sum() for a in bt.algorithms.values()
        }

    def run_opt_parallel(self, config_name, seed):
        self.seed = seed
        self.config = load_config(config_name=config_name)

        tuning_parameters = self.config.get('tuning_parameters')
        n_tuning_steps = tuning_parameters['n_tuning_steps']
        n_processes = tuning_parameters['n_processes']

        for param in self.parameters_to_tune:
            p_min = tuning_parameters.get(f'{param}_min')
            p_max = tuning_parameters.get(f'{param}_max')
            self.parameter_ranges[param] = get_parameter_range(min=p_min, max=p_max, n_steps=n_tuning_steps)

        indices, keys = self.get_indices_and_keys(n_tuning_steps=n_tuning_steps)

        with Pool(processes=n_processes) as pool:
            return_values = pool.starmap(self.run_opt_single, indices)

        return dict(zip(keys, return_values))

    def get_indices_and_keys(self, n_tuning_steps):
        if len(self.parameters_to_tune) == 2:
            indices = [[(i, j)] for i in range(n_tuning_steps) for j in range(n_tuning_steps)]
            keys = [(i, j) for i in range(n_tuning_steps) for j in range(n_tuning_steps)]
        elif len(self.parameters_to_tune) == 1:
            indices = [[(i, 0)] for i in range(n_tuning_steps)]
            keys = [(i, 0) for i in range(n_tuning_steps)]
        else:
            raise NotImplementedError("Grid search only possible for either 1 or 2 parameters.")
        return indices, keys


def get_result_dict(algorithms, params_to_tune, n_tuning_steps, n_iterations_per_training_step):
    d = {}
    for algorithm in algorithms:
        if len(params_to_tune) == 2:
            d[algorithm] = np.empty((n_iterations_per_training_step, n_tuning_steps, n_tuning_steps))
        elif len(params_to_tune) == 1:
            d[algorithm] = np.empty((n_iterations_per_training_step, n_tuning_steps, 1))
        else:
            raise NotImplementedError("Only 2 parameter grid search set up currently.")
    return d


def main(config_name, params_to_tune):
    config = load_config(config_name=config_name)

    algorithms = config.get('algorithms')

    tuning_params = config.get('tuning_parameters')

    if tuning_params is None:
        raise ValueError(f"No tuning parameters provided in config {config_name}")

    # Get the required parameters from the provided config
    n_iterations_per_training_step = tuning_params['n_iterations_per_training_step']
    n_tuning_steps = tuning_params['n_tuning_steps']

    # Prepare the result dictionary to hold the output of tuning
    result_dict = get_result_dict(algorithms=algorithms,
                                  params_to_tune=params_to_tune,
                                  n_iterations_per_training_step=n_iterations_per_training_step,
                                  n_tuning_steps=n_tuning_steps)

    # Set the seed and change it for each iteration
    seed = tuning_params['seed']
    for iteration in range(n_iterations_per_training_step):
        # change the seed for multiple iterations of tuning
        seed += iteration

        tuner = Tuning(params_to_tune=params_to_tune)
        results = tuner.run_opt_parallel(config_name=config_name, seed=seed)

        for key, result in results.items():
            for algorithm, loss in result.items():
                result_dict[algorithm][iteration, key[0], key[1]] = loss

    save_to_path = Path.joinpath(OUTPUT_PATH, config_name, "tuning")
    save_to_path.mkdir(parents=True, exist_ok=True)

    parameter_ranges = {}
    for param in params_to_tune:
        p_min = tuning_params.get(f'{param}_min')
        p_max = tuning_params.get(f'{param}_max')
        parameter_ranges[param] = get_parameter_range(min=p_min, max=p_max, n_steps=n_tuning_steps)

    if len(params_to_tune) == 2:  # heatmap
        fig = plot_heatmap_tuning_results(result_dict=result_dict,
                                          parameter_ranges=parameter_ranges,
                                          params_to_tune=params_to_tune,
                                          save_to=save_to_path)
    else:
        fig = plot_single_parameter_tuning_results(result_dict=result_dict,
                                                   parameter_ranges=parameter_ranges,
                                                   params_to_tune=params_to_tune,
                                                   save_to=save_to_path)

    with open(Path.joinpath(save_to_path, f'results_dict_{"_".join(parameters_to_tune)}.pickle'), 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
