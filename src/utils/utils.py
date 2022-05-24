import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_name):
    config_path = Path.joinpath(Path(__file__).parent.parent.parent.resolve(), "config", f"{config_name}.json")
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config


def get_parameter_range(min, max, n_steps, linear=True):
    if linear:
        return np.linspace(min, max, n_steps)
    else:
        return np.exp(np.linspace(np.log(min), np.log(max), n_steps))


def get_task_epoch_matrix(n_tasks: int, T: int, super_pool_size: int, epoch_pool_size: int, epoch_block_pool_size: int, n_epochs: int,
                          n_switches_per_epoch: int, random=False):
    """
    Build a dataframe of shape (n_tasks, T), with each value representing the index of the best action for that task/trial.
    The small pool switches randomly between epochs.
    :param n_switches_per_epoch:
    :param n_tasks:
    :param T:
    :param super_pool_size:
    :param epoch_pool_size:
    :param n_epochs:
    :param random: Whether or not to switch randomly between experts during an epoch, or to cycle through them
    :return: pandas dataframe
    """
    # ideal_epoch_length = np.ceil(T / n_epochs).astype(int)

    # If T is not perfectly divisible by n_epochs, we will have as many uniform epochs until the last
    # e.g., with T=14 and 3 epochs we will have epoch lengths of [5, 5, 4].
    # The same logic is used for switches within an epoch when get_task_matrix is called
    # epoch_lengths = [ideal_epoch_length + np.min([T - (i + 1) * ideal_epoch_length, 0]) for i in range(n_epochs)]

    assert(T % n_epochs == 0)  # force all epochs to have equal length

    epoch_length = np.ceil(T / n_epochs).astype(int)

    if not random:
        assert (super_pool_size % epoch_pool_size == 0)  # use the entire super pool across epochs

    if random:
        if n_tasks > 1:
            epoch_block_pools = [np.random.choice(a=super_pool_size, size=epoch_block_pool_size, replace=True) for i in range(n_epochs)]
            # epoch_pools = [{t: epoch_block_pools[i] for t in range(n_tasks)} for i in range(n_epochs)]
        else:
            epoch_block_pools = [super_pool_size for i in range(n_epochs)]  # sample from the super pool for each epoch when 1 task

        epoch_pools = [{t: np.random.choice(a=epoch_block_pools[i], size=epoch_pool_size, replace=True) for t in range(n_tasks)} for i in range(n_epochs)]
        # epoch_pools = [{t: np.random.choice(a=super_pool_size, size=epoch_pool_size, replace=True) for t in range(n_tasks)} for i in range(n_epochs)]
    else:
        epoch_pools = [{t: np.array(range(epoch_pool_size)) + ((i * epoch_pool_size) % super_pool_size) for t in range(n_tasks)} for i in
                       range(n_epochs)]

    epochs = [get_task_matrix(n_tasks=n_tasks,
                              T=epoch_length,
                              expert_pool=e_pool,  # a dict, one pool for each task
                              K=n_switches_per_epoch,
                              random=random) for e_pool in epoch_pools]

    return pd.concat(epochs).reset_index(drop=True)


def get_task_matrix(n_tasks: int, T: int, expert_pool: [dict], K: int, random=False):
    """
    Build a dataframe of shape (T, n_tasks), with each value representing the index of the best action for
    that task/trial. (Single epoch).
    :param n_tasks:
    :param T:
    :param expert_pool:
    :param K: the number of switches in this epoch
    :param random: Whether or not to switch randomly between experts during an epoch, or to cycle through them
    :return:
    """

    task_matrix = pd.DataFrame(np.empty([T, n_tasks], dtype=np.int8))

    assert(T % (K+1) == 0)  # force all segments to be the same length

    segment_length = np.ceil(T / (K + 1)).astype(int)

    for task in range(n_tasks):
        e_pool = expert_pool[task]
        m = len(e_pool)
        for k_ in range(K + 1):
            if random:
                task_matrix.loc[k_ * segment_length:(k_ + 1) * segment_length, task] = np.random.choice(e_pool)
            else:
                task_matrix.loc[k_ * segment_length:(k_ + 1) * segment_length, task] = e_pool[(task + k_) % m]

    return task_matrix
