from algorithms.utils import get_algo_factory
import numpy as np
import logging
logger = logging.getLogger(__name__)


class MultiTaskAlgo:
    """
    A wrapper class for algorithms which are not designed for multi-task learning. Instead
    we have an individual instance of a given algorithm for each task. These instances are independent so there
    is no inter-task learning.
    """
    def __init__(self, algo_name, algo_config, global_config):
        # self.name = f"{algo_name} (Multi)"
        self.name = algo_name
        algorithm_factory = get_algo_factory()
        self.global_t = 0
        self.losses = np.zeros(global_config.get('total_trials_global'))
        global_config_ = global_config.copy()

        n_tasks = global_config_.get('n_tasks')
        global_config_['n_switches'] = global_config_.get('n_switches_per_epoch') * global_config_.get('n_epochs')

        # Assume the same number of trials for each task
        global_config_['total_trials'] = np.ceil(global_config_.get('total_trials_global') / global_config_.get('n_tasks')).astype(int)

        global_config_['small_pool'] = int(global_config_.get('epoch_pool_size') * global_config.get('n_epochs'))

        self.instances = {i: algorithm_factory.create(name=algo_name,
                                                      algo_config=algo_config,
                                                      global_config=global_config_) for i in range(n_tasks)}

    def predict(self, task_number, **kwargs):
        return self.instances[task_number].predict(**kwargs)

    def update(self, task_number, **kwargs):
        self.instances[task_number].update(**kwargs)

        # copy the task instance's loss into the global loss vector
        self.losses[self.global_t] = self.instances[task_number].losses[self.instances[task_number].t-1]
        self.global_t += 1
