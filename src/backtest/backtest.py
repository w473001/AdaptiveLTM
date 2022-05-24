import logging
from utils import utils

from algorithms.multitask import MultiTaskAlgo
from algorithms.utils import get_algo_factory
from plotting.plot import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
OUTPUT_PATH = Path.joinpath(Path(__file__).parent.parent.parent.resolve(), "output")
MULTI_TASK_ALGORITHMS = ["SWARM", "ADAPTLTM"]  # list of algorithms that don't need a MultiTaskAlgo wrapper class for multi-task


class Task(object):
    def __init__(self, n_actions, n_experts, good_loss_upper_bound, bad_loss_upper_bound, total_trials):
        self.n_actions = n_actions
        self.n_experts = n_experts
        self.good_loss_upper_bound = good_loss_upper_bound
        self.bad_loss_upper_bound = bad_loss_upper_bound
        self.good_index = 0  # index of the best action
        self.good_expert = 0  # index of the best expert for EXP4 algos
        self.losses = None
        self.total_trials = total_trials
        self.best_action_losses = np.zeros(self.total_trials)

    def generate_losses(self):
        """
        Uniformly generate losses in [0,b] for all actions, except for the good action where we instead
        sample a loss uniformly from [0,a]. Here a < b.

        In the bandit setting the "masking" of zeros on non-chosen actions is done by the algorithm class's
        loss_update() method.

        :return: np.array in [0, 1]^n (or really [0, b]^n)
        """
        good_loss = np.random.uniform(high=self.good_loss_upper_bound)
        losses = np.random.uniform(high=self.bad_loss_upper_bound, size=self.n_actions)
        losses[self.good_index] = good_loss
        return losses

    def store_losses(self, t, loss_vector):
        self.losses[t, :] = loss_vector
        self.best_action_losses[t] = loss_vector[self.good_index]

    def get_expert_predictions(self, epsilon, uniform_bad=False):
        """
        The best expert (policy) has the following distribution:
        P(i) = 1 - epsilon       if i= the 'good' action
        P(i) = epsilon/(a-1)     otherwise

        If uniform_bad  is True then:
            The other experts (policies) have the following distribution:
            P(i) = 1 / a
            for all actions(just a uniform distribution over all actions).

        If uniform_bad is False, then each bad expert's policy will be uniformly sampled at random from the simplex


        :param uniform_bad:
        :param epsilon:
        :return:
        """
        if self.n_experts is not None:

            if uniform_bad:
                # Bad experts have uniform distribution on all actions except the good one (zero)
                e = np.ones((self.n_experts, self.n_actions)) / self.n_actions
            else:
                e = np.random.rand(self.n_experts, self.n_actions)
                # normalize each expert's distribution over the actions
                row_sums = e.sum(axis=1)
                e = e / row_sums[:, np.newaxis]

            # Concentrate the good expert's weight to the best action (and epsilon-uniform on others)
            e[self.good_expert, :] = epsilon / (self.n_actions - 1)
            e[self.good_expert, self.good_index] = 1 - epsilon

            assert (np.allclose(e.sum(axis=1), np.ones(self.n_experts)))
            return e
        else:
            return None

    def set_good_index(self, index):
        self.good_index = index

    def set_good_expert(self, index):
        self.good_expert = index


class Backtest(object):

    def __init__(self):
        """
        n_tasks: int, T:int, super_pool: list, epoch_pool_size: int, n_epochs:int, K: int, random=True):
        """
        # Global parameters to be read from config
        self.n_tasks = None
        self.n_actions = None
        self.total_trials_global = None
        self.n_epochs = None
        self.n_switches_per_epoch = None
        self.super_pool_size = None
        self.epoch_pool_size = None
        self.epoch_block_pool_size = None
        self.good_loss_upper_bound = None
        self.bad_loss_upper_bound = None
        self.random_switches_within_epoch = None
        self.n_experts = None
        self.epsilon = None
        self.task_matrix = None
        self.non_random_task_matrix = None  # if we randomize experts then we still use this for plotting segments, epochs etc.

        self.algorithms = []
        self.algorithm_loss_holder = None  # for multiple repeats
        self.tasks = {}

    def load_values_from_config(self, config):
        global_config = config.get('global')

        self.n_tasks = global_config.get('n_tasks')
        self.n_actions = global_config.get('n_actions')
        self.n_epochs = global_config.get('n_epochs')
        self.n_switches_per_epoch = global_config.get('n_switches_per_epoch')
        self.total_trials_global = global_config.get('total_trials_global')
        self.super_pool_size = global_config.get('super_pool_size')
        self.epoch_pool_size = global_config.get('epoch_pool_size')
        self.epoch_block_pool_size = global_config.get('epoch_block_pool_size')
        self.random_switches_within_epoch = global_config.get('random_switches_within_epoch')
        self.good_loss_upper_bound = global_config.get('good_loss_upper_bound')
        self.bad_loss_upper_bound = global_config.get('bad_loss_upper_bound')
        self.n_experts = global_config.get('n_experts')
        self.epsilon = global_config.get('epsilon')

        assert(self.total_trials_global % self.n_tasks == 0)  # force all tasks to have the same number of trials

        total_trials_per_task = np.ceil(self.total_trials_global / self.n_tasks).astype(int)

        self.tasks = {i: Task(n_actions=self.n_actions,
                              n_experts=self.n_experts,
                              good_loss_upper_bound=self.good_loss_upper_bound,
                              bad_loss_upper_bound=self.bad_loss_upper_bound,
                              total_trials=total_trials_per_task)
                      for i in range(self.n_tasks)}

    def run(self, config: dict = None, from_config: str = None, store_output=True, random_seed=22222):
        """
        Run the backtest with the provided config, or specify the config name with from_config
        to load and run that config
        :param random_seed:
        :param config: dictionary (loaded from config file)
        :param from_config: name of config file to load
        :param store_output:
        :return:
        """
        if config is None:
            assert (from_config is not None)
            config_name = from_config
            config = utils.load_config(config_name)

        self.load_values_from_config(config)

        algorithm_names = config.get("algorithms")

        logger.info(f"""
        Starting backtest with global values:
            n_tasks = {self.n_tasks}
            n_actions = {self.n_actions}
            total_trials_global = {self.total_trials_global}
            n_epochs = {self.n_epochs}
            n_switches_per_epoch = {self.n_switches_per_epoch}
            super_pool_size = {self.super_pool_size}
            epoch_pool_size = {self.epoch_pool_size}
            epoch_block_pool_size = {self.epoch_block_pool_size}
            random_switches_within_epoch = {self.random_switches_within_epoch}
            good_loss_upper_bound = {self.good_loss_upper_bound}
            bad_loss_upper_bound = {self.bad_loss_upper_bound}
            n_experts = {self.n_experts}
            """)

        # For plotting
        self.non_random_task_matrix = utils.get_task_epoch_matrix(n_tasks=self.n_tasks,
                                                                  T=self.total_trials_global,
                                                                  super_pool_size=self.super_pool_size,
                                                                  epoch_pool_size=self.epoch_pool_size,
                                                                  epoch_block_pool_size=self.epoch_block_pool_size,
                                                                  n_epochs=self.n_epochs,
                                                                  n_switches_per_epoch=self.n_switches_per_epoch,
                                                                  random=False)

        algo_config = config.get('local')
        global_config = config.get('global')

        # Initialize all algorithms for testing
        algorithm_factory = get_algo_factory()

        # n_runs = config.get('global').get('n_runs')
        # assert (n_runs >= 1)
        self.algorithm_loss_holder = {a: np.zeros(self.total_trials_global) for a in algorithm_names}
        # add a loss holder for the best expert
        self.algorithm_loss_holder['BEST_EXPERT'] = np.zeros(self.total_trials_global)

        np.random.seed(random_seed)

        # Derive the entire task matrix beforehand, useful for plotting
        self.task_matrix = utils.get_task_epoch_matrix(n_tasks=self.n_tasks,
                                                       T=self.total_trials_global,
                                                       super_pool_size=self.super_pool_size,
                                                       epoch_pool_size=self.epoch_pool_size,
                                                       epoch_block_pool_size=self.epoch_block_pool_size,
                                                       n_epochs=self.n_epochs,
                                                       n_switches_per_epoch=self.n_switches_per_epoch,
                                                       random=self.random_switches_within_epoch)

        if store_output:  # store self.task_matrix
            assert (config_name is not None)
            Path.joinpath(OUTPUT_PATH, config_name).mkdir(parents=True, exist_ok=True)
            self.task_matrix.to_csv(Path.joinpath(OUTPUT_PATH, config_name, f'task_matrix_{random_seed}.csv'))

        self.algorithms = {}  # fresh instances of algorithms for each run
        for algo in algorithm_names:
            if algo in MULTI_TASK_ALGORITHMS:
                self.algorithms[algo] = algorithm_factory.create(name=algo, algo_config=algo_config,
                                                                 global_config=global_config)
            else:
                self.algorithms[algo] = MultiTaskAlgo(algo_name=algo, algo_config=algo_config,
                                                      global_config=global_config)

        task_number = -1
        for t_ in range(self.total_trials_global):
            if t_ % 500 == 0:
                logger.info(f"Trial: {t_}...")

            # iterate through tasks in order, one task per global trial
            task_number = (task_number + 1) % self.n_tasks

            # Set the good action index from the self.task_matrix (and expert index for EXP4)
            self.tasks[task_number].set_good_index(np.random.choice(self.n_actions))
            self.tasks[task_number].set_good_expert(self.task_matrix.loc[t_, task_number])

            # Get the expert predictions
            expert_predictions = self.tasks[task_number].get_expert_predictions(epsilon=self.epsilon)

            # Get the losses for this task/trial
            loss_vector = self.tasks[task_number].generate_losses()

            for algo_name in algorithm_names:
                self.algorithms[algo_name].predict(task_number=task_number, expert_predictions=expert_predictions)
                self.algorithms[algo_name].update(task_number=task_number, loss_vector=loss_vector)

            # Store the best expert's loss
            best_experts_decision = np.random.choice(a=self.n_actions,
                                                     p=expert_predictions[self.task_matrix.loc[t_, task_number], :])
            self.algorithm_loss_holder['BEST_EXPERT'][t_] = loss_vector[best_experts_decision]

        for algo_name in algorithm_names:
            logger.info(f"{self.algorithms[algo_name].name} loss={self.algorithms[algo_name].losses.sum()}")
            self.algorithm_loss_holder[algo_name][:] = self.algorithms[algo_name].losses

        if store_output:  # store self.task_matrix and loss plots
            assert (config_name is not None)

            Path.joinpath(OUTPUT_PATH, config_name).mkdir(parents=True, exist_ok=True)

            plot_cumulative_losses(backtest=self, config_name=config_name, save=True)

            plot_task_matrix(config_name=config_name, n_tasks=self.n_tasks, task_matrix=self.task_matrix,
                             n_epochs=self.n_epochs, total_trials_global=self.total_trials_global,
                             super_pool_size=self.super_pool_size)
