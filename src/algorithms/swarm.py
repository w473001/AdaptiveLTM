import logging

import numpy as np

from .algorithms import Algo

logger = logging.getLogger(__name__)


class SwarmEXP4(Algo):
    def __init__(self, n_tasks, n_experts, **kwargs):
        self.n_experts = n_experts
        self.small_pool = kwargs.get('small_pool')
        self.n_tasks = n_tasks

        Algo.__init__(self, **kwargs)
        self.name = "SWARM"

        logger.info(f"Eta set to {self.eta} for {self.name}")
        self.last_expert_predictions = np.zeros((self.n_experts, self.n_actions))

        self.pi = np.ones(self.n_experts) / self.n_experts
        self.mu = 1 / self.small_pool
        logger.info(f"Setting mu to {self.mu} for {self.name}")
        self.theta = 1 - (self.n_switches / (self.total_trials - self.n_tasks))  # 1-k/(T-s)
        logger.info(f"Setting theta to {self.theta} for {self.name}")
        logger.info(f"alpha would be {1 - self.theta}")
        self.phi = self.n_switches / ((self.small_pool - 1) * (self.total_trials - self.n_tasks))  # k/((m-1)(T-s))
        logger.info(f"Setting phi to {self.phi} for {self.name}")
        self.w = np.zeros([self.n_tasks, self.n_actions])  # action weights,
        self.w_experts = np.ones([self.n_tasks, self.n_experts]) * self.mu

    def get_loss(self, loss_vector):
        return loss_vector[self.last_decision]

    def predict(self, task_number, expert_predictions):
        """
        Given A actions and N experts, we receive a matrix of size (NxA).
        We convert this to a distribution over actions size (1xA).
        We then sample our action from this distribution.

        :param expert_predictions:
        :param task_number:
        :param expert_matrix:
        :return:
        """
        assert (expert_predictions.shape[0] == self.n_experts)
        assert (expert_predictions.shape[1] == self.n_actions)
        self.last_expert_predictions = expert_predictions

        # convert expert weights to actions weights
        w_ = self.pi * self.w_experts[task_number, :]
        w_ /= w_.sum()

        self.w[task_number, :] = w_.dot(expert_predictions)

        assert (np.isclose(self.w[task_number, :].sum(), 1))

        decision = np.random.choice(a=self.n_actions, p=self.w[task_number, :])

        # Store the action (arm) chosen, as we need it to store losses
        self.last_decision = decision
        return decision

    def loss_update(self, loss_vector, task_number):
        bandit_loss_vector = np.zeros_like(loss_vector)
        bandit_loss_vector[self.last_decision] = loss_vector[self.last_decision] / self.w[
            task_number, self.last_decision]

        expert_loss_vector = self.last_expert_predictions @ bandit_loss_vector

        w_exp = self.w_experts[task_number, :].copy()
        delta = w_exp * np.exp(-self.eta * expert_loss_vector)
        beta = self.pi.dot(w_exp) / self.pi.dot(delta)
        epsilon = np.ones(self.n_experts) - w_exp + beta * delta
        self.pi = self.pi * epsilon
        self.w_experts[task_number, :] = (self.phi * (np.ones(self.n_experts) - w_exp) + self.theta * beta * delta) * (
                    1 / epsilon)

    def tune_eta(self):
        m = self.small_pool
        k = self.n_switches
        T = self.total_trials
        n = self.n_experts
        a = self.n_actions
        s = self.n_tasks
        self.eta = np.sqrt(2 * (m * np.log(n / m) + s * m * self.H(1 / m) + (T - s) * self.H(k / (T - s)) + (
                    (m - 1) * (T - s)) * self.H(
            k / ((m - 1) * (T - s)))) / (a * T))


class SwarmFactory:
    """Factory for properly initializing the SWARM algorithm"""

    def create(self, algo_config, global_config) -> Algo:
        algo_config = algo_config.get('SWARM')
        # add required parameters before creating object
        global_config_ = global_config.copy()
        global_config_['total_trials'] = global_config_.get('total_trials_global')
        #  n_switches is the total number of switches for swarm
        global_config_['n_switches'] = global_config_.get('n_switches_per_epoch') * global_config_.get(
            'n_epochs') * global_config_.get('n_tasks')
        global_config_['small_pool'] = global_config_.get('super_pool_size')
        return SwarmEXP4(**{**global_config_, **algo_config})
