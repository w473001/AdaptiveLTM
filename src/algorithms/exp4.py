import numpy as np
from .algorithms import Algo, SwitchingAlgo
from .circadianspecialists import MarkovSpecialists
import logging

logger = logging.getLogger(__name__)


class EXP4(Algo):
    def __init__(self, n_experts, **kwargs):
        self.n_experts = n_experts
        self.name = "Exp4"
        Algo.__init__(self, **kwargs)
        self.w_experts = np.ones(self.n_experts) / self.n_experts
        self.last_expert_predictions = np.zeros((self.n_experts, self.n_actions))

    def get_loss(self, loss_vector):
        return loss_vector[self.last_decision]

    def loss_update(self, loss_vector):
        bandit_loss_vector = np.zeros_like(loss_vector)
        bandit_loss_vector[self.last_decision] = loss_vector[self.last_decision] / self.w[self.last_decision]

        expert_loss_vector = self.last_expert_predictions.dot(bandit_loss_vector)
        self.w_experts *= np.exp(-self.eta * expert_loss_vector)
        self.w_experts /= self.w_experts.sum()

    def predict(self, expert_predictions):
        """
        Given A actions and N experts, we receive a matrix of size (NxA).
        We convert this to a distribution over actions size (1xA).
        We then sample our action from this distribution.

        :param expert_matrix:
        :return:
        """
        assert (expert_predictions.shape[0] == self.n_experts)
        assert (expert_predictions.shape[1] == self.n_actions)
        self.last_expert_predictions = expert_predictions
        self.w = self.w_experts.dot(expert_predictions)
        decision = np.random.choice(a=self.n_actions, p=self.w)
        # Store the action (arm) chosen, as we need it to store losses
        self.last_decision = decision
        return decision

    def tune_eta(self):
        self.eta = np.sqrt(2 * np.log(self.n_experts) / (self.total_trials * self.n_actions))
        logger.info(f"Setting eta to {self.eta} for {self.name}")


class FixedShareEXP4(EXP4, SwitchingAlgo):
    def __init__(self, n_experts, **kwargs):
        self.n_experts = n_experts
        self.w_experts = np.ones(self.n_experts) / self.n_experts
        self.name = "Fixed Share (Exp4)"
        SwitchingAlgo.__init__(self, **kwargs)
        self.last_expert_predictions = np.zeros((self.n_experts, self.n_actions))

    def tune_eta(self):
        k = self.n_switches
        a = self.n_actions
        T = self.total_trials
        n = self.n_experts
        self.eta = np.sqrt(2 * ((k + 1) * np.log(n) + (T - 1) * self.H(k / (T - 1))) / (a * T))
        logger.info(f"Setting eta to {self.eta} for {self.name}")

    def switching_update(self):
        self.w_experts *= (1 - self.alpha)
        self.w_experts += (self.alpha / self.n_experts)


class MarkovSpecialistsEXP4(EXP4, MarkovSpecialists):
    def __init__(self, n_experts, **kwargs):
        self.small_pool = kwargs.get('small_pool')
        EXP4.__init__(self, n_experts, **kwargs)
        self.name = "Circ. Sp. (Exp4)"
        MarkovSpecialists.__init__(self, **kwargs)
        # need to create wake/sleep weights over experts not actions
        self.w = None
        del self.s
        self.w_experts = np.ones(self.n_experts) * (self.p_w / self.n_experts)
        self.s_experts = np.ones(self.n_experts) * (self.p_s / self.n_experts)

    def tune_eta(self):
        m = self.small_pool
        k = self.n_switches
        T = self.total_trials
        n = self.n_experts
        a = self.n_actions
        self.eta = np.sqrt(2 * (m * np.log(n / m) + m * self.H(1 / m) + (T - 1) * self.H(k / (T - 1)) + ((m - 1) * (T - 1)) * self.H(
            k / ((m - 1) * (T - 1)))) / (n * T))
        logger.info(f"Setting eta to {self.eta} for {self.name}")

    def predict(self, expert_predictions):
        """
        Given A actions and N experts, we receive a matrix of size (NxA).
        We convert this to a distribution over actions size (1xA).
        We then sample our action from this distribution.

        :param expert_matrix:
        :return:
        """
        assert (expert_predictions.shape[0] == self.n_experts)
        assert (expert_predictions.shape[1] == self.n_actions)

        self.last_expert_predictions = expert_predictions

        w_experts = self.w_experts / self.w_experts.sum()  # normalize expert weights
        self.w = w_experts.dot(expert_predictions)

        decision = np.random.choice(a=self.n_actions, p=self.w)

        # Store the action (arm) chosen, as we need it to store losses
        self.last_decision = decision

        return decision

    def loss_update(self, loss_vector):
        w_sum = self.w_experts.sum()
        EXP4.loss_update(self, loss_vector)
        self.w_experts *= w_sum
        assert(np.isclose(self.w_experts.sum(), self.p_w))

    def switching_update(self):
        # weights are normalized after loss update
        w_temp = self.w_experts.copy()
        s_temp = self.s_experts.copy()

        self.w_experts = self.p_ww * w_temp + self.p_sw * s_temp
        self.s_experts = self.p_ss * s_temp + self.p_ws * w_temp


class EXP4Factory:
    """Factory for properly initializing the EXP4 algorithm"""
    def create(self, algo_config, global_config) -> EXP4:
        algo_config = algo_config.get('EXP4')
        return EXP4(**{**global_config, **algo_config})


class FixedShareEXP4Factory:
    """Factory for properly initializing the fixed share algorithm"""
    def create(self, algo_config, global_config) -> Algo:
        algo_config = algo_config.get('FIXED_SHARE_EXP4')
        return FixedShareEXP4(**{**global_config, **algo_config})


class MarkovSpecialistsEXP4Factory:
    """Factory for properly initializing the fixed share algorithm"""
    def create(self, algo_config, global_config) -> Algo:
        algo_config = algo_config.get('MARKOV_SPECIALISTS_EXP4')
        return MarkovSpecialistsEXP4(**{**global_config, **algo_config})
