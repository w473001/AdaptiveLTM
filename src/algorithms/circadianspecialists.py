import numpy as np
from .algorithms import Algo
import logging

logger = logging.getLogger(__name__)


class MarkovSpecialists(Algo):

    def __init__(self, alpha_optimal, theta_optimal, **kwargs):
        self.small_pool = kwargs.get('small_pool')
        Algo.__init__(self, **kwargs)
        self.name = "Circ. Sp."

        self.p_w = None
        self.p_s = None
        self.p_ww = None
        self.p_ws = None
        self.p_ss = None
        self.p_sw = None

        if alpha_optimal and theta_optimal:
            self.tune(n_switches=self.n_switches, pool_size=self.small_pool, total_trials=self.total_trials)
        else:
            self.p_ws = kwargs['alpha']
            self.p_ww = 1.0 - self.p_ws

            self.p_sw = kwargs['theta']
            self.p_ss = 1.0 - self.p_sw

            self.p_w = kwargs['theta'] / (kwargs['alpha'] + kwargs['theta'])
            self.p_s = 1.0 - self.p_w

        # weights of Algo class are normalized after init() so need to correct this
        self.w = np.ones(self.n_actions) * (self.p_w / self.n_actions)
        self.s = np.ones(self.n_actions) * (self.p_s / self.n_actions)

    def predict(self, **kwargs):
        # Normalize weights for prediction and storage for the specialists algorithm
        w_ = self.w / self.w.sum()
        decision = np.random.choice(a=self.n_actions, p=w_)
        # Store the action (arm) chosen, as we need it to store losses
        self.last_decision = decision

        return decision

    def get_loss(self, loss_vector):
        w = self.w / self.w.sum()
        return w.dot(loss_vector)  # default for e.g. Hedge, NOT for bandits.

    def loss_update(self, loss_vector):
        w_sum = self.w.sum()

        self.w *= np.exp(-self.eta * loss_vector)
        self.w /= self.w.sum()
        self.w *= w_sum

    def switching_update(self):
        # weights are normalized after loss update
        w_temp = self.w.copy()
        s_temp = self.s.copy()

        self.w = self.p_ww * w_temp + self.p_sw * s_temp
        self.s = self.p_ss * s_temp + self.p_ws * w_temp

    def tune(self, n_switches, pool_size, total_trials):
        self.p_w = 1.0 / pool_size
        self.p_s = 1.0 - self.p_w

        self.p_ws = n_switches / (total_trials - 1)
        self.p_ww = 1.0 - self.p_ws

        self.p_sw = n_switches / ((pool_size - 1) * (total_trials - 1))
        self.p_ss = 1.0 - self.p_sw

    def tune_eta(self):
        m = self.small_pool
        k = self.n_switches
        T = self.total_trials
        n = self.n_actions
        self.eta = np.sqrt(2 * (m * np.log(n / m) + m * self.H(1 / m) + (T - 1) * self.H(k / (T - 1)) + ((m - 1) * (T - 1)) * self.H(
            k / ((m - 1) * (T - 1)))) / T)


class MarkovSpecialistsFactory:
    """Factory for properly initializing the markov specialists algorithm"""
    def create(self, algo_config, global_config) -> Algo:
        algo_config = algo_config.get('MARKOV_SPECIALISTS')
        return MarkovSpecialists(**{**global_config, **algo_config})

