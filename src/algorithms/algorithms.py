import numpy as np
import logging

logger = logging.getLogger(__name__)


class Algo(object):
    def __init__(self, eta=None, n_actions=None, eta_optimal=False, total_trials=None, n_switches=None, **kwargs):
        # Global config values (n, T, k etc)
        self.n_actions = n_actions
        self.total_trials = total_trials
        self.n_switches = n_switches

        # config parameters
        if eta_optimal:
            self.tune_eta()
        else:
            self.eta = eta

        self.w = np.ones(self.n_actions) / self.n_actions  # normalized weights

        # Keep track of performance internally
        self.last_decision = None
        self.losses = np.zeros(self.total_trials)
        # self.weights = np.zeros([self.total_trials, self.n_actions])
        self.t = 0

    def tune_eta(self):
        raise NotImplementedError(f"Need to define tuning of eta.")

    def H(self, p):
        return -p * np.log(p) - (1-p) * np.log(1-p)

    def predict(self, **kwargs):
        decision = np.random.choice(a=self.n_actions, p=self.w)
        # Store the action (arm) chosen, as we need it to store losses
        self.last_decision = decision

        # Store weights
        # self.weights[self.t, :] = self.w
        return decision

    def get_loss(self, loss_vector):
        return self.w.dot(loss_vector)  # default for e.g. Hedge, NOT for bandit algorithms

    def update(self, loss_vector, **kwargs):
        self.losses[self.t] = self.get_loss(loss_vector=loss_vector)
        self.loss_update(loss_vector=loss_vector, **kwargs)
        self.switching_update(**kwargs)
        self.t += 1

    def loss_update(self, loss_vector, **kwargs):
        self.w *= np.exp(-self.eta * loss_vector)
        self.w /= self.w.sum()

    def switching_update(self, **kwargs):
        pass

    def get_weights(self):
        return self.w

    def tune(self, **kwargs):
        pass


class SwitchingAlgo(Algo):
    def __init__(self, alpha_optimal, alpha, **kwargs):
        """
        If alpha is provided, the n_switches and total_trials must be None.
        If n_switches and total_trials are provided then alpha must be None, as it will be set to the optimal value
        :param total_trials: int
        :param n_switches: int
        :param alpha: float in [0,1]
        :param kwargs:Algo expects n_actions, eta
        """
        Algo.__init__(self, **kwargs)

        if alpha_optimal:
            n_switches = kwargs['n_switches']
            total_trials = kwargs['total_trials']
            self.tune(n_switches=n_switches, total_trials=total_trials)
        else:
            self.alpha = alpha

    def tune(self, n_switches, total_trials):
        self.alpha = n_switches / (total_trials - 1)
        logger.info(f"Using optimal value of alpha = {self.alpha} for algorithm {self.name}")


