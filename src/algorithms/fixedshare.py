import numpy as np
from .algorithms import SwitchingAlgo
import logging

logger = logging.getLogger(__name__)


class FixedShare(SwitchingAlgo):
    def __init__(self, **kwargs):
        self.name = "Fixed Share"
        SwitchingAlgo.__init__(self, **kwargs)

    def switching_update(self):
        self.w *= (1 - self.alpha)
        self.w += (self.alpha / self.n_actions)

    def tune_eta(self):
        k = self.n_switches
        T = self.total_trials
        n = self.n_actions
        self.eta = np.sqrt(2 * ((k + 1) * np.log(n) + (T - 1) * self.H(k / (T - 1))) / T)
        logger.info(f"Setting eta to {self.eta} for {self.name}")


class FixedShareFactory:
    """Factory for properly initializing the fixed share algorithm"""

    def create(self, algo_config, global_config) -> FixedShare:
        algo_config = algo_config.get('FIXED_SHARE')
        return FixedShare(**{**global_config, **algo_config})
