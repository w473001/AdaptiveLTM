import numpy as np
from .algorithms import Algo
import logging
logger = logging.getLogger(__name__)


class AdaptLTM(Algo):

    def __init__(self, n_tasks, n_experts, alpha, rho, **kwargs):
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        n_actions = kwargs.get('n_actions')
        super().__init__(**kwargs)
        self.name = "ADAPTLTM"

        self.w_experts = np.zeros([self.n_tasks, self.n_experts])
        self.w = np.zeros([self.n_tasks, self.n_actions])  # action weights,

        self.alpha = alpha
        self.rho = rho
        self.phi = np.ones(self.n_experts) / self.n_experts

        self.sigma = np.zeros([self.n_experts])
        self.sigma_td = np.zeros([self.n_experts])  # now only needed for intermediate computation
        self.xi = np.ones([self.n_tasks, self.n_experts]) * (self.alpha ** 2)
        self.xi_td = np.ones([self.n_tasks, self.n_experts]) * self.alpha * (1 - self.alpha) / 2
        self.beta = np.zeros([self.n_tasks, self.n_experts, np.ceil(np.log2(self.total_trials)).astype(int)+1])
        self.beta_td = np.zeros([self.n_tasks, self.n_experts, np.ceil(np.log2(self.total_trials)).astype(int)+1])
        self.beta[:, :, 0] = self.alpha
        self.beta_td[:, :, 0] = (1 - self.alpha) / 2
        self.beta_temp = self.beta.copy()
        self.beta_td_temp = self.beta_td.copy()

        self.beta_2q_weights_ = np.array([2**q for q in range(self.beta.shape[-1])])
        # logger.info(f"beta_2q_weights_ = {self.beta_2q_weights_}")
        self.last_expert_predictions = np.zeros((n_experts, n_actions))
        self.epsilon = np.zeros(self.n_experts)
        self.z_ = 0
        self.z_hat_ = 0
        self.psi = np.zeros(self.n_experts)
        self.task_times = {i: 1 for i in range(self.n_tasks)}  # store local times for each task
        self.t = 0

        # Store expert weights
        self.phi_history = np.zeros((self.total_trials, self.n_experts))

    def copy_betas(self, task_number):
        self.beta_temp[task_number, :, :] = self.beta[task_number, :, :].copy()
        self.beta_td_temp[task_number, :, :] = self.beta_td[task_number, :, :].copy()

    def get_loss(self, loss_vector):
        return loss_vector[self.last_decision]

    def predict(self, task_number, expert_predictions):
        # store the current global  weight vector phi
        self.phi_history[self.t, :] = self.phi.copy()

        self.sigma[:] = (self.beta[task_number, :, :] * self.beta_2q_weights_).sum(axis=1)
        # logger.info(f"t = {self.t}, sigma = {self.sigma}")
        self.sigma_td[:] = (self.beta_td[task_number, :, :] * self.beta_2q_weights_).sum(axis=1)
        # logger.info(f"t = {self.t}, sigma_td = {self.sigma_td}")

        self.epsilon[:] = (self.sigma * self.phi) / ((self.sigma + self.sigma_td) + self.rho)
        # logger.info(f"t = {self.t}, epsilon = {self.epsilon}")
        self.z_ = self.epsilon.sum()
        # logger.info(f"t = {self.t}, z={self.z_}")
        self.w_experts[task_number, :] = self.epsilon / self.z_
        # logger.info(f"t = {self.t}, w_experts = {self.w_experts}")

        assert (expert_predictions.shape[0] == self.n_experts)
        assert (expert_predictions.shape[1] == self.n_actions)
        self.last_expert_predictions[:, :] = expert_predictions

        # convert expert weights to actions weights
        self.w[task_number, :] = self.w_experts[task_number, :]@expert_predictions
        # logger.info(f"t = {self.t}, w={self.w}")

        assert (np.isclose(self.w[task_number, :].sum(), 1))
        decision = np.random.choice(a=self.n_actions, p=self.w[task_number, :])

        # Store the action (arm) chosen, as we need it to store losses
        self.last_decision = decision
        # logger.info(f"t = {self.t}, last_decision={self.last_decision}")

        return decision

    def loss_update(self, loss_vector, task_number):
        bandit_loss_vector = np.zeros_like(loss_vector)
        bandit_loss_vector[self.last_decision] = loss_vector[self.last_decision] / self.w[
            task_number, self.last_decision]
        # logger.info(f"t = {self.t}, loss_vector={loss_vector}")
        # logger.info(f"t = {self.t}, bandit_loss_vector={bandit_loss_vector}")
        expert_loss_vector = self.last_expert_predictions @ bandit_loss_vector
        # logger.info(f"t = {self.t}, expert_loss_vector={expert_loss_vector}")

        self.z_hat_ = (self.epsilon * np.exp(-self.eta * expert_loss_vector)).sum()
        # logger.info(f"t = {self.t}, z_hat={self.z_hat_}")
        self.psi[:] = np.exp(-self.eta * expert_loss_vector) * self.z_ / self.z_hat_
        # logger.info(f"t = {self.t}, psi={self.psi}")
        self.phi *= (((self.psi * self.sigma) + self.sigma_td) + self.rho) / (self.sigma + self.sigma_td + self.rho)
        # logger.info(f"t = {self.t}, phi={self.phi}")
        assert(np.isclose(self.phi.sum(), 1))

    def switching_update(self, task_number):

        qt = self.get_qt(self.task_times[task_number])
        # logger.info(f"t = {self.t}, task_time={self.task_times[task_number]}, qt={qt}")

        self.xi_td[task_number, :] = self.alpha*self.beta_td[task_number, :, :].sum(axis=1)
        # logger.info(f"t = {self.t}, xi_td={self.xi_td}")
        self.xi[task_number, :] = self.alpha * self.psi * (self.beta[task_number, :, :].sum(axis=1))
        # logger.info(f"t = {self.t}, xi={self.xi}")

        # store previous vectors required for update
        beta_t_0 = self.beta[task_number, :, 0].copy()
        beta_t_1 = self.beta[task_number, :, 1].copy()
        beta_td_t_0 = self.beta_td[task_number, :, 0].copy()
        beta_td_t_1 = self.beta_td[task_number, :, 1].copy()
        beta_t_qt = self.beta[task_number, :, qt].copy()
        beta_t_qt1 = self.beta[task_number, :, qt + 1].copy()
        beta_td_t_qt = self.beta_td[task_number, :, qt].copy()
        beta_td_t_qt1 = self.beta_td[task_number, :, qt + 1].copy()

        self.copy_betas(task_number)  # copy this slice of beta/beta_td, update the temp versions then copy back
        self.beta_temp[task_number, :, :] *= self.psi.reshape((self.psi.shape[0], -1))

        if qt == 0:
            self.beta_temp[task_number, :, 0] = self.xi_td[task_number, :]
            self.beta_td_temp[task_number, :, 0] = self.xi[task_number, :]
            self.beta_temp[task_number, :, 1] = (1 - self.alpha) * self.psi * ((beta_t_0 / 2) + beta_t_1)
            self.beta_td_temp[task_number, :, 1] = (1 - self.alpha) * ((beta_td_t_0 / 2) + beta_td_t_1)
        else:
            self.beta_temp[task_number, :, 0] = (beta_t_0 *(1 - self.alpha) * self.psi) + self.xi_td[task_number, :]
            self.beta_td_temp[task_number, :, 0] = ((1 - self.alpha) * beta_td_t_0) + self.xi[task_number, :]

            self.beta_temp[task_number, :, qt] = 0
            self.beta_td_temp[task_number, :, qt] = 0

            self.beta_temp[task_number, :, qt+1] = ((1-self.alpha) * self.psi * beta_t_qt1) + ((2-self.alpha) * self.psi * beta_t_qt / 4)
            self.beta_td_temp[task_number, :, qt+1] = ((1-self.alpha) * beta_td_t_qt1) + ((2-self.alpha)*beta_td_t_qt / 4)

        # logger.info(f"t = {self.t}, beta={self.beta[task_number, :, :]}")
        # logger.info(f"t = {self.t}, beta_td={self.beta_td[task_number, :, :]}")

        self.beta[task_number, :, :] = self.beta_temp[task_number, :, :].copy()
        self.beta_td[task_number, :, :] = self.beta_td_temp[task_number, :, :].copy()

        self.task_times[task_number] += 1  # update local task time

        # if self.t % 100 == 0:
        #     self.log_dump()

    def log_dump(self):
        logger.info(f"t = {self.t}, epsilon = {self.epsilon}")
        logger.info(f"t = {self.t}, w_experts = {self.w_experts}")
        logger.info(f"t = {self.t}, w={self.w}")
        logger.info(f"t = {self.t}, last_decision={self.last_decision}")
        logger.info(f"t = {self.t}, z={self.z_}")
        logger.info(f"t = {self.t}, z_hat={self.z_hat_}")
        logger.info(f"t = {self.t}, psi={self.psi}")
        logger.info(f"t = {self.t}, phi={self.phi}")
        logger.info(f"t = {self.t}, task_times={self.task_times}")
        logger.info(f"t = {self.t}, sigma={self.sigma}")
        logger.info(f"t = {self.t}, sigma_td={self.sigma_td}")
        logger.info(f"t = {self.t}, beta={self.beta}")
        logger.info(f"t = {self.t}, beta_td={self.beta_td}")
        logger.info(f"t = {self.t}, xi={self.xi}")
        logger.info(f"t = {self.t}, xi_td={self.xi_td}")

    def get_qt(self, t):
        # O(log(T)) implementation
        found = False
        i = 0
        while not found:
            if (t + 1) % (2**i) != 0:
                found = True
            else:
                i += 1
        return i-1


class AdaptLTMFactory:
    """Factory for properly initializing the markov specialists algorithm"""

    def create(self, algo_config, global_config) -> Algo:
        algo_config = algo_config.get('ADAPTLTM')
        # add required parameters before creating object
        global_config_ = global_config.copy()
        global_config_['total_trials'] = global_config_.get('total_trials_global')
        return AdaptLTM(**{**global_config_, **algo_config})
