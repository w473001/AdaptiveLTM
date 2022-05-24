import unittest
import numpy as np
from utils.utils import load_config, get_task_epoch_matrix
from src.backtest.backtest import Task


class TestTaskMatrix(unittest.TestCase):
    def setUp(self):
        self.config = load_config("test_config")
        global_config = self.config.get('global')

        self.n_tasks = global_config.get('n_tasks')
        self.total_trials_global = global_config.get('total_trials_global')
        self.super_pool_size = global_config.get('super_pool_size')

        self.epoch_pool_size = global_config.get('epoch_pool_size')
        self.epoch_block_pool_size = global_config.get('epoch_block_pool_size')
        self.n_epochs = global_config.get('n_epochs')
        self.n_switches_per_epoch = global_config.get('n_switches_per_epoch')
        self.random_switches_within_epoch = global_config.get('random_switches_within_epoch')

        self.super_pool = [i for i in range(self.super_pool_size)]

    def test_task_matrix_correct_shape(self):
        task_matrix = get_task_epoch_matrix(n_tasks=self.n_tasks,
                                            T=self.total_trials_global,
                                            super_pool_size=self.super_pool_size,
                                            epoch_pool_size=self.epoch_pool_size,
                                            epoch_block_pool_size=self.epoch_block_pool_size,
                                            n_epochs=self.n_epochs,
                                            n_switches_per_epoch=self.n_switches_per_epoch,
                                            random=self.random_switches_within_epoch)

        self.assertEqual(task_matrix.shape[0], self.total_trials_global)
        self.assertEqual(task_matrix.shape[1], self.n_tasks)

    def test_single_task_matrix_correct_shape(self):
        task_matrix = get_task_epoch_matrix(n_tasks=1,
                                            T=self.total_trials_global,
                                            super_pool_size=self.super_pool_size,
                                            epoch_pool_size=self.epoch_pool_size,
                                            epoch_block_pool_size=self.epoch_block_pool_size,
                                            n_epochs=self.n_epochs,
                                            n_switches_per_epoch=self.n_switches_per_epoch,
                                            random=self.random_switches_within_epoch)
        self.assertEqual(task_matrix.shape[0], self.total_trials_global)
        self.assertEqual(task_matrix.shape[1], 1)

    def test_n_switches(self):
        task_matrix = get_task_epoch_matrix(n_tasks=self.n_tasks,
                                            T=self.total_trials_global,
                                            super_pool_size=self.super_pool_size,
                                            epoch_pool_size=self.epoch_pool_size,
                                            epoch_block_pool_size=self.epoch_block_pool_size,
                                            n_epochs=self.n_epochs,
                                            n_switches_per_epoch=self.n_switches_per_epoch,
                                            random=self.random_switches_within_epoch)

        diffs = task_matrix.diff() != 0

        # first trials of all epochs get counted as a switch
        expected_switches = (self.n_switches_per_epoch + 1) * self.n_epochs

        self.assertEqual(diffs.loc[:, 0].sum(), expected_switches)


class TestTaskObject(unittest.TestCase):
    def setUp(self):
        self.config = load_config("test_config")
        global_config = self.config.get('global')

        self.n_tasks = global_config.get('n_tasks')
        self.n_actions = global_config.get('n_actions')
        self.n_experts = global_config.get('n_experts')
        self.good_loss_upper_bound = global_config.get('good_loss_upper_bound')
        self.bad_loss_upper_bound = global_config.get('bad_loss_upper_bound')
        self.total_trials = global_config.get('total_trials_global')
        self.epsilon = global_config.get('epsilon')

    def test_expert_predictions_zero_epsilon(self):
        task_object = Task(n_actions=self.n_actions,
                           n_experts=self.n_experts,
                           good_loss_upper_bound=self.good_loss_upper_bound,
                           bad_loss_upper_bound=self.bad_loss_upper_bound,
                           total_trials=self.total_trials)

        good_expert = 2
        good_action = 4
        task_object.set_good_expert(index=good_expert)
        task_object.set_good_index(index=good_action)

        preds = task_object.get_expert_predictions(epsilon=0)

        self.assertEqual(preds[good_expert, good_action], 1)

    def test_expert_predictions_non_zero_epsilon(self):
        task_object = Task(n_actions=self.n_actions,
                           n_experts=self.n_experts,
                           good_loss_upper_bound=self.good_loss_upper_bound,
                           bad_loss_upper_bound=self.bad_loss_upper_bound,
                           total_trials=self.total_trials)

        good_expert = 2
        good_action = 4
        task_object.set_good_expert(index=good_expert)
        task_object.set_good_index(index=good_action)

        preds = task_object.get_expert_predictions(epsilon=self.epsilon)

        self.assertEqual(preds[good_expert, good_action], 1-self.epsilon)

    def test_expert_predictions_are_distributions(self):
        task_object = Task(n_actions=self.n_actions,
                           n_experts=self.n_experts,
                           good_loss_upper_bound=self.good_loss_upper_bound,
                           bad_loss_upper_bound=self.bad_loss_upper_bound,
                           total_trials=self.total_trials)

        good_expert = 2
        good_action = 4
        task_object.set_good_expert(index=good_expert)
        task_object.set_good_index(index=good_action)

        preds = task_object.get_expert_predictions(epsilon=self.epsilon)

        self.assertTrue(np.allclose(preds.sum(axis=1), np.ones(self.n_experts)))


if __name__ == '__main__':
    unittest.main()
