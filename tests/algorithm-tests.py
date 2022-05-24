import unittest
import numpy as np
from utils.utils import load_config
from algorithms.multitask import MultiTaskAlgo
from algorithms.utils import get_algo_factory


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.config = load_config("test_config")

    def test_initialise_exp4(self):
        algo_config = self.config.get('local')
        global_config = self.config.get('global')

        # Use multi-task wrapper class
        algo = MultiTaskAlgo(algo_name="EXP4", algo_config=algo_config, global_config=global_config)
        # Get an actual fixed share instance
        exp4 = algo.instances[0]

        self.assertEqual(exp4.eta, 2.0)
        self.assertEqual(exp4.name, "Exp4")

    def test_initialise_fixed_share(self):
        algo_config = self.config.get('local')
        global_config = self.config.get('global')

        k = global_config.get('n_switches_per_epoch') * global_config.get('n_epochs')
        T = np.ceil(global_config.get('total_trials_global') / global_config.get('n_tasks'))

        # Use multi-task wrapper class
        algo = MultiTaskAlgo(algo_name="FIXED_SHARE", algo_config=algo_config, global_config=global_config)
        # Get an actual fixed share instance
        fs = algo.instances[0]

        expected_alpha = k / (T - 1)

        self.assertEqual(fs.alpha, expected_alpha)

    def test_fixed_share_update(self):
        algo_config = self.config.get('local')
        global_config = self.config.get('global')
        n_actions = global_config.get("n_actions")

        # Use multi-task wrapper class
        algo = MultiTaskAlgo(algo_name="FIXED_SHARE", algo_config=algo_config, global_config=global_config)

        # Get an actual fixed share instance
        fs = algo.instances[0]

        zero_loss_vector = np.zeros(n_actions)
        fs.update(zero_loss_vector)

        uniform_vector = np.ones(n_actions) / n_actions

        self.assertTrue(np.allclose(fs.w, uniform_vector))

    def test_adaptltm_prediction(self):
        algo_config = self.config.get('local')
        global_config = self.config.get('global')
        n_actions = global_config.get("n_actions")
        n_experts = global_config.get('n_experts')
        algo_factory = get_algo_factory()
        adaptltm = algo_factory.create(name="ADAPTLTM", algo_config=algo_config, global_config=global_config)

        good_action = 4  # put all the mass on this action for every expert

        expert_preds = np.zeros((n_experts, n_actions))
        expert_preds[:, good_action] = 1

        adaptltm_pred = adaptltm.predict(task_number=1, expert_predictions=expert_preds)

        self.assertEqual(adaptltm_pred, good_action)


if __name__ == '__main__':
    unittest.main()
