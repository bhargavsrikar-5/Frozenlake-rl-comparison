import unittest
from unittest.mock import patch

import numpy as np

from policy_gradient import PolicyGradientAgent
from q_learning import QLearningAgent


class FakeSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return 0


class OneStepEnv:
    def __init__(self, terminated, truncated, reward=0.0):
        self.observation_space = FakeSpace(4)
        self.action_space = FakeSpace(2)
        self.terminated = terminated
        self.truncated = truncated
        self.reward = reward
        self.steps = 0

    def reset(self):
        self.steps = 0
        return 0, {}

    def step(self, action):
        self.steps += 1
        return 1, self.reward, self.terminated, self.truncated, {}


class QLearningTests(unittest.TestCase):
    def test_q_table_dimensions(self):
        agent = QLearningAgent(OneStepEnv(False, False))
        self.assertEqual(agent.Q.shape, (4, 2))

    def test_terminal_update_does_not_bootstrap(self):
        agent = QLearningAgent(OneStepEnv(False, False), alpha=1.0, gamma=0.5)
        agent.Q[1] = [10.0, 8.0]
        agent.update(0, 0, 1.0, 1, terminated=True)
        self.assertEqual(agent.Q[0, 0], 1.0)

    def test_nonterminal_update_bootstraps(self):
        agent = QLearningAgent(OneStepEnv(False, False), alpha=1.0, gamma=0.5)
        agent.Q[1] = [2.0, 4.0]
        agent.update(0, 0, 1.0, 1, terminated=False)
        self.assertEqual(agent.Q[0, 0], 3.0)

    def test_truncation_ends_episode_and_bootstraps(self):
        env = OneStepEnv(terminated=False, truncated=True, reward=0.25)
        agent = QLearningAgent(env, alpha=1.0, gamma=0.5, epsilon=0.0)
        agent.Q[1] = [2.0, 4.0]
        agent.select_action = lambda state: 0

        rewards = agent.train(num_episodes=1)

        self.assertEqual(env.steps, 1)
        self.assertEqual(rewards, [0.25])
        self.assertEqual(agent.Q[0, 0], 2.25)


class PolicyGradientTests(unittest.TestCase):
    def test_policy_probabilities_are_valid_and_softmax_is_stable(self):
        agent = PolicyGradientAgent(OneStepEnv(False, False))
        np.testing.assert_allclose(agent.policy.sum(axis=1), 1.0)
        self.assertTrue(np.all(agent.policy >= 0.0))

        agent.policy_logits[0] = [1000.0, 999.0]
        agent._refresh_policy(0)
        self.assertTrue(np.all(np.isfinite(agent.policy[0])))
        self.assertTrue(np.all(agent.policy[0] >= 0.0))
        self.assertAlmostEqual(float(agent.policy[0].sum()), 1.0)

    def test_reinforce_score_update_increases_relative_action_logit(self):
        agent = PolicyGradientAgent(OneStepEnv(False, False), learning_rate=0.01)
        agent.update_policy([(0, 1, 1.0), (0, 0, 0.0)])

        self.assertGreater(agent.policy_logits[0, 1], agent.policy_logits[0, 0])
        np.testing.assert_allclose(agent.policy.sum(axis=1), 1.0)
        self.assertTrue(np.all(agent.policy >= 0.0))

    def test_natural_termination_updates_and_records_reward(self):
        env = OneStepEnv(terminated=True, truncated=False, reward=1.0)
        agent = PolicyGradientAgent(env)

        with patch.object(agent, "update_policy", wraps=agent.update_policy) as update:
            rewards = agent.train(num_episodes=1)

        update.assert_called_once()
        self.assertEqual(rewards, [1.0])

    def test_truncation_skips_update_and_records_reward(self):
        env = OneStepEnv(terminated=False, truncated=True, reward=0.25)
        agent = PolicyGradientAgent(env)

        with patch.object(agent, "update_policy", wraps=agent.update_policy) as update:
            rewards = agent.train(num_episodes=1)

        update.assert_not_called()
        self.assertEqual(rewards, [0.25])


if __name__ == "__main__":
    unittest.main()
