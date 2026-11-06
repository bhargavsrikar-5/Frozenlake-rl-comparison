"""Tabular REINFORCE for Gymnasium's discrete FrozenLake environment."""

import numpy as np


class PolicyGradientAgent:
    """Optimize a tabular categorical policy using REINFORCE.

    The policy is represented by one set of logits per state. Action
    probabilities are the softmax of those logits.
    """

    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.lr = learning_rate
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.policy_logits = np.zeros(
            (self.num_states, self.num_actions), dtype=np.float64
        )
        self.policy = np.full(
            (self.num_states, self.num_actions),
            1.0 / self.num_actions,
            dtype=np.float64,
        )

    def _refresh_policy(self, state):
        logits = self.policy_logits[state]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        self.policy[state] = probabilities / np.sum(probabilities)

    def select_action(self, state):
        return int(np.random.choice(self.num_actions, p=self.policy[state]))

    def update_policy(self, trajectory):
        """Apply one episode's discounted-return policy-gradient update."""
        if not trajectory:
            return

        returns = np.empty(len(trajectory), dtype=np.float64)
        running_return = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            running_return = trajectory[index][2] + self.gamma * running_return
            returns[index] = running_return

        # Preserve the original implementation's per-episode return
        # normalization while applying the score-function gradient to logits.
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)
        gradients = np.zeros_like(self.policy_logits)

        # Use the behavior policy for every score term in this episode, then
        # update logits once after accumulating the trajectory gradient.
        for (state, action, _), discounted_return in zip(trajectory, returns):
            probabilities = self.policy[state]
            score = -probabilities.copy()
            score[action] += 1.0
            gradients[state] += discounted_return * score

        self.policy_logits += self.lr * gradients
        for state, _, _ in trajectory:
            self._refresh_policy(state)

    def train(self, num_episodes=1000):
        rewards_per_episode = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            trajectory = []
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                total_reward += reward
                state = next_state

            if terminated:
                self.update_policy(trajectory)
            rewards_per_episode.append(total_reward)
        return rewards_per_episode
