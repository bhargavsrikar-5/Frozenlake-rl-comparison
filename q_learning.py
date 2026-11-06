"""Tabular Q-Learning for Gymnasium's discrete FrozenLake environment."""

import numpy as np


class QLearningAgent:
    """Learn state-action values with a fixed-epsilon greedy policy."""

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros(
            (env.observation_space.n, env.action_space.n), dtype=np.float64
        )

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, terminated):
        """Apply one Q-learning update, bootstrapping only if nonterminal.

        A time-limit truncation ends the episode loop, but still bootstraps from
        its final observation. A true terminal state does not bootstrap.
        """
        best_next_value = 0.0 if terminated else np.max(self.Q[next_state])
        target = reward + self.gamma * best_next_value
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def train(self, num_episodes=1000):
        rewards_per_episode = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update(state, action, reward, next_state, terminated)
                total_reward += reward
                state = next_state

            rewards_per_episode.append(total_reward)
        return rewards_per_episode
