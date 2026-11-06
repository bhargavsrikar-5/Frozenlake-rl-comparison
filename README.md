# Q-Learning vs Policy Gradient on Stochastic FrozenLake

## Project objective

This project compares tabular Q-Learning with a tabular Policy Gradient method on the stochastic FrozenLake task. The individual implementation and repository maintenance are by **Chegu Bhargav Srikar**.

## FrozenLake environment

The comparison uses Gymnasium's `FrozenLake-v1` environment with `map_name="8x8"` and `is_slippery=True`. Both agents use the environment's original sparse reward: reaching the goal gives reward 1; other transitions give reward 0. No custom reward shaping is used in the paired comparison.

## Q-Learning methodology

Q-Learning stores one value for each state-action pair and applies the one-step Bellman update. It selects actions with the original fixed-epsilon greedy approach (`epsilon=0.1`): random actions are chosen with probability epsilon, otherwise it selects the action with the highest current value. A terminal transition does not bootstrap; an external time-limit truncation ends the episode while Q-Learning bootstraps from the final observation.

## Tabular Policy Gradient methodology

The policy is categorical and tabular, with one set of action logits per state. A softmax converts each row to action probabilities. After a complete, naturally terminated episode, discounted returns are normalized as in the original notebook, and the REINFORCE score-function gradient updates the logits. Episodes that end through external truncation are still included in the recorded episode rewards, but are excluded from policy updates because their trajectories are incomplete. This is tabular REINFORCE and does not use a neural network or a value-function critic.

## Existing experiment configuration

The paired notebook uses 1,000 training episodes for each agent, learning rate 0.1 for Q-Learning, learning rate 0.01 for Policy Gradient, and discount factor 0.99 for both. Both agents receive the same environment configuration and default reward. The notebook plots the per-episode training rewards; it does not perform a separate evaluation protocol or repeated-seed study.

## Recorded training run

One completed Colab run used the configuration above and produced these mean training rewards:

| Agent | Mean training reward |
|---|---:|
| Q-Learning | 0.0000 |
| Tabular Policy Gradient | 0.0010 |

These are the observed values from one stochastic training run, not a repeated-seed benchmark or a separate evaluation. They do not establish that either algorithm is generally better. The saved notebook includes the corresponding training-reward plot.

## Installation and execution

From this directory, install the listed dependencies and open the notebook:

```bash
python -m pip install -r requirements.txt
jupyter notebook frozenlake_comparison.ipynb
```

Run the notebook cells to train both agents and display their training-reward curves. The notebook uses `q_learning.py` and `policy_gradient.py` rather than duplicating their implementations.

## Limitations

The original project does not establish reproducible comparative performance. Results depend on random environment transitions and action selection; the notebook does not set seeds, repeat runs, or evaluate trained policies separately. Episode rewards during training are not an independent test set. The recorded values above describe one completed run only; historical research-document claims are not treated as verified results. The original hyperparameter-tuning attempts did not complete successfully and are not claimed as successful.
