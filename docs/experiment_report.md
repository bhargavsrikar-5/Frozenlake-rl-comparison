# Experiment report

## Objective

Compare tabular Q-Learning and tabular Policy Gradient (REINFORCE) on the same stochastic FrozenLake task, preserving the project's original two-algorithm scope.

## Experiment configuration

| Setting | Value |
|---|---|
| Environment | Gymnasium `FrozenLake-v1` |
| Map | `8x8` |
| Slippery transitions | Enabled (`True`) |
| Reward | Default sparse reward: 1 at goal, 0 otherwise |
| Episodes | 1,000 per algorithm |
| Q-Learning | `alpha=0.1`, `gamma=0.99`, fixed `epsilon=0.1` |
| Tabular REINFORCE | `learning_rate=0.01`, `gamma=0.99` |

## Algorithms and training procedure

The agents were trained sequentially using the same environment configuration and reward definition. Q-Learning updates a tabular action-value function after each transition. Tabular REINFORCE samples from a softmax policy and updates its logits from normalized discounted returns after naturally terminated episodes. Externally truncated Policy Gradient episodes contribute to recorded episode rewards but are excluded from policy updates.

The source modules are [`q_learning.py`](../q_learning.py) and [`policy_gradient.py`](../policy_gradient.py); the saved comparison is [`notebooks/frozenlake_comparison.ipynb`](../notebooks/frozenlake_comparison.ipynb).

## Recorded results

The saved Colab output records one stochastic training run:

| Algorithm | Mean training reward |
|---|---:|
| Q-Learning | 0.0000 |
| Tabular Policy Gradient (REINFORCE) | 0.0010 |

These are training means, not results from a separate evaluation. The [comparison graph](../results/figures/training_comparison.png) is the PNG embedded in the saved notebook output. The same aggregate values and configuration are recorded in [`training_summary.csv`](../results/metrics/training_summary.csv).

## Observations

Both means are very close to zero under FrozenLake's sparse reward. The recorded run shows little goal-reaching reward, and the 0.0010 difference is not enough to conclude that one method is generally better. No per-episode reward series, seed, or multi-run summary is recorded in the repository.

## Limitations

- This report describes one stochastic Colab training run.
- There is no separate evaluation of a frozen learned policy and no repeated-seed study.
- The means do not provide a success rate, variance, confidence interval, or statistical significance; none is inferred here.
- A 10-episode Gymnasium smoke test was reported as passing in Colab. It checks compatibility, not algorithm performance.
- Historical research documents include conflicting experiment settings and results. Their claims are not combined with or substituted for the saved notebook outputs.
- The original hyperparameter-tuning attempts did not complete successfully. No tuning conclusion is claimed.

## Conclusion

The project preserves a compact Q-Learning versus tabular REINFORCE comparison. The available run records mean training rewards of 0.0000 and 0.0010. These very low, single-run observations do not establish comparative superiority or reproducible performance.
