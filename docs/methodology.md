# Methodology

## Environment

The comparison uses Gymnasium `FrozenLake-v1` with `map_name="8x8"` and `is_slippery=True`. The grid has 64 discrete states and four discrete actions. Slippery transitions make the resulting state stochastic relative to the selected action.

The paired experiment uses the environment's default sparse reward: reaching the goal yields 1; other transitions yield 0. No custom reward shaping is applied.

## Q-Learning

The Q-table has one row per state and one column per action. The agent selects a random action with fixed probability `epsilon=0.1`; otherwise it chooses the action with the largest current Q-value. The update is:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

For a naturally terminated transition, the next-state value is masked to zero. A Gymnasium external truncation ends the episode loop, but the Q update bootstraps from the final observation because the truncation is not a natural terminal state.

## Tabular Policy Gradient (REINFORCE)

The policy has a trainable logit for each state-action pair. A numerically stabilized row-wise softmax converts each state's logits into categorical action probabilities. The agent samples actions from those probabilities.

For a naturally terminated trajectory, discounted reward-to-go is calculated backward:

```text
G_t = r_t + gamma * G_(t+1)
```

The trajectory returns are normalized within the episode, as in the original paired notebook approach. The score-function gradient for the selected action is `one_hot(action) - policy(state)`. The implementation sums the trajectory gradients and applies the update to the policy logits after the episode.

There is no value-function critic. If Gymnasium ends an episode through external truncation before natural termination, the episode's reward is still recorded, but its incomplete trajectory is excluded from the REINFORCE update. This avoids treating a truncated trajectory as a complete Monte Carlo return without a value estimate.

## Gymnasium episode API

Both agents unpack `reset()` as `(observation, info)` and `step()` as `(observation, reward, terminated, truncated, info)`. Training ends when either flag is true. The algorithms handle the resulting updates differently: Q-Learning bootstraps across truncation, while REINFORCE updates only on natural termination.

## Experiment configuration

| Setting | Value |
|---|---|
| Environment | `FrozenLake-v1` |
| Map | `8x8` |
| Slippery | `True` |
| Reward | Default sparse FrozenLake reward |
| Episodes per algorithm | 1,000 |
| Q-Learning learning rate (`alpha`) | 0.1 |
| Q-Learning exploration (`epsilon`) | 0.1, fixed |
| Policy Gradient learning rate | 0.01 |
| Discount factor (`gamma`) | 0.99 for both |

The configuration is read from the paired comparison notebook and source modules. The recorded means describe one Colab run; see the [experiment report](experiment_report.md).
