The goal of the project is to train an RL model to choose an optimal moment for net deployment in the context of drone interception.

The model has a discrete output (to shoot or not to shoot), guidance is taken care of by a True Pro-Nav algorithm.

This environment is built using Gymnasium and trained using Stable-Baselines3.

## Installation

To install your new environment, run the following commands:

```{shell}
cd net_interception_env
pip install -e .
```

