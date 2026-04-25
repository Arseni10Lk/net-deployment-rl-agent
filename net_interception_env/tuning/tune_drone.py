import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
import net_interception_env
import gymnasium as gym
from stable_baselines3 import DQN
import json
import train_drone

# 1. Define the model
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel)

# 1. Define bounds
bounds_list = [
    (1e-5, 1e-2),   # learning_rate
    (0.01, 1.0),    # exploration_initial_eps
    (0.001, 0.1),   # exploration_final_eps
    (0.1, 0.7),     # exploration_fraction
    (32.0, 128.0)   # batch_size (keep as float for the math, cast to int later)
]

# 2. Sampler
def bayesian_sample(bounds, past_params, past_scores, best_score):

    if len(past_scores) < 5:
        return np.array([np.random.uniform(low, high) for low, high in bounds])

    # Fit on past data
    gp.fit(past_params, past_scores)

    starting_guess = np.array([np.random.uniform(low, high) for low, high in bounds_list])

    result = minimize(
        expected_improvement,
        x0=starting_guess,
        bounds=bounds_list,
        args=(gp, best_score),
        method='L-BFGS-B'
    )

    return result.x

def expected_improvement(x, gp, best_score, xi=0.01):
    prediction, uncertainty = gp.predict(x.reshape(1, -1), return_std=True)

    if uncertainty == 0.0:
        return 0.0

    Z = (prediction - best_score - xi) / uncertainty
    ei = (prediction - best_score - xi) * norm.cdf(Z) + uncertainty * norm.pdf(Z)

    return -ei[0]  # Return negative because scipy minimizes


# 3. Objective Function
def evaluate_model(params):
    env = gym.make("DroneNet-3D")

    model = DQN("MultiInputPolicy", env, **params)

    model.learn(total_timesteps=50000)

    accuracy = train_drone.verify("DroneNet-3D", model)

    return accuracy, model

# 4. Main Tuner Loop
num_trials = 10
best_score = -float("inf")
best_params = {}

past_params_list = []
past_scores_list = []

for trial in range(num_trials):
    raw_params = bayesian_sample(bounds_list, np.array(past_params_list), past_scores_list, best_score)

    # Convert the array back to a usable dictionary
    model_kwargs = {
        "learning_rate": raw_params[0],
        "exploration_initial_eps": raw_params[1],
        "exploration_final_eps": raw_params[2],
        "exploration_fraction": raw_params[3],
        "batch_size": int(round(raw_params[4]))
    }

    # Evaluate the chosen parameters
    score, trained_model = evaluate_model(model_kwargs)

    past_params_list.append(raw_params)
    past_scores_list.append(score)

    if score > best_score:
        best_score = score
        best_params = model_kwargs

        print(f"New high score! Saving model with score: {best_score}")
        trained_model.save("best_drone_model")

        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)