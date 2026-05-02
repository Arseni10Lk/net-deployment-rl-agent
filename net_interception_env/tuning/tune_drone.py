import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.optimize import minimize
import net_interception_env
import gymnasium as gym
from stable_baselines3 import DQN
import json
import train_drone
import os
import csv

# 1. Define the model
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel)

# 1. Define bounds
bounds_list = [
    (3.0, 5.0),         # 0: the negative power of 10 for learning_rate
    (0.01, 0.1),        # 1: exploration_initial_eps
    (0.001, 0.01),      # 2: exploration_final_eps
    (0.1, 0.5),         # 3: exploration_fraction
    (5.0, 8.0),         # 4: the power of 2 for batch_size
    (0.99, 0.9999),     # 5: gamma
    (1000.0, 15000.0)   # 6: target_update_interval
]

chunks = 10

# 2. Sampler
def bayesian_sample(bounds, past_params, past_scores, best_score):

    if len(past_scores) < 5:
        return np.array([np.random.uniform(low, high) for low, high in bounds])

    scaler = StandardScaler()

    scaled_past_params = scaler.fit_transform(past_params)

    # Fit on scaled past data
    gp.fit(scaled_past_params, past_scores)

    def scaled_expected_improvement(x, gp, best_score, scaler, xi=0.01):
        # The optimizer guesses in the original space, so we must scale it
        scaled_x = scaler.transform(x.reshape(1, -1))
        prediction, uncertainty = gp.predict(scaled_x, return_std=True)

        if uncertainty == 0.0:
            return 0.0

        Z = (prediction - best_score - xi) / uncertainty
        ei = (prediction - best_score - xi) * norm.cdf(Z) + uncertainty * norm.pdf(Z)

        return -ei[0]

    best_x = None
    best_ei = float('inf')

    for _ in range(10):  # Multi-start 10 times
        starting_guess = np.array([np.random.uniform(low, high) for low, high in bounds])

        result = minimize(
            scaled_expected_improvement,
            x0=starting_guess,
            bounds=bounds_list,
            args=(gp, best_score, scaler),
            method='L-BFGS-B'
        )

        if result.fun < best_ei:
            best_ei = result.fun
            best_x = result.x

    return best_x

    return result.x

# 3. Objective Function
def evaluate_model(params, chunk_history, num_trials):

    env = gym.make("DroneNet-3D")
    model = DQN("MultiInputPolicy", env, **params)
    max_num_eval_episodes = 500
    num_eval_episodes = [250, 200, 150, 200, 200, 300, 300, 400, 400, 500]
    steps_total = 1e6
    steps_per_chunk = int(steps_total // chunks)
    pruned = False

    for chunk in range(chunks):
        memory = num_trials // (chunk + 1)

        model.learn(total_timesteps=steps_per_chunk, reset_num_timesteps=False)
        accuracy, score = train_drone.verify("DroneNet-3D", model, num_eval_episodes[chunk])

        if len(chunk_history[chunk]) >= memory:
            if score < min(chunk_history[chunk]):
                print(f"--> PRUNED! Trial killed at chunk {chunk + 1}/{chunks}. Score: {score}")
                pruned = True
                env.close()
                return accuracy, score, model, pruned
            else:
                print(f"Trial is at {chunk + 1}/{chunks}. Score: {score}")

        chunk_history[chunk].append(score)
        chunk_history[chunk] = sorted(chunk_history[chunk], reverse=True)[:memory]

    env.close()

    return accuracy, score, model, pruned

if __name__ == "__main__":

    num_trials = 40
    best_score = -float("inf")
    best_params = {}

    past_params_list = []
    past_scores_list = []
    chunk_history = {i: [] for i in range(chunks)}

    log_file = "tuning_log.csv"
    # Create the file and write the header if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Trial", "Status", "Score", "Accuracy", "Learning_Rate",
                "Init_Eps", "Final_Eps", "Eps_Fraction", "Batch_Size",
                "gamma", "target_update_interval"
            ])

    for trial in range(num_trials):
        raw_params = bayesian_sample(bounds_list, np.array(past_params_list), past_scores_list, best_score)

        # Convert the array back to a usable dictionary
        model_kwargs = {
            "learning_rate": 10**-raw_params[0],
            "exploration_initial_eps": raw_params[1],
            "exploration_final_eps": raw_params[2],
            "exploration_fraction": raw_params[3],
            "batch_size": int(2 ** round(raw_params[4])),
            "gamma": raw_params[5],
            "target_update_interval": int(round(raw_params[6])),
        }

        # Evaluate the chosen parameters
        accuracy, score, trained_model, pruned = evaluate_model(model_kwargs, chunk_history, num_trials)

        past_params_list.append(raw_params)
        past_scores_list.append(score)

        status = "Pruned" if pruned else "Completed"

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                trial + 1, status, score, accuracy,
                model_kwargs["learning_rate"],
                model_kwargs["exploration_initial_eps"],
                model_kwargs["exploration_final_eps"],
                model_kwargs["exploration_fraction"],
                model_kwargs["batch_size"],
                model_kwargs["gamma"],
                model_kwargs["target_update_interval"]
            ])

        if score > best_score:
            best_score = score
            best_params = model_kwargs

            print(f"New high score! Saving model with score: {best_score}, accuracy: {accuracy}")
            trained_model.save("best_drone_model")

            with open("best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)