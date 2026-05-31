import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.optimize import minimize
from stable_baselines3.common.callbacks import BaseCallback
from torch.jit import last_executed_optimized_graph
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

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

chunks = 5
steps_total = 1e6
num_trials = 45

# 1. Define bounds
bounds_list = [
    (3.0, 4.5),             # 0: the negative power of 10 for learning_rate
    (0.5, 2.0),             # 1: negative power of 10 for exploration_initial_eps
    (2.0, 4.0),             # 2: negative power of 10 for exploration_final_eps
    (0.1, 1.0),             # 3: exploration_fraction
    (3.0, 4.0),             # 4: negative power of 10 for 1-gamma
    (0.3, 1.5)              # 5: target_update_interval / 1e4
]

class PruningCallback(BaseCallback):
    def __init__(self,
                env_name: str,
                verbose=0,
                chunks=chunks,
                num_eval_episodes=500,
                chunk_history=None,
                threshold_fraction=1):
        super().__init__(verbose)
        self.env = env_name
        self.verbose = verbose
        self.current_chunk = 0
        self.timesteps_per_chunk = steps_total / chunks
        self.num_eval_episodes = num_eval_episodes

        if chunk_history is None:
            self.chunk_history = {i: [] for i in range(chunks)}
        else:
            self.chunk_history = chunk_history

        self.scores = []

        self.pruned = False
        self.threshold_fraction = threshold_fraction
        self.accuracy = 0

    def _on_step(self) -> bool:

        if self.n_calls % self.timesteps_per_chunk == 0:

            self.current_chunk += 1

            memory = num_trials // (self.current_chunk + 1)

            print(f"Verifying chunk {self.current_chunk}/{chunks}...")
            self.model.save(f"training_model_checkpoints/dqn_drone3D_checkpoint_{self.current_chunk}")
            self.accuracy, score = train_drone.verify("DroneNet-3D", self.model, self.num_eval_episodes)

            if len(self.chunk_history[self.current_chunk - 1]) >= memory:
                threshold = min(self.chunk_history[self.current_chunk - 1])  * 0.0

                if score < threshold:
                    print(f"--> PRUNED! Trial killed at chunk {self.current_chunk}/{chunks}. Score: {score}")
                    self.pruned = True

            self.scores.append(score)
            self.chunk_history[self.current_chunk - 1].append(score)
            self.chunk_history[self.current_chunk - 1] = sorted(self.chunk_history[self.current_chunk - 1], reverse=True)[:memory]

            return False if self.pruned else True

        return True

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

# 3. Objective Function
def evaluate_model(params, chunk_history):

    env_name = "DroneNet-3D"
    # env = make_vec_env(env_name, n_envs=8, vec_env_cls=SubprocVecEnv)
    env = gym.make(env_name)
    model = DQN("MultiInputPolicy", env, **params, device="cpu")
    num_eval_episodes = 500

    my_pruning_callback = PruningCallback(
        env_name, verbose=0, chunks=chunks, num_eval_episodes=num_eval_episodes, chunk_history=chunk_history, threshold_fraction=0
    )

    model.learn(total_timesteps=steps_total, callback=my_pruning_callback)
    env.close()

    accuracy = my_pruning_callback.accuracy
    pruned = my_pruning_callback.pruned
    scores = my_pruning_callback.scores

    return accuracy, scores, model, pruned

def load_params_from_the_last_run(filename, number=5):
    with open(filename, mode='r', newline='') as tuning_log:
        csvreader = csv.reader(tuning_log)
        next(csvreader)

        # check scores
        scores = []

        for row in csvreader:
            if row[2].strip() != "":
                scores.append(float(row[2]))

        scores = sorted(scores, reverse=True)

        # extract hyperparameters

        tuning_log.seek(0)
        next(csvreader)
        hyperparameters = []

        for row in csvreader:
            if row[2].strip() != "":
                if scores[number] < float(row[2]):
                    hyperparameters.append(row[4:8] + row[9:11])

    return [[float(val) for val in row] for row in hyperparameters]

def pre_trials(log_file_old, log_file_new, number=5):
    last_run_params = load_params_from_the_last_run(log_file_old, number)
    past_params_list = []
    past_scores_list = []

    best_score = -float("inf")

    for pre_trial in range(number):

        model_kwargs = {
            "learning_rate": last_run_params[pre_trial][0],
            "exploration_initial_eps": last_run_params[pre_trial][1],
            "exploration_final_eps": last_run_params[pre_trial][2],
            "exploration_fraction": last_run_params[pre_trial][3],
            "batch_size": 256,
            "gamma": last_run_params[pre_trial][4],
            "target_update_interval": int(last_run_params[pre_trial][5]),
        }

        accuracy, scores, trained_model, pruned = evaluate_model(model_kwargs, chunk_history)
        final_score = scores[-1]

        raw_params = model_to_raw_params(last_run_params[pre_trial])
        past_params_list.append(raw_params)
        past_scores_list.append(final_score)

        status = "Pruned" if pruned else "Completed"

        with open(log_file_new, mode='a', newline='') as file:
            all_col_scores = scores + [""] * (chunks - len(scores))

            writer = csv.writer(file)
            writer.writerow([
                pre_trial + 1, status, final_score, accuracy,
                model_kwargs["learning_rate"],
                model_kwargs["exploration_initial_eps"],
                model_kwargs["exploration_final_eps"],
                model_kwargs["exploration_fraction"],
                model_kwargs["gamma"],
                model_kwargs["target_update_interval"]]
                + all_col_scores
            )

        if final_score > best_score:
            best_score = final_score
            best_params = model_kwargs

            print(f"New high score! Saving model with score: {best_score}, accuracy: {accuracy}")
            trained_model.save("best_drone_model")

            with open("best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)

    return past_params_list, past_scores_list, best_score

def load_past_trials(log_file):
    past_params_list = []
    past_scores_list = []
    best_score = -float("inf")
    chunk_history = {i: [] for i in range(chunks)}
    with open(log_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            past_scores_list.append(float(row[2]))
            best_score = max(best_score, float(row[2]))
            model_params = row[4:10]
            past_params_list.append(model_to_raw_params(model_params))

            for i in range(chunks):
                if row[10+i] != "":
                    chunk_history[i].append(float(row[10+i]))

    for chunk in range(chunks):
        memory = num_trials // (chunk + 2)
        chunk_history[chunk] = sorted(chunk_history[chunk], reverse=True)[:memory]

    return past_params_list, past_scores_list, best_score, chunk_history

def model_to_raw_params(model_params):
    model_params = [float(p) for p in model_params]

    raw_params = [
        - np.log10(model_params[0]),
        - np.log10(model_params[1]),
        - np.log10(model_params[2]),
        model_params[3],
        - np.log10(1 - model_params[4]),
        model_params[5] / 1e4
    ]
    return raw_params

if __name__ == "__main__":

    best_score = -float("inf")
    best_params = {}

    past_params_list = []
    past_scores_list = []
    chunk_history = {i: [] for i in range(chunks)}

    log_file = "tuning_log_v3.csv"
    # Create the file and write the header if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            chunks_header = []
            for chunk in range(chunks):
                chunks_header.append(f"After {int(steps_total // chunks * (chunk+1))}")

            writer.writerow([
                "Trial", "Status", "Final Score", "Accuracy", "Learning_Rate",
                "Init_Eps", "Final_Eps", "Eps_Fraction",
                "gamma", "target_update_interval"] + chunks_header
                )

        # num_pre_trials = 4
        # past_params_list, past_scores_list, best_score = pre_trials('tuning_log_v1.csv', log_file, num_pre_trials)

    else:
        past_params_list, past_scores_list, best_score, chunk_history = load_past_trials(log_file)

    for trial in range(num_trials - len(past_params_list)):
        raw_params = bayesian_sample(bounds_list, np.array(past_params_list), past_scores_list, best_score)

        # Convert the array back to a usable dictionary
        model_kwargs = {
            "learning_rate":            10**-raw_params[0],
            "exploration_initial_eps":  10**-raw_params[1],
            "exploration_final_eps":    10**-raw_params[2],
            "exploration_fraction":     raw_params[3],
            "batch_size":               256,
            "gamma":                    1 - 10**-raw_params[4],
            "target_update_interval":   int(round(1e4 * raw_params[5])),
        }

        # Evaluate the chosen parameters
        accuracy, scores, trained_model, pruned = evaluate_model(model_kwargs, chunk_history)
        final_score = scores[-1]

        status = "Pruned" if pruned else "Completed"

        current_trial_num = len(past_params_list) + 1
        trained_model.save(f'models/model_{current_trial_num}')

        past_params_list.append(raw_params)
        past_scores_list.append(final_score)

        with open(log_file, mode='a', newline='') as file:
            all_col_scores = scores + [""] * (chunks - len(scores))
            writer = csv.writer(file)
            writer.writerow([len(past_params_list), status, final_score, accuracy,
                                model_kwargs["learning_rate"],
                                model_kwargs["exploration_initial_eps"],
                                model_kwargs["exploration_final_eps"],
                                model_kwargs["exploration_fraction"],
                                model_kwargs["gamma"],
                                model_kwargs["target_update_interval"]] +
                                all_col_scores)

        if final_score > best_score:
            best_score = final_score
            best_params = model_kwargs

            print(f"New high score! Score: {best_score}, accuracy: {accuracy}")

            with open("best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)