import gymnasium as gym
import json

from gymnasium.envs import toy_text
from stable_baselines3 import DQN
import csv
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import net_interception_env

class BatchContinuationCallback(BaseCallback):
    def __init__(self, env_name: str, verbose=1, batches=10, total_timesteps=1e6):
        super().__init__(verbose)
        self.env = env_name
        self.verbose = verbose
        self.current_batch = 0
        self.timesteps_per_batch = total_timesteps / batches
        self.batches = batches

    def _on_step(self) -> bool:

        if self.n_calls % self.timesteps_per_batch == 0:
            self.current_batch += 1

            print(f"Verifying batch {self.current_batch}/{self.batches}...")
            self.model.save(f"training_model_checkpoints/dqn_drone3D_checkpoint_{self.current_batch}")
            self.accuracy, self.accuracy_score = verify("DroneNet-3D", self.model)

            record_verification(self.current_batch, self.accuracy, self.accuracy_score)

        return True

def verify(env_name, model, num_episodes=500):
    test_env = gym.make(env_name)
    success = 0
    timed_out = 0
    miss = 0

    for ep in range(num_episodes):
        obs, info = test_env.reset()
        running = True
        while running:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if terminated:
                running = False
                if reward > 0:
                    success += 1
                else:
                    miss += 1
            elif truncated:
                running = False
                timed_out += 1

    accuracy = (success / num_episodes) * 100
    miss_percentage = (miss / num_episodes) * 100
    accuracy_score = accuracy + miss_percentage / 10 # encourage trying

    print(f"Evaluation finished! Accuracy: {accuracy:.2f}% ({success}/{num_episodes})\n"
          f"Misses: {miss_percentage:.2f}% ({miss}/{num_episodes})\n"
          f"Timed out: {timed_out}/{num_episodes}\n"
          f"Score: {accuracy_score:.2f}")

    test_env.close()

    return accuracy, accuracy_score

def record_verification(batch, accuracy, accuracy_score):
    with open("verification.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([batch + 1, accuracy, accuracy_score])

if __name__ == "__main__":
    env_name = "DroneNet-3D"
    env = make_vec_env(env_name, n_envs=8, vec_env_cls=SubprocVecEnv)

    with open("net_interception_env/tuning/best_params.json", "r") as json_file:
        params = json.load(json_file)

    model = DQN("MultiInputPolicy", env, device="cuda", verbose=1, **params)

    batches = 25
    timesteps = int(5e6)

    batch_callback = BatchContinuationCallback(
        env_name, verbose = 1, batches = batches, total_timesteps = timesteps
    )

    model.learn(
        total_timesteps=timesteps,
        log_interval=100,
        callback=batch_callback,
    )

    model.save("dqn_drone3D_final")

    env.close()