import gymnasium as gym

from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
import net_interception_env

env = gym.make("DroneNet-3D")

model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=1e-3, exploration_initial_eps=0.05, exploration_final_eps=0.01, exploration_fraction=0.5)
model.learn(total_timesteps=100000, log_interval=100)
model.save("dqn_drone3D")

env.close()

print("Training finished! Booting up visual mode...")

# --- VISUAL WATCHING ---
visual_env = gym.make("DroneNet-3D", render_mode="human")
obs, info = visual_env.reset()

running = True
while running:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = visual_env.step(action)
    if terminated:
        running = False
        print("The target was successfully reached!")
    elif truncated:
        running = False
        print("The simulation timed out")

visual_env.close()