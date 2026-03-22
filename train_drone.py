import gymnasium as gym

from stable_baselines3 import DQN
import net_interception_env

env = gym.make("DroneNet-v0")

model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_drone1D")

env.close()

print("Training finished! Booting up visual mode...")

# --- VISUAL WATCHING ---
visual_env = gym.make("DroneNet-v0", render_mode="human")
obs, info = visual_env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = visual_env.step(action)
    if terminated or truncated:
        obs, info = visual_env.reset()
