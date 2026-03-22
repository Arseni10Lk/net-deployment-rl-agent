import gymnasium as gym

from stable_baselines3 import DQN
import net_interception_env

env = gym.make("DroneNet-v0")

model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=100000, log_interval=100)
model.save("dqn_drone1D")

env.close()

print("Training finished! Booting up visual mode...")

# --- VISUAL WATCHING ---
visual_env = gym.make("DroneNet-v0", render_mode="human")
obs, info = visual_env.reset()

running = True
while running:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = visual_env.step(action)
    if terminated:
        running = False
        print("The target was successfully reached!")
    elif truncated:
        print("The simulation timed out")

visual_env.close()