import gymnasium as gym

from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
import net_interception_env

env = gym.make("DroneNet-3D")

model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=1e-3, exploration_initial_eps=0.05, exploration_final_eps=0.01, exploration_fraction=0.5)
model.learn(total_timesteps=100000, log_interval=100)
model.save("dqn_drone3D")

env.close()

# RESULT VERIFICATION
test_env = gym.make("DroneNet-3D")
success = 0
num_episodes = 100

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
        elif truncated:
            running = False

accuracy = (success / num_episodes) * 100
print(f"Evaluation finished! Accuracy: {accuracy:.2f}% ({success}/{num_episodes})")

# --- VISUAL WATCHING ---
visual_env = gym.make("DroneNet-3D", render_mode="human")
obs, info = visual_env.reset()

running = True
while running:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = visual_env.step(action)
    if terminated:
        running = False
        if reward > 0:
            print("The target was successfully reached!")
        else:
            print("The target was missed!")
    elif truncated:
        running = False
        print("The simulation timed out")

visual_env.close()