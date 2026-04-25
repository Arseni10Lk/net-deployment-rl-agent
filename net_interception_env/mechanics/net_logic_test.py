import gymnasium as gym
import net_interception_env  # Ensures the environment is registered
import time
import numpy as np
# Initialize the environment with human rendering
env = gym.make("DroneNet-3D", render_mode="human")
obs, info = env.reset()
# Hack the internal state for a perfect side-profile shot
env.unwrapped.pursuer_location = np.array([100.0, 256.0, 256.0], dtype=np.float32)
env.unwrapped.pursuer_velocity = np.array([10.0, 0.0, 0.0], dtype=np.float32)
env.unwrapped.target_location = np.array([400.0, 256.0, 256.0], dtype=np.float32)
env.unwrapped.target_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
# 1 corresponds to Actions.do_shoot
action = 1
running = True
print("Firing net...")
while running:
    obs, reward, terminated, truncated, info = env.step(action)
    # Switch to 'dont_shoot' after the first frame so we don't overwrite the net
    action = 0
    # Slow down the loop so your eyes can track the PyGame rendering
    time.sleep(0.05)
    if terminated or truncated:
        print(f"Simulation ended. Final Reward: {reward}")
        running = False
time.sleep(2)  # Pause before closing the window
env.close()