import gymnasium as gym
from stable_baselines3 import DQN

if __name__ == '__main__':
    visual_env = gym.make("DroneNet-3D", render_mode="human")

    model = DQN.load("dqn_drone3D_final.zip")

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