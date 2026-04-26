import gymnasium as gym
import json
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
import net_interception_env

def verify(env_name, model):
    test_env = gym.make(env_name)
    success = 0
    timed_out = 0
    miss = 0
    num_episodes = 500

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
    print(f"Evaluation finished! Accuracy: {accuracy:.2f}% ({success}/{num_episodes})\n"
          f"Misses: {miss_percentage:.2f}% ({miss}/{num_episodes})\n"
          f"Timed out: {timed_out}/{num_episodes}")
    accuracy_score = accuracy + miss_percentage / 10 # encourage trying

    return accuracy, accuracy_score

if __name__ == "__main__":
    env = gym.make("DroneNet-3D")

    with open("net_interception_env/tuning/best_params.json", "r") as json_file:
        params = json.load(json_file)

    model = DQN("MultiInputPolicy", env, verbose=1, **params)

    batches = 10
    timesteps = 2000000
    timesteps_per_batch = int(timesteps/batches)
    for batch in range(batches):
        print(f"Starting batch {batch + 1}/{batches}...")
        model.learn(
            total_timesteps=timesteps_per_batch,
            log_interval=50,
            reset_num_timesteps=False
        )
        model.save(f"dqn_drone3D_checkpoint_{batch + 1}")
        verify("DroneNet-3D", model)

    model.save("dqn_drone3D_final")

    env.close()

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