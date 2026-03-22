from gymnasium.envs.registration import register

# Register the environment
register(
    id="DroneNet-v0",
    entry_point="drone_net_env:DroneNetEnv", # filename:ClassName
    max_episode_steps=1000,
)