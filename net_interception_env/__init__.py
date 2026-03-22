from gymnasium.envs.registration import register

# Register the environment
register(
    id="DroneNet-v0",
    entry_point="net_interception_env.envs.drone_net_env:DroneNetEnv", # filename:ClassName
    max_episode_steps=1000,
)