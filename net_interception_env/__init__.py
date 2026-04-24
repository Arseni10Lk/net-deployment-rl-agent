from gymnasium.envs.registration import register

# Register the environment
register(
    id="DroneNet-1D",
    entry_point="net_interception_env.envs.drone_net_env_1D:DroneNetEnv", # filename:ClassName
    max_episode_steps=1000,
)

register(
    id="DroneNet-3D",
    entry_point="net_interception_env.envs.drone_net_env_3D:DroneNetEnv", # filename:ClassName
    max_episode_steps=1000,
)

register(
    id="DroneNet-3D-no-net",
    entry_point="net_interception_env.envs.drone_net_env_3D_no_net:DroneNetEnv", # filename:ClassName
    max_episode_steps=1000,
)