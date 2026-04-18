from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from net_interception_env.mechanics import Pro_Nav_logic as tpn, Constraints


class Actions(Enum):
    dont_shoot = 0
    do_shoot = 1

class DroneNetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, marker_size=5, size=512, max_steps=1000):
        self.size = size  # The size of the environment
        self.marker_size = marker_size # The size of the markers
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "distance": spaces.Box(0, size * np.sqrt(3), shape=(1,)),
            }
        )

        # We have 2 actions, shooting (1) or not shooting (0)
        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.net_cone_start = 5
        self.net_cone_end = 50

        self.max_steps = max_steps


    def _get_obs(self):
        return {"distance": np.linalg.norm(self.target_location - self.pursuer_location)}

    def _get_info(self):
        return {"pursuer location": self.pursuer_location,
                "pursuer velocity": self.pursuer_velocity,
                "target location": self.target_location,
                "target velocity": self.target_velocity}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location and velocity uniformly at random
        self.pursuer_location = self.np_random.uniform(0, self.size, size=3).astype(np.float32)
        self.pursuer_velocity = self.np_random.uniform(
            -Constraints.MAX_UAV_SPEED, Constraints.MAX_UAV_SPEED, size=3
        ).astype(np.float32)
        # The target's location and velocity are chosen the same way
        self.target_location = self.np_random.uniform(0, self.size, size=3).astype(np.float32)
        self.target_velocity = self.np_random.uniform(
            -Constraints.MAX_TARGET_SPEED, Constraints.MAX_TARGET_SPEED, size=3
        ).astype(np.float32)
        self.target_acceleration = self.np_random.uniform(
            -Constraints.MAX_TARGET_ACCELERATION, Constraints.MAX_TARGET_ACCELERATION, size=3
        ).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.timestep = 0

        return observation, info

    def step(self, action):

        reward = -0.001

        self.timestep += 1

        terminated = False
        truncated = False

        # Moving the pursuer

        uav_acceleration = tpn.get_tpn_acceleration(
            self.pursuer_velocity, self.target_velocity,
            self.pursuer_location, self.target_location
        )

        uav_location, self.pursuer_velocity = tpn.get_new_location(
            uav_acceleration, self.pursuer_velocity, self.pursuer_location, Constraints.MAX_UAV_SPEED
        )

        self.pursuer_location = np.clip(
            uav_location, 0, self.size - 1
        ).astype(np.float32)

        # Moving the target

        self.target_acceleration = tpn.target_accelaration(self.target_acceleration)

        new_target_location, self.target_velocity = tpn.get_new_location(
            self.target_acceleration, self.target_velocity, self.target_location, Constraints.MAX_TARGET_SPEED
        )

        self.target_location = np.clip(
            new_target_location, 0, self.size - 1
        ).astype(np.float32)

        if self.timestep >= self.max_steps:
            truncated = True

        if action == Actions.do_shoot.value:
            distance = np.linalg.norm(self.target_location - self.pursuer_location)
            if distance >= self.net_cone_start and distance <= self.net_cone_end:
                terminated = True
                reward = 10
            else:
                terminated = True
                reward = -1
        elif truncated:
                reward = -1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size * 3, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size*3, self.window_size))
        canvas.fill((255, 255, 255))

        # Split the window into three parts

        pygame.draw.line(
            canvas, (0, 0 ,0), (self.window_size, self.window_size), (self.window_size, 0), 5
        )

        pygame.draw.line(
            canvas, (0, 0, 0), (self.window_size*2, self.window_size), (self.window_size*2, 0), 5
        )

        # Define views first (XY XZ YZ)

        x_indices = [0, 0, 1]
        y_indices = [1, 2, 2]

        # First we draw the target (three times)

        for i in range(3):

            x_mapped = self.window_size * i + float(self.target_location[x_indices[i]])
            y_mapped = float(self.target_location[y_indices[i]])

            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    (x_mapped, y_mapped),
                    (self.marker_size, self.marker_size),
                ),
            )

        # Now we draw the agent

        for i in range(3):

            x_mapped = self.window_size * i + float(self.pursuer_location[x_indices[i]])
            y_mapped = float(self.pursuer_location[y_indices[i]])
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (x_mapped, y_mapped),
                self.marker_size * 3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
