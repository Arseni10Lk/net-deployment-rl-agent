from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    dont_shoot = 0
    do_shoot = 1

class DroneNetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, marker_size=5, size=512, max_steps=1000):
        self.size = size  # The size of the environment
        self.marker_size = marker_size # The size of the markers
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(1,)),
                "target": spaces.Box(0, size - 1, shape=(1,)),
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
        self.net_cone_end = 10

        self.max_steps = max_steps


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": abs(self._target_location - self._agent_location)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.uniform(0, self.size, size=1)

        # The target's location is chosen the same way
        self._target_location = self.np_random.uniform(0, self.size, size=1)
        self.target_direction = self.np_random.integers(-1, 2)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.timestep = 0

        return observation, info

    def step(self, action):

        self.timestep += 1

        terminated = False
        truncated = False

        direction = np.sign(self._target_location - self._agent_location)
        self.target_direction = np.clip(self.np_random.normal(
            self.target_direction,
            0.2,
        ), -1.0, 1.0)

        self._agent_location = np.clip(
            self._agent_location + direction*0.5, 0, self.size - 1
        )

        self._target_location = np.clip(
            self._target_location + self.target_direction, 0, self.size - 1
        )

        if self.timestep >= self.max_steps:
            truncated = True

        if action == Actions.do_shoot.value:
            distance = self._target_location - self._agent_location
            if distance >= self.net_cone_start and distance <= self.net_cone_end:
                terminated = True
                reward = 1
            elif truncated:
                reward = -1
            else:
                reward = -1
        else:
            reward = -0.001

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
            self.window = pygame.display.set_mode((self.window_size, self.marker_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.marker_size))
        canvas.fill((255, 255, 255))

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (float(self._target_location[0]), 0),
                (self.marker_size, self.marker_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (float(self._agent_location[0]), self.marker_size / 2),
            self.marker_size / 3,
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
