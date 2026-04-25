from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from torch._C import dtype

from net_interception_env.mechanics import Pro_Nav_logic as tpn, Constraints, Pro_Nav_logic


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
                "distance": spaces.Box(0, size * np.sqrt(3), shape=(1,), dtype=np.float32),
                "closing velocity": spaces.Box(
                    -(Constraints.MAX_UAV_SPEED+Constraints.MAX_TARGET_SPEED+1),
                    Constraints.MAX_UAV_SPEED+Constraints.MAX_TARGET_SPEED+1,
                    shape=(1,),
                    dtype=np.float32
                ),
                "UAV location": spaces.Box(
                    np.zeros((3,), dtype=np.float32),
                    np.array([self.size, self.size, self.size], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32
                ),
                "UAV velocity": spaces.Box(
                    np.array([-Constraints.MAX_UAV_SPEED, -Constraints.MAX_UAV_SPEED, -Constraints.MAX_UAV_SPEED], dtype=np.float32),
                    np.array([Constraints.MAX_UAV_SPEED, Constraints.MAX_UAV_SPEED, Constraints.MAX_UAV_SPEED], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32
                ),
                "Target location": spaces.Box(
                    np.zeros((3,), dtype=np.float32),
                    np.array([self.size, self.size, self.size], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32
                ),
                "Target velocity": spaces.Box(
                    np.array([-Constraints.MAX_TARGET_SPEED, -Constraints.MAX_TARGET_SPEED, -Constraints.MAX_TARGET_SPEED],
                             dtype=np.float32),
                    np.array([Constraints.MAX_TARGET_SPEED, Constraints.MAX_TARGET_SPEED, Constraints.MAX_TARGET_SPEED],
                             dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32
                )
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

        self.max_steps = max_steps


    def _get_obs(self):
        return {"distance": np.array([np.linalg.norm(self.target_location - self.pursuer_location)], dtype=np.float32),
                "closing velocity": np.array([np.linalg.norm(self.target_velocity - self.pursuer_velocity)], dtype=np.float32),
                "UAV location": self.pursuer_location,
                "UAV velocity": self.pursuer_velocity,
                "Target location": self.target_location,
                "Target velocity": self.target_velocity
                }

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

        # The interceptor is carrying a net
        self.separate_flight = False

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

        uav_location, self.pursuer_velocity, _ = tpn.get_new_location(
            uav_acceleration, self.pursuer_velocity, self.pursuer_location, Constraints.MAX_UAV_SPEED
        )

        self.pursuer_location = np.clip(
            uav_location, 0, self.size - 1
        ).astype(np.float32)

        # Moving the target

        self.target_acceleration = tpn.target_accelaration(self.target_acceleration)

        new_target_location, self.target_velocity, _ = tpn.get_new_location(
            self.target_acceleration, self.target_velocity, self.target_location, Constraints.MAX_TARGET_SPEED
        )

        self.target_location = np.clip(
            new_target_location, 0, self.size - 1
        ).astype(np.float32)

        if self.timestep >= self.max_steps:
            truncated = True

        if action == Actions.do_shoot.value and not self.separate_flight:
            self.net_location = self.pursuer_location.copy()
            self.net_radius = 0.1  # m
            self.separate_flight = True
            self.net_direction = self.pursuer_velocity / np.linalg.norm(self.pursuer_velocity)
            self.net_velocity = self.pursuer_velocity + self.net_direction*Constraints.EXTRA_VELOCITY
            self.fire_distance = max(0.1, np.linalg.norm(self.target_location - self.pursuer_location))

            # Normalize it to get a pure direction
            dir_to_target = (self.target_location - self.pursuer_location) / self.fire_distance

            # Calculate alignment (1.0 is perfect aim, 0.0 is sideways, -1.0 is completely backward)
            self.shot_alignment = np.dot(self.net_direction, dir_to_target)

        if self.separate_flight:
            self.net_location, _, self.net_radius = Pro_Nav_logic.get_new_location(
                0, self.net_velocity, self.net_location, old_radius=self.net_radius
            )
            net_to_target = self.target_location - self.net_location
            net_to_target_projection = np.dot(self.net_direction, net_to_target)
            target_offset = np.sqrt(max(0.0, np.linalg.norm(net_to_target)**2 - net_to_target_projection**2))

            net_speed = np.linalg.norm(self.net_velocity)
            step_distance = net_speed * Constraints.dt
            # print(f"Step distance: {step_distance}\n"
            #       f"target_offset: {target_offset}\n"
            #       f"net_speed: {net_speed}\n"
            #       f"net_to_target: {net_to_target_projection}\n"
            #       f"net_radius: {self.net_radius}")
            if -step_distance < net_to_target_projection < 0.1 and target_offset < self.net_radius:
                terminated = True
                reward = 30
            elif net_to_target_projection < -step_distance:
                terminated = True
                if self.shot_alignment < 0:
                    reward = -30
                else:
                    miss_distance = target_offset - self.net_radius
                    reward = -miss_distance/self.fire_distance
            elif truncated:
                reward = -20


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

            # Now we draw the expanding net (if it has been fired)
        if hasattr(self, 'separate_flight') and self.separate_flight:

            # 1. Create a 3D coordinate system for the face of the net
            normal = self.net_direction

            # Find an arbitrary vector not parallel to the normal to compute a cross product
            arbitrary = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(normal, arbitrary)) > 0.99:
                arbitrary = np.array([0.0, 1.0, 0.0])

            # U and V are orthogonal vectors that lie flat on the plane of the net
            u = np.cross(normal, arbitrary)
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)

            # 2. Generate 3D points around the net's circumference
            num_points = 20  # Resolution of the drawn shape
            angles = np.linspace(0, 2 * np.pi, num_points)

            disk_points_3d = []
            for angle in angles:
                point = self.net_location + self.net_radius * np.cos(angle) * u + self.net_radius * np.sin(
                    angle) * v
                disk_points_3d.append(point)

            # 3. Project and draw the shape on each of the 3 viewing planes
            for i in range(3):
                points_2d = []
                for pt in disk_points_3d:
                    x_mapped = self.window_size * i + float(pt[x_indices[i]])
                    y_mapped = float(pt[y_indices[i]])
                    points_2d.append((x_mapped, y_mapped))

                # Draw the projected bounding polygon (which forms the correctly angled ellipse)
                if len(points_2d) > 2:
                    pygame.draw.polygon(canvas, (0, 255, 0), points_2d, 1)

                # Draw the center point
                center_x = self.window_size * i + float(self.net_location[x_indices[i]])
                center_y = float(self.net_location[y_indices[i]])
                pygame.draw.circle(canvas, (0, 255, 0), (center_x, center_y), 2)

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