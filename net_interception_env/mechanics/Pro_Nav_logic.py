import numpy as np
from net_interception_env.mechanics import Constraints

def get_tpn_acceleration(v_pursuer, v_target, r_pursuer, r_target):
    relative_velocity = v_target - v_pursuer
    r = r_target - r_pursuer
    closing_velocity = np.linalg.norm(np.dot(relative_velocity, r))

    if np.linalg.norm(r) > 1e-6:
        if np.linalg.norm(v_pursuer) > 1e-6:
            heading = v_pursuer / np.linalg.norm(v_pursuer)
        else:
            heading = r / np.linalg.norm(r)

        los_rot_rate = np.cross(r, relative_velocity) / np.dot(r, r)
        acceleration = - Constraints.N * closing_velocity * np.cross(heading, los_rot_rate)
    else:
        acceleration = np.zeros(3)

    if np.linalg.norm(acceleration) > Constraints.MAX_UAV_ACCELERATION:
        acceleration = acceleration / np.linalg.norm(acceleration) * Constraints.MAX_UAV_ACCELERATION

    return acceleration

def get_new_location(acceleration, old_v, old_r, max_speed=1000, old_radius=0):

    # Semi-implicit Euler integration

    velocity = old_v + Constraints.dt * acceleration
    if np.linalg.norm(velocity) > max_speed:
        velocity = velocity / np.linalg.norm(velocity) * max_speed
    position = old_r + Constraints.dt * velocity
    radius = old_radius + Constraints.dt * velocity * Constraints.EXPANSION_RATE

    return position, velocity, radius

def target_accelaration(old_a):

    acceleration = np.random.uniform(
        old_a - Constraints.TARGET_DELTA_ACCELERATION, old_a + Constraints.TARGET_DELTA_ACCELERATION, size=3
    ).astype(np.float32)

    if np.linalg.norm(acceleration) > Constraints.MAX_TARGET_ACCELERATION:
        acceleration = acceleration / np.linalg.norm(acceleration) * Constraints.MAX_TARGET_ACCELERATION

    return acceleration