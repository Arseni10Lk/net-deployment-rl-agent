import numpy as np
import Constraints

def get_tpn_acceleration(v_pursuer, v_target, r_pursuer, r_target):
    closing_velocity = v_target - v_pursuer
    r = r_target - r_pursuer

    if r != 0:
        los_rot_rate = np.cross(r, closing_velocity) / np.dot(r, r)
        acceleration = Constraints.N * np.cross(closing_velocity, los_rot_rate)
    else:
        acceleration = np.zeros(3)

    return acceleration