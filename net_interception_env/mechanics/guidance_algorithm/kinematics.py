import numpy as np
from typing import Dict


def compute_relative_kinematics(
    p_i: np.ndarray, v_i: np.ndarray, p_t: np.ndarray, v_t: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes the fundamental engagement kinematics between an interceptor and a target.

    Args:
        p_i (np.ndarray): Interceptor position [x, y, z] in meters.
        v_i (np.ndarray): Interceptor velocity [vx, vy, vz] in m/s.
        p_t (np.ndarray): Target position [x, y, z] in meters.
        v_t (np.ndarray): Target velocity [vx, vy, vz] in m/s.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            'r': Relative position vector (target - interceptor)
            'v': Relative velocity vector of target with respect to interceptor
            'distance': Scalar relative distance
            'omega_los': Line-of-sight angular velocity vector
    """
    r = p_t - p_i
    v = (
        v_t - v_i
    )  # Velocity of target relative to interceptor (Closing velocity vector)
    distance = np.linalg.norm(r)

    if distance > 1e-6:
        omega_los = np.cross(r, v) / (distance**2)
    else:
        omega_los = np.zeros(3)

    return {"r": r, "v": v, "distance": np.array([distance]), "omega_los": omega_los}
