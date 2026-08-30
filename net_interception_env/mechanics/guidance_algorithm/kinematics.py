import numpy as np
from typing import Dict

def compute_relative_kinematics(p_i: np.ndarray, v_i: np.ndarray, p_t: np.ndarray, v_t: np.ndarray) -> Dict[str, np.ndarray]:
    r = p_t - p_i
    v = v_t - v_i  # Velocity of target relative to interceptor (Closing velocity vector)
    distance = np.linalg.norm(r)
    
    if distance > 1e-6:
        omega_los = np.cross(r, v) / (distance ** 2)
    else:
        omega_los = np.zeros(3)
        
    return {
        'r': r,
        'v': v,
        'distance': np.array([distance]),
        'omega_los': omega_los
    }
