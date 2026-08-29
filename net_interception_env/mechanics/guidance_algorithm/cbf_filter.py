import numpy as np
from typing import Tuple

def compute_cbf_matrices(r: np.ndarray, v: np.ndarray, a_t: np.ndarray, 
                         d_min: float = 1.5, k1: float = 1.5, k2: float = 3.0) -> Tuple[np.ndarray, float]:
    """
    Computes the linear constraint matrices (A_cbf * a_i <= b_cbf) for the QP safety filter.
    This is the mathematically corrected strict Relative-Degree 2 HOCBF formulation.

    Args:
        r (np.ndarray): Relative position vector [x, y, z] in meters.
        v (np.ndarray): Relative velocity vector [vx, vy, vz] in m/s.
        a_t (np.ndarray): Target acceleration vector [ax, ay, az] in m/s^2.
        d_min (float): Minimum allowable physical standoff distance in meters.
        k1 (float): First HOCBF linear gain (acts as the primary braking horizon).
        k2 (float): Second HOCBF linear gain.

    Returns:
        Tuple[np.ndarray, float]: 
            - A_cbf: The (1, 3) linear constraint matrix.
            - b_cbf: The scalar constraint upper bound.
    """
    distance_sq = np.dot(r, r)
    if distance_sq < 1e-6:
        return np.zeros((1, 3)), 0.0

    # 1. Base safety function h(x) (Relative Degree 2)
    h_x = distance_sq - (d_min ** 2)
    
    # 2. First derivative h_dot(x)
    h_dot_x = 2.0 * np.dot(r, v)
    
    # 3. Drift portion of the second derivative (f_drift)
    f_drift = 2.0 * np.dot(v, v) + 2.0 * np.dot(r, a_t)
    
    # 4. Mathematically rigorous HOCBF constraints
    # Derived from: f_drift - L_g*a_i + (k1+k2)*h_dot_x + k1*k2*h_x >= 0
    
    A_cbf = 2.0 * r
    b_cbf = f_drift + (k1 + k2) * h_dot_x + (k1 * k2) * h_x
    
    return A_cbf.reshape(1, 3), float(b_cbf)
