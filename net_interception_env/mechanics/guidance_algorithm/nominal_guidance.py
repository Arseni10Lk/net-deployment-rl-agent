import numpy as np

def compute_bgpn_acceleration(v_c: np.ndarray, omega_los: np.ndarray, a_t: np.ndarray, r: np.ndarray, N: float = 4.0, k_axial: float = 2.0, v_close_des: float = 15.0) -> np.ndarray:
    """
    Computes the Blended Generalized Proportional Navigation (B-GPN) nominal acceleration.
    """
    # Proportional Navigation (Vector Form)
    # v_c is the closing velocity vector (v_t - v_i)
    # To steer towards the target's future position, we cross v_c with omega_los.
    a_png = N * np.cross(v_c, omega_los)
    
    distance = np.linalg.norm(r)
    if distance > 1e-6:
        los_unit = r / distance
        a_t_perp = a_t - np.dot(a_t, los_unit) * los_unit
        
        # Axial closure logic to ensure the drone doesn't just match bearings but actually closes the distance
        # Closing speed is the projection of -v_c onto the LOS
        closing_speed = np.dot(-v_c, los_unit)
        a_axial = k_axial * (v_close_des - closing_speed) * los_unit
    else:
        a_t_perp = np.zeros(3)
        a_axial = np.zeros(3)
        
    return a_png + (N / 2.0) * a_t_perp + a_axial
