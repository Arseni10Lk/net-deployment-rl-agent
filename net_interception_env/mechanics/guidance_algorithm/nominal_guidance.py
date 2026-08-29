import numpy as np

def compute_bgpn_acceleration(v_rel: np.ndarray, omega_los: np.ndarray, a_t: np.ndarray, r: np.ndarray, N: float = 4.0) -> np.ndarray:
    """
    Computes the Blended Generalized Proportional Navigation (B-GPN) nominal acceleration.
    This acts as the aggressive objective-seeking command before safety filtering.

    Args:
        v_rel (np.ndarray): Closing velocity vector [vx, vy, vz] in m/s.
        omega_los (np.ndarray): Line-of-sight angular velocity vector [wx, wy, wz] in rad/s.
        a_t (np.ndarray): Target acceleration vector [ax, ay, az] in m/s^2.
        r (np.ndarray): Relative position vector [x, y, z] in meters.
        N (float): Non-dimensional navigation constant (typically 3.0 to 5.0).

    Returns:
        np.ndarray: Nominal commanded acceleration vector [ax, ay, az].
    """
    # Standard Pro-Nav component
    a_png = N * np.cross(v_rel, omega_los)
    
    # Target acceleration compensation (projected normal to LOS)
    distance = np.linalg.norm(r)
    if distance > 1e-6:
        los_unit = r / distance
        # Remove the component of target acceleration that is parallel to LOS
        a_t_perp = a_t - np.dot(a_t, los_unit) * los_unit
    else:
        a_t_perp = np.zeros(3)
        
    return a_png + (N / 2.0) * a_t_perp
