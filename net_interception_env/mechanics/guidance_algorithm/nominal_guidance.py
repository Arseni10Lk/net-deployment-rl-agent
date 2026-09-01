import numpy as np


def compute_bgpn_acceleration(
    v_c: np.ndarray,
    omega_los: np.ndarray,
    a_t: np.ndarray,
    r: np.ndarray,
    N: float = 4.0,
    k_axial: float = 2.0,
    v_close_max: float = 15.0,
) -> np.ndarray:
    """
    Computes the Blended Generalized Proportional Navigation (B-GPN) nominal acceleration,
    augmented with a distance-dependent axial closure term to ensure smooth pursuit without ramming.

    Args:
        v_c (np.ndarray): Relative velocity vector (target - interceptor) in m/s.
        omega_los (np.ndarray): Line-of-sight angular velocity vector [wx, wy, wz] in rad/s.
        a_t (np.ndarray): Target acceleration vector [ax, ay, az] in m/s^2.
        r (np.ndarray): Relative position vector (target - interceptor) in meters.
        N (float): Non-dimensional navigation constant (typically 3.0 to 5.0).
        k_axial (float): Proportional gain for the axial closing speed controller.
        v_close_max (float): Maximum allowed closing speed (m/s).

    Returns:
        np.ndarray: Nominal commanded acceleration vector [ax, ay, az].
    """
    distance = np.linalg.norm(r)
    if distance > 1e-6:
        los_unit = r / distance
        
        # 1. True Proportional Navigation (TPN) with absolute closing speed
        # The closing speed is positive if the distance is decreasing.
        closing_speed = np.dot(-v_c, los_unit)
        
        # Using effective_vc prevents the PN command from flipping direction 
        # if the drone is temporarily pushed backwards. It must always steer towards target.
        effective_vc = max(1.0, abs(closing_speed))
        
        # TPN formula: a_cmd = N * V_c * (omega_los X los_unit)
        a_png = N * effective_vc * np.cross(omega_los, los_unit)

        # 2. Target Acceleration Compensation (Feedforward)
        # Cap a_t to avoid saturating the interceptor's motors when tracking highly acrobatic targets
        a_t_norm = np.linalg.norm(a_t)
        if a_t_norm > 15.0:
            a_t_capped = a_t / a_t_norm * 15.0
        else:
            a_t_capped = a_t
            
        a_t_perp = a_t_capped - np.dot(a_t_capped, los_unit) * los_unit

        # 3. Axial Closure Logic (Distance-dependent velocity profile)
        # We want to approach the target but slow down smoothly as we approach the CBF boundary (1.5m).
        # A proportional controller on distance: v_desired = k * (distance - d_safe)
        d_safe = 2.5 
        v_close_des = min(v_close_max, 2.5 * max(0.0, distance - d_safe))
        
        a_axial = k_axial * (v_close_des - closing_speed) * los_unit
    else:
        a_png = np.zeros(3)
        a_t_perp = np.zeros(3)
        a_axial = np.zeros(3)

    return a_png + (N / 2.0) * a_t_perp + a_axial
