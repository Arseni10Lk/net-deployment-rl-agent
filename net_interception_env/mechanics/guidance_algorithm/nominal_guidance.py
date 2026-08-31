import numpy as np


def compute_bgpn_acceleration(
    v_c: np.ndarray,
    omega_los: np.ndarray,
    a_t: np.ndarray,
    r: np.ndarray,
    N: float = 4.0,
    k_axial: float = 2.0,
    v_close_des: float = 15.0,
) -> np.ndarray:
    """
    Computes the Blended Generalized Proportional Navigation (B-GPN) nominal acceleration,
    augmented with an axial closure term to ensure the drone aggressively closes the distance.

    Args:
        v_c (np.ndarray): Closing velocity vector (target velocity - interceptor velocity) in m/s.
        omega_los (np.ndarray): Line-of-sight angular velocity vector [wx, wy, wz] in rad/s.
        a_t (np.ndarray): Target acceleration vector [ax, ay, az] in m/s^2.
        r (np.ndarray): Relative position vector (target - interceptor) in meters.
        N (float): Non-dimensional navigation constant (typically 3.0 to 5.0).
        k_axial (float): Proportional gain for the axial closing speed controller.
        v_close_des (float): Desired closing speed (m/s) along the line of sight.

    Returns:
        np.ndarray: Nominal commanded acceleration vector [ax, ay, az].
    """
    # Proportional Navigation (Vector Form)
    # To steer towards the target's future position, we cross v_c with omega_los.
    a_png = N * np.cross(v_c, omega_los)

    distance = np.linalg.norm(r)
    if distance > 1e-6:
        los_unit = r / distance
        a_t_perp = a_t - np.dot(a_t, los_unit) * los_unit

        # Axial closure logic to ensure the drone actually closes the distance
        closing_speed = np.dot(-v_c, los_unit)
        a_axial = k_axial * (v_close_des - closing_speed) * los_unit
    else:
        a_t_perp = np.zeros(3)
        a_axial = np.zeros(3)

    return a_png + (N / 2.0) * a_t_perp + a_axial
