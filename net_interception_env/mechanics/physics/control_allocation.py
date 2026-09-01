import numpy as np


def compute_mixing_matrix(kf: float, km: float, arm_length: float) -> np.ndarray:
    """
    Computes the 4x4 X-configuration mixing matrix in the FLU body frame.

    Args:
        kf (float): Thrust coefficient [N / (rad/s)^2].
        km (float): Yaw drag torque coefficient [N*m / (rad/s)^2].
        arm_length (float): Distance L from drone center to motor center [m].

    Returns:
        np.ndarray: 4x4 mixing matrix mapping [Omega_1^2, Omega_2^2, Omega_3^2, Omega_4^2]^T
                    to [T, tau_phi, tau_theta, tau_psi]^T.
    """
    l = arm_length * (np.sqrt(2.0) / 2.0)  # perpendicular moment arm to roll/pitch axes

    # Motors: 1=Back-Right, 2=Front-Right, 3=Back-Left, 4=Front-Left
    # Row 0: Total Thrust T
    # Row 1: Roll torque tau_phi (y_i * F_i)
    # Row 2: Pitch torque tau_theta (-x_i * F_i)
    # Row 3: Yaw torque tau_psi
    M = np.array(
        [
            [kf, kf, kf, kf],
            [-kf * l, -kf * l, kf * l, kf * l],
            [kf * l, -kf * l, kf * l, -kf * l],
            [km, -km, -km, km],
        ],
        dtype=np.float64,
    )
    return M


def motor_speeds_to_thrust_torques(
    omega_motors: np.ndarray,
    M: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Computes total thrust T and torques tau = [tau_phi, tau_theta, tau_psi]
    from current motor speeds Omega.

    Args:
        omega_motors (np.ndarray): Current rotor angular speeds [Omega_1, Omega_2, Omega_3, Omega_4] in rad/s.
        M (np.ndarray): 4x4 mixing matrix.

    Returns:
        tuple[float, np.ndarray]: (T, tau), where T is scalar thrust (N) and tau is 3D torque vector (N*m).
    """
    omega_sq = omega_motors**2
    out = M @ omega_sq
    T = float(out[0])
    tau = out[1:4]
    return T, tau


def thrust_torques_to_motor_speeds(
    T: float,
    tau: np.ndarray,
    M_inv: np.ndarray,
    omega_min: float = 0.0,
    omega_max: float = 2000.0,
) -> np.ndarray:
    """
    Inverts the control allocation matrix to compute desired motor speeds Omega_cmd
    from desired total thrust T and torques tau.

    Args:
        T (float): Commanded total thrust [N].
        tau (np.ndarray): Commanded torques [tau_phi, tau_theta, tau_psi] [N*m].
        M_inv (np.ndarray): Inverse of the 4x4 mixing matrix.
        omega_min (float): Minimum achievable motor speed [rad/s].
        omega_max (float): Maximum achievable motor speed [rad/s].

    Returns:
        np.ndarray: Commanded motor speeds [Omega_1, Omega_2, Omega_3, Omega_4] in rad/s.
    """
    target_vec = np.array([T, tau[0], tau[1], tau[2]], dtype=np.float64)
    omega_sq_cmd = M_inv @ target_vec

    # Physical clamping: motor squared speed cannot be negative or exceed max^2
    omega_sq_clamped = np.clip(omega_sq_cmd, omega_min**2, omega_max**2)
    omega_cmd = np.sqrt(omega_sq_clamped)
    return omega_cmd


def compute_motor_derivatives(
    omega_current: np.ndarray,
    omega_cmd: np.ndarray,
    tau_m: float,
) -> np.ndarray:
    """
    Computes the time derivative of motor speeds based on first-order motor lag:
        dot{Omega}_i = (1 / tau_m) * (Omega_cmd,i - Omega_i)

    Args:
        omega_current (np.ndarray): Physical rotor speeds [Omega_1, Omega_2, Omega_3, Omega_4].
        omega_cmd (np.ndarray): Commanded rotor speeds.
        tau_m (float): Motor time constant (seconds).

    Returns:
        np.ndarray: dot{Omega} rotor speed acceleration [rad/s^2].
    """
    return (omega_cmd - omega_current) / tau_m
