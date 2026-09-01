from dataclasses import dataclass, field
import numpy as np

from net_interception_env.mechanics.physics.rotations import (
    quat_to_rot_matrix,
    quat_derivative,
    normalize_quat,
)
from net_interception_env.mechanics.physics.aerodynamics import compute_drag_force
from net_interception_env.mechanics.physics.control_allocation import (
    compute_mixing_matrix,
    motor_speeds_to_thrust_torques,
    compute_motor_derivatives,
)


@dataclass
class QuadrotorParams:
    """Physical parameters of the 6-DOF interceptor quadrotor."""

    mass: float = 1.0  # kg
    arm_length: float = 0.15  # m (wheelbase ~30cm)
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.005, 0.005, 0.009])
    )  # kg*m^2
    kf: float = 1.0e-5  # Thrust coeff [N / (rad/s)^2]
    km: float = 1.6e-7  # Torque coeff [N*m / (rad/s)^2]
    tau_m: float = 0.03  # Motor time constant [s] (30ms lag)
    d_lin: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.1, 0.2])
    )  # Linear drag [N / (m/s)]
    d_quad: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.15])
    )  # Quadratic drag [N / (m/s)^2]
    omega_min: float = 0.0  # rad/s
    omega_max: float = 1500.0  # rad/s (~14,300 RPM)
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )  # m/s^2

    # Cached mixing matrix and its inverse
    mixing_matrix: np.ndarray = field(init=False)
    inv_mixing_matrix: np.ndarray = field(init=False)
    inv_inertia: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mixing_matrix = compute_mixing_matrix(
            self.kf, self.km, self.arm_length
        )
        self.inv_mixing_matrix = np.linalg.inv(self.mixing_matrix)
        self.inv_inertia = np.linalg.inv(self.inertia)

    @property
    def hover_motor_speed(self) -> float:
        """Motor speed [rad/s] required to maintain steady hover against gravity."""
        return float(np.sqrt((self.mass * abs(self.gravity[2])) / (4.0 * self.kf)))


@dataclass
class QuadrotorState:
    """17-variable physical state of the quadrotor."""

    pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )  # [x, y, z] in world frame
    vel: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )  # [vx, vy, vz] in world frame
    quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )  # [qw, qx, qy, qz]
    omega: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )  # [p, q, r] in body frame
    motor_speeds: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )  # [Omega1, Omega2, Omega3, Omega4] in rad/s

    @property
    def rot_matrix(self) -> np.ndarray:
        """Current 3x3 rotation matrix (body to world)."""
        return quat_to_rot_matrix(self.quat)

    def copy(self) -> "QuadrotorState":
        return QuadrotorState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            quat=self.quat.copy(),
            omega=self.omega.copy(),
            motor_speeds=self.motor_speeds.copy(),
        )

    def to_array(self) -> np.ndarray:
        """Packs 17 state variables into a flat 1D numpy array."""
        return np.concatenate(
            [self.pos, self.vel, self.quat, self.omega, self.motor_speeds]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "QuadrotorState":
        """Unpacks flat 17-element array into QuadrotorState."""
        return cls(
            pos=arr[0:3].copy(),
            vel=arr[3:6].copy(),
            quat=normalize_quat(arr[6:10].copy()),
            omega=arr[10:13].copy(),
            motor_speeds=arr[13:17].copy(),
        )


def compute_derivatives(
    state: QuadrotorState,
    omega_cmd: np.ndarray,
    params: QuadrotorParams,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """
    Computes time derivatives of all 17 state variables:
        dot{p} = v
        dot{v} = g + (1/m) * (R @ T_body - F_drag)
        dot{q} = 0.5 * q (x) [0, omega]
        dot{omega} = J^{-1} * (tau - omega x (J @ omega))
        dot{Omega} = (1 / tau_m) * (Omega_cmd - Omega)

    Returns:
        tuple[np.ndarray, dict]: Flat 17-element derivative vector and info dictionary.
    """
    R = state.rot_matrix

    # Aerodynamic drag in world frame
    f_drag_world, f_drag_body = compute_drag_force(
        state.vel, R, params.d_lin, params.d_quad
    )

    # Motor speeds produce thrust and torques
    T, tau = motor_speeds_to_thrust_torques(state.motor_speeds, params.mixing_matrix)

    # In FLU body frame, thrust is strictly along +Z: T_body = [0, 0, T]
    T_body = np.array([0.0, 0.0, T], dtype=np.float64)
    thrust_world = R @ T_body

    # Translational dynamics
    pos_dot = state.vel
    vel_dot = params.gravity + (thrust_world - f_drag_world) / params.mass

    # Rotational kinematics (quaternion derivative)
    quat_dot = quat_derivative(state.quat, state.omega)

    # Rotational dynamics (Euler's equations)
    J_omega = params.inertia @ state.omega
    gyro_torque = np.cross(state.omega, J_omega)
    omega_dot = params.inv_inertia @ (tau - gyro_torque)

    # Motor lag dynamics
    motor_dot = compute_motor_derivatives(
        state.motor_speeds, omega_cmd, params.tau_m
    )

    deriv_array = np.concatenate([pos_dot, vel_dot, quat_dot, omega_dot, motor_dot])

    info = {
        "thrust": T,
        "tau": tau,
        "f_drag_world": f_drag_world,
        "f_drag_body": f_drag_body,
        "R": R,
    }
    return deriv_array, info


def integrate_rk4(
    state: QuadrotorState,
    omega_cmd: np.ndarray,
    params: QuadrotorParams,
    dt: float,
) -> QuadrotorState:
    """
    Steps the 17-variable system forward by dt using 4th-Order Runge-Kutta (RK4).
    Guarantees quaternion normalization at the end of each step.
    """
    omega_cmd_clamped = np.clip(omega_cmd, params.omega_min, params.omega_max)

    y0 = state.to_array()

    # k1
    k1, _ = compute_derivatives(QuadrotorState.from_array(y0), omega_cmd_clamped, params)

    # k2
    y1 = y0 + 0.5 * dt * k1
    k2, _ = compute_derivatives(QuadrotorState.from_array(y1), omega_cmd_clamped, params)

    # k3
    y2 = y0 + 0.5 * dt * k2
    k3, _ = compute_derivatives(QuadrotorState.from_array(y2), omega_cmd_clamped, params)

    # k4
    y3 = y0 + dt * k3
    k4, _ = compute_derivatives(QuadrotorState.from_array(y3), omega_cmd_clamped, params)

    # Update state
    y_next = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Construct next state with normalized quaternion and motor clamping
    next_state = QuadrotorState.from_array(y_next)
    next_state.quat = normalize_quat(next_state.quat)
    next_state.motor_speeds = np.clip(
        next_state.motor_speeds, params.omega_min, params.omega_max
    )
    return next_state


def integrate_euler(
    state: QuadrotorState,
    omega_cmd: np.ndarray,
    params: QuadrotorParams,
    dt: float,
) -> QuadrotorState:
    """
    Steps the 17-variable system forward by dt using 1st-Order Euler integration.
    """
    omega_cmd_clamped = np.clip(omega_cmd, params.omega_min, params.omega_max)
    deriv, _ = compute_derivatives(state, omega_cmd_clamped, params)
    y_next = state.to_array() + dt * deriv

    next_state = QuadrotorState.from_array(y_next)
    next_state.quat = normalize_quat(next_state.quat)
    next_state.motor_speeds = np.clip(
        next_state.motor_speeds, params.omega_min, params.omega_max
    )
    return next_state
