import numpy as np

from net_interception_env.mechanics.physics.quadrotor_dynamics import (
    QuadrotorParams,
    QuadrotorState,
    integrate_rk4,
)
from net_interception_env.mechanics.physics.geometric_controller import (
    SO3GeometricController,
)
from net_interception_env.mechanics.physics.aerodynamics import compute_drag_force
from net_interception_env.mechanics.physics.control_allocation import (
    thrust_torques_to_motor_speeds,
)


class Quadrotor6DOF:
    """
    Complete 6-DOF Quadrotor Interceptor Simulation with 17 dynamic states:
    [pos(3), vel(3), quat(4), omega(3), motor_speeds(4)].

    Driven by:
    - SO(3) Geometric Flight Controller with Drag Feedforward
    - X-frame control allocation in FLU body coordinates
    - Anisotropic aerodynamic drag
    - First-order motor spool lag
    - RK4 numerical integration
    """

    def __init__(
        self,
        params: QuadrotorParams | None = None,
        controller: SO3GeometricController | None = None,
    ):
        self.params = params if params is not None else QuadrotorParams()
        self.state = QuadrotorState()

        if controller is not None:
            self.controller = controller
        else:
            self.controller = SO3GeometricController(
                mass=self.params.mass,
                J=self.params.inertia,
                g=self.params.gravity,
            )

        # Set initial motor speeds to hover equilibrium
        self.reset()

    def reset(
        self,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        quat: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        motor_speeds: np.ndarray | None = None,
    ) -> QuadrotorState:
        """
        Resets the 17-state quadrotor to initial conditions.
        Defaults to steady hover at position [0, 0, 0].
        """
        self.state.pos = (
            np.zeros(3, dtype=np.float64) if pos is None else np.array(pos, dtype=np.float64)
        )
        self.state.vel = (
            np.zeros(3, dtype=np.float64) if vel is None else np.array(vel, dtype=np.float64)
        )
        self.state.quat = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if quat is None
            else np.array(quat, dtype=np.float64)
        )
        self.state.omega = (
            np.zeros(3, dtype=np.float64)
            if omega is None
            else np.array(omega, dtype=np.float64)
        )

        if motor_speeds is None:
            hover_rpm = self.params.hover_motor_speed
            self.state.motor_speeds = np.full(4, hover_rpm, dtype=np.float64)
        else:
            self.state.motor_speeds = np.array(motor_speeds, dtype=np.float64)

        return self.state.copy()

    def step_acceleration(
        self,
        a_cmd: np.ndarray,
        dt: float,
        yaw_desired: float = 0.0,
        omega_d: np.ndarray | None = None,
        omega_d_dot: np.ndarray | None = None,
    ) -> tuple[QuadrotorState, dict]:
        """
        Executes a simulation step using an acceleration command a_cmd (e.g. from HOCBF-PN).

        1. Computes drag force for feedforward compensation.
        2. SO(3) Geometric Controller generates desired thrust T, torques tau, and R_d.
        3. Control allocation matrix maps (T, tau) to commanded motor speeds Omega_cmd.
        4. RK4 integrates the 17-state dynamics over dt, capturing motor spool lag.

        Args:
            a_cmd (np.ndarray): Commanded acceleration [ax, ay, az] in world frame [m/s^2].
            dt (float): Physics time step in seconds.
            yaw_desired (float): Desired yaw heading in radians.
            omega_d (np.ndarray | None): Desired angular velocity for feedforward.
            omega_d_dot (np.ndarray | None): Desired angular acceleration for feedforward.

        Returns:
            tuple[QuadrotorState, dict]: Next state and detailed telemetry dictionary.
        """
        R = self.state.rot_matrix

        # 1. Compute current drag force for Drag Feedforward
        f_drag_world, f_drag_body = compute_drag_force(
            self.state.vel, R, self.params.d_lin, self.params.d_quad
        )

        # 2. SO(3) Controller
        T_des, tau_des, R_d = self.controller.compute_commands(
            a_cmd=a_cmd,
            R=R,
            omega=self.state.omega,
            f_drag=f_drag_world,
            yaw_desired=yaw_desired,
            omega_d=omega_d,
            omega_d_dot=omega_d_dot,
        )

        # 3. Control allocation to motor speeds
        omega_cmd = thrust_torques_to_motor_speeds(
            T=T_des,
            tau=tau_des,
            M_inv=self.params.inv_mixing_matrix,
            omega_min=self.params.omega_min,
            omega_max=self.params.omega_max,
        )

        # 4. Integrate 17-state dynamics with RK4
        self.state = integrate_rk4(
            state=self.state,
            omega_cmd=omega_cmd,
            params=self.params,
            dt=dt,
        )

        telemetry = {
            "T_des": T_des,
            "tau_des": tau_des,
            "R_d": R_d,
            "omega_cmd": omega_cmd,
            "f_drag_world": f_drag_world,
            "f_drag_body": f_drag_body,
            "actual_motor_speeds": self.state.motor_speeds.copy(),
        }
        return self.state.copy(), telemetry

    def step_motor_speeds(
        self,
        omega_cmd: np.ndarray,
        dt: float,
    ) -> QuadrotorState:
        """
        Direct low-level motor speed control step using RK4.
        """
        self.state = integrate_rk4(
            state=self.state,
            omega_cmd=omega_cmd,
            params=self.params,
            dt=dt,
        )
        return self.state.copy()
