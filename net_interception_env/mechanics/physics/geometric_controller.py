import numpy as np

from net_interception_env.mechanics.physics.rotations import (
    so3_attitude_error,
    so3_angular_rate_error,
)


class SO3GeometricController:
    """
    Geometric Tracking Controller on SO(3) based on Lee (2010).
    Converts 3D commanded acceleration into body-frame thrust and torques.
    """

    def __init__(
        self,
        mass: float = 1.0,
        J: np.ndarray | None = None,
        g: np.ndarray | None = None,
        k_R: np.ndarray | float = 8.0,
        k_omega: np.ndarray | float = 1.5,
    ):
        """
        Args:
            mass (float): Interceptor mass [kg].
            J (np.ndarray | None): 3x3 inertia matrix [kg*m^2].
            g (np.ndarray | None): 3D gravity vector [m/s^2] in world frame (default [0, 0, -9.81]).
            k_R (np.ndarray | float): Attitude error gain (scalar or 3-element vector/matrix).
            k_omega (np.ndarray | float): Angular rate error gain (scalar or 3-element vector/matrix).
        """
        self.mass = mass
        if J is None:
            # Standard 5-inch racing / interceptor drone inertia
            self.J = np.diag([0.005, 0.005, 0.009])
        else:
            self.J = np.array(J, dtype=np.float64)

        if g is None:
            self.g = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        else:
            self.g = np.array(g, dtype=np.float64)

        if np.isscalar(k_R):
            self.k_R = np.diag([float(k_R)] * 3)
        elif isinstance(k_R, np.ndarray) and k_R.ndim == 1:
            self.k_R = np.diag(k_R)
        else:
            self.k_R = np.array(k_R, dtype=np.float64)

        if np.isscalar(k_omega):
            self.k_omega = np.diag([float(k_omega)] * 3)
        elif isinstance(k_omega, np.ndarray) and k_omega.ndim == 1:
            self.k_omega = np.diag(k_omega)
        else:
            self.k_omega = np.array(k_omega, dtype=np.float64)

    def compute_desired_force(
        self,
        a_cmd: np.ndarray,
        f_drag: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes the desired force vector with gravity compensation and drag feedforward:
            F_des = m * (a_cmd - g) + F_drag

        Args:
            a_cmd (np.ndarray): Commanded acceleration [ax, ay, az] in world frame.
            f_drag (np.ndarray | None): World-frame aerodynamic drag force (feedforward).

        Returns:
            np.ndarray: Desired force vector F_des in world frame [N].
        """
        if f_drag is None:
            f_drag = np.zeros(3, dtype=np.float64)

        return self.mass * (a_cmd - self.g) + f_drag

    def compute_desired_thrust(
        self,
        F_des: np.ndarray,
        R: np.ndarray,
    ) -> float:
        """
        Projects desired force vector onto current body z-axis (FLU frame: +Z is Up):
            T = F_des . z_body

        Args:
            F_des (np.ndarray): Desired force vector in world frame.
            R (np.ndarray): Current 3x3 rotation matrix (body to world).

        Returns:
            float: Total scalar thrust T [N].
        """
        z_body = R[:, 2]  # Body Z-axis in world frame
        T = float(np.dot(F_des, z_body))
        return max(0.0, T)

    def compute_desired_attitude(
        self,
        F_des: np.ndarray,
        yaw_desired: float = 0.0,
    ) -> np.ndarray:
        """
        Constructs the desired rotation matrix R_d in SO(3).
        Aligns body Z-axis with F_des while respecting the commanded yaw heading.

        Args:
            F_des (np.ndarray): Desired force vector in world frame.
            yaw_desired (float): Commanded yaw angle in radians.

        Returns:
            np.ndarray: 3x3 desired rotation matrix R_d.
        """
        norm_f = np.linalg.norm(F_des)
        if norm_f < 1e-6:
            # Fallback if commanded zero force: hover attitude with yaw
            cy, sy = np.cos(yaw_desired), np.sin(yaw_desired)
            return np.array(
                [
                    [cy, -sy, 0.0],
                    [sy, cy, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        z_d = F_des / norm_f

        # Reference heading direction in the horizontal plane
        x_proj = np.array([np.cos(yaw_desired), np.sin(yaw_desired), 0.0], dtype=np.float64)

        # Cross product to get desired body Y-axis
        y_d_dir = np.cross(z_d, x_proj)
        norm_y_d = np.linalg.norm(y_d_dir)

        if norm_y_d < 1e-6:
            # Singularity: z_d is aligned with x_proj (drone pitching 90 deg)
            # Use alternative reference vector in Y
            y_proj = np.array([-np.sin(yaw_desired), np.cos(yaw_desired), 0.0], dtype=np.float64)
            x_d_dir = np.cross(y_proj, z_d)
            x_d = x_d_dir / np.linalg.norm(x_d_dir)
            y_d = np.cross(z_d, x_d)
        else:
            y_d = y_d_dir / norm_y_d
            x_d = np.cross(y_d, z_d)

        R_d = np.column_stack([x_d, y_d, z_d])
        return R_d

    def compute_control_torques(
        self,
        R: np.ndarray,
        R_d: np.ndarray,
        omega: np.ndarray,
        omega_d: np.ndarray | None = None,
        omega_d_dot: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes control torques on SO(3) with gyroscopic cancellation and angular acceleration feedforward:
            tau = -k_R * e_R - k_omega * e_omega + omega x (J @ omega)
                  - J @ (omega x (R^T @ R_d @ omega_d) - R^T @ R_d @ omega_d_dot)

        Args:
            R (np.ndarray): Current 3x3 rotation matrix.
            R_d (np.ndarray): Desired 3x3 rotation matrix.
            omega (np.ndarray): Current body angular velocity [p, q, r].
            omega_d (np.ndarray | None): Desired body angular velocity (default 0).
            omega_d_dot (np.ndarray | None): Desired angular acceleration (feedforward).

        Returns:
            np.ndarray: Commanded torque vector tau = [tau_phi, tau_theta, tau_psi] [N*m].
        """
        if omega_d is None:
            omega_d = np.zeros(3, dtype=np.float64)
        if omega_d_dot is None:
            omega_d_dot = np.zeros(3, dtype=np.float64)

        # SO(3) attitude and rate error vectors
        e_R = so3_attitude_error(R, R_d)
        e_omega = so3_angular_rate_error(omega, R, R_d, omega_d)

        # Gyroscopic torque compensation
        J_omega = self.J @ omega
        gyro_comp = np.cross(omega, J_omega)

        # Feedforward term
        R_rel = R.T @ R_d
        omega_ref = R_rel @ omega_d
        feedforward = self.J @ (np.cross(omega, omega_ref) - R_rel @ omega_d_dot)

        # Full SO(3) control law
        tau = -self.k_R @ e_R - self.k_omega @ e_omega + gyro_comp - feedforward
        return tau

    def compute_commands(
        self,
        a_cmd: np.ndarray,
        R: np.ndarray,
        omega: np.ndarray,
        f_drag: np.ndarray | None = None,
        yaw_desired: float = 0.0,
        omega_d: np.ndarray | None = None,
        omega_d_dot: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Convenience wrapper computing desired thrust, torques, and desired attitude.

        Returns:
            tuple[float, np.ndarray, np.ndarray]: (T, tau, R_d)
        """
        F_des = self.compute_desired_force(a_cmd, f_drag=f_drag)
        R_d = self.compute_desired_attitude(F_des, yaw_desired=yaw_desired)
        T = self.compute_desired_thrust(F_des, R)
        tau = self.compute_control_torques(
            R, R_d, omega, omega_d=omega_d, omega_d_dot=omega_d_dot
        )
        return T, tau, R_d
