import numpy as np


def compute_drag_force(
    v_world: np.ndarray,
    R: np.ndarray,
    d_lin: np.ndarray,
    d_quad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes anisotropic aerodynamic drag force on the quadrotor body.

    As detailed in Physics.md:
    1. Transform velocity to Body Frame: v_body = R^T @ v_world
    2. Compute Body-Frame Drag: F_drag_body = D_lin @ v_body + D_quad @ (|v_body| * v_body)
    3. Transform back to World Frame: F_drag_world = R @ F_drag_body

    Args:
        v_world (np.ndarray): Velocity in world frame [vx, vy, vz], shape (3,).
        R (np.ndarray): 3x3 rotation matrix from body to world frame.
        d_lin (np.ndarray): Linear drag coefficients [Dx, Dy, Dz], shape (3,) or 3x3 diagonal.
        d_quad (np.ndarray): Quadratic drag coefficients [Dx_q, Dy_q, Dz_q], shape (3,) or 3x3 diagonal.

    Returns:
        tuple[np.ndarray, np.ndarray]: (F_drag_world, F_drag_body), each shape (3,).
    """
    # Transform velocity to Body Frame (FLU)
    v_body = R.T @ v_world

    # Support both 1D vector and 2D diagonal matrix
    if d_lin.ndim == 1:
        d_lin_diag = d_lin
    else:
        d_lin_diag = np.diag(d_lin)

    if d_quad.ndim == 1:
        d_quad_diag = d_quad
    else:
        d_quad_diag = np.diag(d_quad)

    # Compute anisotropic drag in body frame
    f_drag_lin = d_lin_diag * v_body
    f_drag_quad = d_quad_diag * (np.abs(v_body) * v_body)
    f_drag_body = f_drag_lin + f_drag_quad

    # Transform drag force back to World Frame
    f_drag_world = R @ f_drag_body

    return f_drag_world, f_drag_body
