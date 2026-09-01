import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """
    Normalizes a quaternion [qw, qx, qy, qz] to unit length.
    Guarantees qw >= 0 for uniqueness if desired, but preserves sign if close.
    """
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts a unit quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix R in SO(3).
    R transforms a vector from the body frame to the world frame: v_world = R @ v_body.
    """
    q_norm = normalize_quat(q)
    qw, qx, qy, qz = q_norm

    R = np.array(
        [
            [
                1.0 - 2.0 * (qy**2 + qz**2),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx**2 + qz**2),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )
    return R


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix R in SO(3) to a unit quaternion [qw, qx, qy, qz]
    using Shepperd's algorithm for numerical robustness.
    """
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return normalize_quat(q)


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiplies two quaternions q1 * q2 (Hamilton convention).
    Quaternions are in format [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Computes the quaternion time derivative:
        q_dot = 0.5 * q (x) [0, p, q, r]^T
    where omega = [p, q, r] is body-frame angular velocity.
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
    return 0.5 * quat_mult(q, omega_quat)


def hat(v: np.ndarray) -> np.ndarray:
    """
    Lie algebra hat operator: maps a 3D vector v in R^3 to a 3x3 skew-symmetric matrix in so(3).
    For any vector u, hat(v) @ u = v x u.
    """
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def vee(S: np.ndarray) -> np.ndarray:
    """
    Lie algebra vee operator: extracts the 3D vector v in R^3 from a 3x3 skew-symmetric matrix S in so(3).
    Inverse of the hat operator.
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def so3_attitude_error(R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """
    Computes the attitude error vector e_R in R^3 on SO(3) between current rotation R and desired R_d:
        e_R = 0.5 * vee(R_d^T @ R - R^T @ R_d)
    """
    error_matrix = 0.5 * (R_d.T @ R - R.T @ R_d)
    return vee(error_matrix)


def so3_angular_rate_error(
    omega: np.ndarray,
    R: np.ndarray,
    R_d: np.ndarray,
    omega_d: np.ndarray,
) -> np.ndarray:
    """
    Computes the angular velocity error vector e_omega in R^3:
        e_omega = omega - R^T @ R_d @ omega_d
    """
    return omega - R.T @ R_d @ omega_d
