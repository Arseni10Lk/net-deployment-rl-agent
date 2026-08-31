import numpy as np
from net_interception_env.mechanics.guidance_algorithm.kinematics import (
    compute_relative_kinematics,
)


def test_compute_relative_kinematics_standard():
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([10.0, 0.0, 0.0])
    p_t = np.array([10.0, 0.0, 0.0])
    v_t = np.array([0.0, 5.0, 0.0])
    result = compute_relative_kinematics(p_i, v_i, p_t, v_t)
    np.testing.assert_array_equal(result["r"], np.array([10.0, 0.0, 0.0]))
    np.testing.assert_array_equal(result["v"], np.array([-10.0, 5.0, 0.0]))
    assert np.isclose(result["distance"][0], 10.0)
    np.testing.assert_allclose(result["omega_los"], np.array([0.0, 0.0, 0.5]))
