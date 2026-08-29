import numpy as np
from net_interception_env.mechanics.guidance_algorithm.kinematics import compute_relative_kinematics

def test_compute_relative_kinematics_standard():
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([10.0, 0.0, 0.0])
    p_t = np.array([10.0, 0.0, 0.0])
    v_t = np.array([0.0, 5.0, 0.0])

    result = compute_relative_kinematics(p_i, v_i, p_t, v_t)

    np.testing.assert_array_equal(result['r'], np.array([10.0, 0.0, 0.0]))
    np.testing.assert_array_equal(result['v'], np.array([-10.0, 5.0, 0.0]))
    np.testing.assert_array_equal(result['v_rel'], np.array([10.0, -5.0, 0.0]))
    assert np.isclose(result['distance'][0], 10.0)

    # omega_los = (r x v) / R^2
    # r x v = [10, 0, 0] x [-10, 5, 0] = [0, 0, 50]
    # omega_los = [0, 0, 50] / 100 = [0, 0, 0.5]
    np.testing.assert_allclose(result['omega_los'], np.array([0.0, 0.0, 0.5]))

def test_compute_relative_kinematics_zero_distance():
    p_i = np.array([5.0, 5.0, 5.0])
    v_i = np.array([10.0, 0.0, 0.0])
    p_t = np.array([5.0, 5.0, 5.0])
    v_t = np.array([10.0, 0.0, 0.0])

    result = compute_relative_kinematics(p_i, v_i, p_t, v_t)
    
    assert np.isclose(result['distance'][0], 0.0)
    np.testing.assert_array_equal(result['omega_los'], np.zeros(3))
