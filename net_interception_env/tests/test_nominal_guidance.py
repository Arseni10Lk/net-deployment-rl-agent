import numpy as np
from net_interception_env.mechanics.guidance_algorithm.nominal_guidance import compute_bgpn_acceleration

def test_compute_bgpn_acceleration_no_target_accel():
    v_c = np.array([-10.0, 5.0, 0.0])
    omega_los = np.array([0.0, 0.0, 0.5])
    a_t = np.zeros(3)
    r = np.array([10.0, 0.0, 0.0])
    
    a_cmd = compute_bgpn_acceleration(v_c, omega_los, a_t, r, N=4.0, k_axial=2.0, v_close_des=15.0)
    
    # a_png = 4.0 * cross([-10, 5, 0], [0, 0, 0.5]) = 4.0 * [2.5, 5.0, 0.0] = [10.0, 20.0, 0.0]
    # closing speed = np.dot([10, -5, 0], [1, 0, 0]) = 10.0
    # a_axial = 2.0 * (15 - 10) * [1, 0, 0] = [10.0, 0.0, 0.0]
    # total = [20.0, 20.0, 0.0]
    expected_a_cmd = np.array([20.0, 20.0, 0.0])
    np.testing.assert_allclose(a_cmd, expected_a_cmd)
