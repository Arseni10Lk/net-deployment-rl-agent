import numpy as np
from net_interception_env.mechanics.guidance_algorithm.nominal_guidance import compute_bgpn_acceleration

def test_compute_bgpn_acceleration_no_target_accel():
    v_rel = np.array([10.0, -5.0, 0.0])
    omega_los = np.array([0.0, 0.0, 0.5])
    a_t = np.zeros(3)
    r = np.array([10.0, 0.0, 0.0])
    
    a_cmd = compute_bgpn_acceleration(v_rel, omega_los, a_t, r, N=4.0)
    
    # a_png = 4.0 * cross([10, -5, 0], [0, 0, 0.5])
    # cross = [-2.5, -5.0, 0.0]
    # a_png = [-10.0, -20.0, 0.0]
    expected_a_cmd = np.array([-10.0, -20.0, 0.0])
    np.testing.assert_allclose(a_cmd, expected_a_cmd)

def test_compute_bgpn_acceleration_with_target_accel():
    v_rel = np.array([10.0, 0.0, 0.0])
    omega_los = np.array([0.0, 0.0, 0.0]) # Pure head-on/tail-chase
    a_t = np.array([2.0, 4.0, 0.0])
    r = np.array([10.0, 0.0, 0.0])
    
    a_cmd = compute_bgpn_acceleration(v_rel, omega_los, a_t, r, N=4.0)
    
    # los_unit = [1, 0, 0]
    # a_t_parallel = [2, 0, 0]
    # a_t_perp = [0, 4, 0]
    # command = a_png (0) + (4/2) * [0, 4, 0] = [0, 8, 0]
    expected_a_cmd = np.array([0.0, 8.0, 0.0])
    np.testing.assert_allclose(a_cmd, expected_a_cmd)
