import numpy as np
from net_interception_env.mechanics.guidance_algorithm.cbf_filter import compute_cbf_matrices

def test_compute_cbf_matrices_standard():
    r = np.array([2.0, 0.0, 0.0])
    v = np.array([-5.0, 0.0, 0.0])
    a_t = np.array([0.0, 0.0, 0.0])
    d_min = 1.0
    k1 = 1.0
    k2 = 1.0
    
    A_cbf, b_cbf = compute_cbf_matrices(r, v, a_t, d_min=d_min, k1=k1, k2=k2)
    
    # distance_sq = 4.0
    # h_x = 4.0 - 1.0 = 3.0
    # h_dot_x = 2 * r_dot_v = 2 * (-10.0) = -20.0
    # f_drift = 2 * v_sq + 2 * r_dot_at = 2 * 25.0 + 0 = 50.0
    # A_cbf = 2 * r = [4.0, 0.0, 0.0]
    # b_cbf = f_drift + (k1+k2)*h_dot_x + k1*k2*h_x
    # b_cbf = 50.0 + (2.0)*(-20.0) + (1.0)*3.0 = 50.0 - 40.0 + 3.0 = 13.0
    
    np.testing.assert_allclose(A_cbf, np.array([[4.0, 0.0, 0.0]]))
    assert np.isclose(b_cbf, 13.0)

def test_compute_cbf_matrices_zero_distance():
    r = np.zeros(3)
    v = np.array([5.0, 0.0, 0.0])
    a_t = np.zeros(3)
    
    A_cbf, b_cbf = compute_cbf_matrices(r, v, a_t)
    
    np.testing.assert_allclose(A_cbf, np.zeros((1, 3)))
    assert np.isclose(b_cbf, 0.0)
