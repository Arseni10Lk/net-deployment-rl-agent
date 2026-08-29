import numpy as np
from net_interception_env.mechanics.guidance_algorithm.guidance_controller import HOCBFGuidanceController

def test_guidance_controller_far_field():
    """Test when the drone is far away, the CBF should not interfere, and it should output B-GPN."""
    controller = HOCBFGuidanceController(d_min=1.5, N=4.0, a_max_xy=20.0, a_max_z=10.0, solver='clarabel')
    
    # Far away (30m)
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([10.0, 0.0, 0.0])
    p_t = np.array([30.0, 5.0, 0.0])
    v_t = np.array([5.0, 0.0, 0.0])
    a_t = np.array([0.0, 0.0, 0.0])
    
    a_cmd = controller.compute_command(p_i, v_i, p_t, v_t, a_t)
    
    assert isinstance(a_cmd, np.ndarray)
    
    from net_interception_env.mechanics.guidance_algorithm.kinematics import compute_relative_kinematics
    from net_interception_env.mechanics.guidance_algorithm.nominal_guidance import compute_bgpn_acceleration
    
    kin = compute_relative_kinematics(p_i, v_i, p_t, v_t)
    a_nom = compute_bgpn_acceleration(kin['v_rel'], kin['omega_los'], a_t, kin['r'], N=4.0)
    
    np.testing.assert_allclose(a_cmd, a_nom, atol=1e-3)

def test_guidance_controller_near_field_braking():
    """Test when the drone approaches the safety horizon, the CBF should force braking."""
    controller = HOCBFGuidanceController(d_min=1.5, N=4.0, a_max_xy=20.0, a_max_z=10.0, solver='clarabel')
    
    # Approaching the braking horizon (3m away, closing at 5m/s)
    # The CBF should smoothly engage to prevent breaching d_min=1.5m
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([5.0, 0.0, 0.0])
    p_t = np.array([3.0, 0.0, 0.0])
    v_t = np.array([0.0, 0.0, 0.0])
    a_t = np.array([0.0, 0.0, 0.0])
    
    a_cmd = controller.compute_command(p_i, v_i, p_t, v_t, a_t)
    
    # The nominal guidance would output 0 (perfect tail chase)
    # BUT the safety filter should override it and command braking in the x direction
    
    assert a_cmd[0] < -5.0 # Should be braking!

