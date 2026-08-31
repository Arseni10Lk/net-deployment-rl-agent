import numpy as np
from net_interception_env.mechanics.guidance_algorithm.guidance_controller import (
    HOCBFGuidanceController,
)


def test_guidance_controller_far_field():
    controller = HOCBFGuidanceController(
        d_min=1.5, N=4.0, a_max_xy=20.0, a_max_z=10.0, solver="clarabel"
    )
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([10.0, 0.0, 0.0])
    p_t = np.array([30.0, 5.0, 0.0])
    v_t = np.array([5.0, 0.0, 0.0])
    a_t = np.array([0.0, 0.0, 0.0])

    a_cmd = controller.compute_command(p_i, v_i, p_t, v_t, a_t)
    assert isinstance(a_cmd, np.ndarray)


def test_guidance_controller_near_field_braking():
    controller = HOCBFGuidanceController(
        d_min=1.5, k1=1.5, k2=3.0, N=4.0, a_max_xy=20.0, a_max_z=10.0, solver="clarabel"
    )
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([5.0, 0.0, 0.0])
    p_t = np.array([3.0, 0.0, 0.0])
    v_t = np.array([0.0, 0.0, 0.0])
    a_t = np.array([0.0, 0.0, 0.0])
    a_cmd = controller.compute_command(p_i, v_i, p_t, v_t, a_t)
    assert a_cmd[0] < -5.0  # Should be braking!


def test_guidance_controller_points_towards_target():
    controller = HOCBFGuidanceController(
        d_min=1.5, N=4.0, a_max_xy=20.0, a_max_z=10.0, solver="clarabel"
    )
    p_i = np.array([0.0, 0.0, 0.0])
    p_t = np.array([30.0, 0.0, 0.0])
    v_i = np.array([-10.0, 0.0, 0.0])
    v_t = np.array([0.0, 0.0, 0.0])
    a_t = np.array([0.0, 0.0, 0.0])
    a_cmd = controller.compute_command(p_i, v_i, p_t, v_t, a_t)
    assert (
        a_cmd[0] > 5.0
    ), f"Expected strong positive acceleration towards target, got {a_cmd[0]}"
