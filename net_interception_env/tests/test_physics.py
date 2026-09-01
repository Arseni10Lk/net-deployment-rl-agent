import numpy as np

from net_interception_env.mechanics.physics import (
    Quadrotor6DOF,
    QuadrotorParams,
    QuadrotorState,
    SO3GeometricController,
    compute_derivatives,
    integrate_rk4,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    so3_attitude_error,
    thrust_torques_to_motor_speeds,
)


def test_hover_invariant():
    """
    Sanity Check 1: The Hover Invariant
    Condition: a_cmd = [0, 0, 0]
    Expected: F_des = [0, 0, 9.81*m], motors settle at Omega = sqrt(mg / 4kf), a = 0.
    """
    params = QuadrotorParams(mass=1.2, kf=1.5e-5)
    controller = SO3GeometricController(mass=params.mass, J=params.inertia, g=params.gravity)
    drone = Quadrotor6DOF(params=params, controller=controller)

    # Commanded zero acceleration (hover)
    a_cmd = np.zeros(3)
    F_des = controller.compute_desired_force(a_cmd, f_drag=np.zeros(3))

    # Desired force should exactly cancel gravity
    expected_force = np.array([0.0, 0.0, params.mass * abs(params.gravity[2])])
    assert np.allclose(F_des, expected_force, atol=1e-6)

    # Initial state is at rest with hover motor speeds
    hover_speed = params.hover_motor_speed
    assert np.allclose(drone.state.motor_speeds, hover_speed, atol=1e-6)

    # Derivative at hover equilibrium should have zero linear acceleration and zero angular acceleration
    deriv, info = compute_derivatives(drone.state, np.full(4, hover_speed), params)
    pos_dot = deriv[0:3]
    vel_dot = deriv[3:6]
    omega_dot = deriv[10:13]

    assert np.allclose(pos_dot, np.zeros(3), atol=1e-6)
    assert np.allclose(vel_dot, np.zeros(3), atol=1e-6)
    assert np.allclose(omega_dot, np.zeros(3), atol=1e-6)


def test_freefall_and_terminal_velocity():
    """
    Sanity Check 2: Freefall & Terminal Velocity Check
    Condition: Motors turned off (Omega_cmd = 0).
    Expected: Initial acceleration is [0, 0, -9.81]. As -vz increases,
              drag increases until F_drag_z = mg, reaching terminal velocity.
    """
    params = QuadrotorParams(mass=1.0, d_lin=np.array([0.1, 0.1, 0.2]), d_quad=np.array([0.05, 0.05, 0.15]))
    drone = Quadrotor6DOF(params=params)
    drone.reset(pos=[0, 0, 100], vel=[0, 0, 0], motor_speeds=[0, 0, 0, 0])

    # Initial step: zero velocity, zero motor speeds -> pure gravitational acceleration
    deriv, _ = compute_derivatives(drone.state, np.zeros(4), params)
    vel_dot_0 = deriv[3:6]
    assert np.allclose(vel_dot_0, params.gravity, atol=1e-6)

    # Simulate free fall for 15 seconds to reach terminal velocity
    dt = 0.01
    for _ in range(1500):
        drone.step_motor_speeds(np.zeros(4), dt=dt)

    # At terminal velocity, vz < 0 and acceleration should be near zero
    final_vz = drone.state.vel[2]
    assert final_vz < -5.0  # falling fast

    deriv_final, _ = compute_derivatives(drone.state, np.zeros(4), params)
    accel_final = deriv_final[3:6]
    assert np.isclose(accel_final[2], 0.0, atol=0.05)


def test_flu_mixing_matrix_signs():
    """
    Sanity Check 3: FLU Coordinate Mixing Matrix Verification
    Condition: Command a pure positive Roll torque (tau_phi > 0).
    Expected: By the Right-Hand Rule in FLU (X=Forward, Y=Left, Z=Up),
              positive roll dips the right side and lifts the left side.
              Left motors 3 (BL) and 4 (FL) must increase RPM,
              while Right motors 1 (BR) and 2 (FR) must decrease RPM.
    """
    params = QuadrotorParams()
    hover_thrust = params.mass * abs(params.gravity[2])

    # Command hover thrust + positive roll torque (tau_phi > 0)
    tau_roll_positive = np.array([0.05, 0.0, 0.0])
    omega_cmd = thrust_torques_to_motor_speeds(
        T=hover_thrust,
        tau=tau_roll_positive,
        M_inv=params.inv_mixing_matrix,
        omega_min=params.omega_min,
        omega_max=params.omega_max,
    )

    hover_speed = params.hover_motor_speed

    # Motors: 1=BR, 2=FR (Right side) -> should be LESS than hover
    # Motors: 3=BL, 4=FL (Left side)  -> should be GREATER than hover
    assert omega_cmd[0] < hover_speed, "Motor 1 (BR) should decrease for positive roll"
    assert omega_cmd[1] < hover_speed, "Motor 2 (FR) should decrease for positive roll"
    assert omega_cmd[2] > hover_speed, "Motor 3 (BL) should increase for positive roll"
    assert omega_cmd[3] > hover_speed, "Motor 4 (FL) should increase for positive roll"


def test_first_order_motor_lag():
    """
    Sanity Check 4: First-Order Motor Lag Time-Constant
    Condition: A step input commands motors from 0 -> 1000 rad/s.
    Expected: Simulated Omega reaches approx 63.2% (1 - 1/e) of the step at t = tau_m.
    """
    tau_m = 0.03
    params = QuadrotorParams(tau_m=tau_m)
    state = QuadrotorState(motor_speeds=np.zeros(4))

    target_omega = 1000.0
    omega_cmd = np.full(4, target_omega)

    dt = 0.001
    steps = int(round(tau_m / dt))  # Exactly tau_m seconds

    for _ in range(steps):
        state = integrate_rk4(state, omega_cmd, params, dt=dt)

    expected_omega = target_omega * (1.0 - np.exp(-1.0))  # ~632.12 rad/s
    assert np.allclose(state.motor_speeds, expected_omega, rtol=0.01)


def test_so3_attitude_shortest_path():
    """
    Sanity Check 5: SO(3) Attitude Shortest-Path Tracking
    Condition: Perturb orientation across boundaries.
    Expected: SO(3) attitude error takes minimal path without double-cover 360-degree unwinding.
    """
    # Create two orientations 10 degrees apart
    theta1 = np.deg2rad(175.0)
    theta2 = np.deg2rad(-175.0)  # Total physical difference is 10 degrees across the +/- 180 boundary

    R1 = np.array([
        [np.cos(theta1), -np.sin(theta1), 0.0],
        [np.sin(theta1), np.cos(theta1), 0.0],
        [0.0, 0.0, 1.0],
    ])
    R2 = np.array([
        [np.cos(theta2), -np.sin(theta2), 0.0],
        [np.sin(theta2), np.cos(theta2), 0.0],
        [0.0, 0.0, 1.0],
    ])

    # In quaternions, q and -q can cause issues, but in SO(3):
    e_R = so3_attitude_error(R1, R2)
    # The error magnitude should correspond to sin(10 deg) ~ 0.1736, not sin(350 deg)
    assert np.isclose(abs(e_R[2]), np.sin(np.deg2rad(10.0)), atol=1e-4)


def test_quaternion_normalization_conservation():
    """
    Sanity Check 6: Quaternion Normalization Conservation
    Condition: Continuous numerical integration over 10,000 steps with aggressive angular rates.
    Expected: The norm of the quaternion state strictly remains 1.0.
    """
    params = QuadrotorParams()
    # High tumble rates
    state = QuadrotorState(
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([5.0, -8.0, 12.0]),
    )

    dt = 0.005
    for _ in range(10000):
        state = integrate_rk4(state, np.full(4, params.hover_motor_speed), params, dt=dt)

    assert np.isclose(np.linalg.norm(state.quat), 1.0, atol=1e-12)
