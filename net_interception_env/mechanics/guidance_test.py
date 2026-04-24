import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from net_interception_env.mechanics import Constraints
from net_interception_env.mechanics import Pro_Nav_logic as tpn


def test_pro_nav():
    # Initial conditions
    pursuer_pos = np.random.rand(3)*200
    pursuer_vel = np.random.rand(3)*Constraints.MAX_UAV_SPEED

    target_pos = np.random.rand(3)*200
    target_vel = np.random.rand(3)*Constraints.MAX_TARGET_SPEED
    target_acc = np.array([0.0, 0.0, 0.0])

    # Tracking for plotting
    p_hist = [pursuer_pos.copy()]
    t_hist = [target_pos.copy()]

    max_steps = 50000

    # Simulation loop
    for step in range(max_steps):
        # 1. Calculate Pro-Nav acceleration
        p_acc = tpn.get_tpn_acceleration(pursuer_vel, target_vel, pursuer_pos, target_pos)
        target_acc = tpn.target_accelaration(target_acc)

        # 2. Update Kinematics
        pursuer_pos, pursuer_vel, _ = tpn.get_new_location(
            p_acc, pursuer_vel, pursuer_pos, Constraints.MAX_UAV_SPEED
        )
        target_pos, target_vel, _ = tpn.get_new_location(
            target_acc, target_vel, target_pos, Constraints.MAX_TARGET_SPEED
        )

        # 3. Store history
        p_hist.append(pursuer_pos.copy())
        t_hist.append(target_pos.copy())

        # 4. Check for intercept (e.g., within 0.5 meters)
        if np.linalg.norm(target_pos - pursuer_pos) < 0.5:
            print(f"Target intercepted at step {step}!")
            break

    # Visualization
    p_hist = np.array(p_hist)
    t_hist = np.array(t_hist)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], label='Pursuer Trajectory', color='blue')
    ax.plot(t_hist[:, 0], t_hist[:, 1], t_hist[:, 2], label='Target Trajectory', color='red')

    ax.scatter(p_hist[-1, 0], p_hist[-1, 1], p_hist[-1, 2], color='blue', marker='o')
    ax.scatter(t_hist[-1, 0], t_hist[-1, 1], t_hist[-1, 2], color='red', marker='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    test_pro_nav()