import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from net_interception_env.mechanics.physics import (
    Quadrotor6DOF,
    QuadrotorParams,
    SO3GeometricController,
    quat_to_rot_matrix,
)
from net_interception_env.mechanics.guidance_algorithm.guidance_controller import (
    HOCBFGuidanceController,
)
from net_interception_env.dataset_loader import (
    load_neurobem_csv,
    load_midair_trajectory,
)


def quat_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Converts quaternion [qw, qx, qy, qz] to roll, pitch, yaw in degrees (FLU frame)."""
    qw, qx, qy, qz = q
    # Roll (x-axis)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def run_physics_simulation(
    dataset_path: str | None = None,
    duration: float = 10.0,
    output_dir: str = "logs",
) -> tuple[dict[str, np.ndarray], float]:
    """
    Simulates the 17-state 6-DOF Quadrotor guided by HOCBF-PN against a target trajectory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Physical parameters (Mass = 1.0 kg, arm length = 15cm)
    params = QuadrotorParams(
        mass=1.0,
        arm_length=0.15,
        tau_m=0.03,  # 30 ms motor lag
        d_lin=np.array([0.1, 0.1, 0.2]),
        d_quad=np.array([0.05, 0.05, 0.15]),
    )
    controller = SO3GeometricController(
        mass=params.mass,
        J=params.inertia,
        g=params.gravity,
        k_R=12.0,
        k_omega=2.0,
    )
    drone = Quadrotor6DOF(params=params, controller=controller)

    guidance = HOCBFGuidanceController(
        d_min=1.5,
        k1=2.0,
        k2=4.0,
        N=4.0,
        a_max_xy=25.0,
        a_max_z=15.0,
        solver="clarabel",
    )

    if dataset_path:
        print(f"Loading target trajectory from: {dataset_path}")
        if dataset_path.endswith(".csv"):
            traj = load_neurobem_csv(dataset_path)
        elif dataset_path.endswith(".hdf5"):
            import h5py

            with h5py.File(dataset_path, "r") as f:
                traj_names = [k for k in f.keys() if k.startswith("trajectory_")]
            traj = load_midair_trajectory(dataset_path, traj_names[0])
        else:
            raise ValueError("Unsupported dataset format. Use .csv or .hdf5")

        dt = traj.dt
        total_steps = len(traj.time)
        max_steps = int(duration / dt)
        start_step = int(30.0 / dt) if total_steps > int(45.0 / dt) else 0

        steps = min(max_steps, total_steps - start_step)
        t_slice = traj.time[start_step : start_step + steps]
        p_t_slice = traj.position[start_step : start_step + steps]
        v_t_slice = traj.velocity[start_step : start_step + steps]
        a_t_slice = traj.acceleration[start_step : start_step + steps]

        # Initial conditions: interceptor spawns behind target
        v_init_dir = v_t_slice[0] / (np.linalg.norm(v_t_slice[0]) + 1e-6)
        p_init = p_t_slice[0] - v_init_dir * 12.0
        p_init[2] -= 2.0
        v_init = v_t_slice[0] * 0.4
    else:
        print("No dataset provided. Simulating high-speed dynamic weave target.")
        dt = 0.005  # 200 Hz
        steps = int(duration / dt)
        t_slice = np.linspace(0, duration, steps)

        # Dynamic slalom target
        p_t_slice = np.column_stack(
            [
                10.0 + 4.0 * t_slice,
                8.0 * np.sin(0.8 * t_slice),
                5.0 + 3.0 * np.cos(0.5 * t_slice),
            ]
        )
        v_t_slice = np.gradient(p_t_slice, dt, axis=0)
        a_t_slice = np.gradient(v_t_slice, dt, axis=0)

        p_init = np.array([0.0, 0.0, 3.0])
        v_init = np.array([2.0, 0.0, 0.0])

    drone.reset(pos=p_init, vel=v_init)

    # History buffers
    history = {
        "time": [],
        "p_i": [],
        "v_i": [],
        "quat": [],
        "euler": [],  # roll, pitch, yaw [deg]
        "omega": [],
        "motor_speeds": [],
        "p_t": [],
        "v_t": [],
        "a_cmd": [],
        "thrust": [],
        "tau": [],
        "f_drag": [],
        "actual_accel": [],
    }

    print(
        f"Simulating 17-state 6-DOF physics at {1/dt:.1f}Hz for {steps} steps ({steps*dt:.1f}s)..."
    )

    prev_vel = drone.state.vel.copy()

    for i in range(steps):
        p_t = p_t_slice[i]
        v_t = v_t_slice[i]
        a_t = a_t_slice[i]

        p_i = drone.state.pos
        v_i = drone.state.vel

        # 1. HOCBF-PN guidance acceleration
        a_cmd = guidance.compute_command(p_i, v_i, p_t, v_t, a_t)

        # Point yaw towards target
        r_rel = p_t - p_i
        yaw_target = float(np.arctan2(r_rel[1], r_rel[0]))

        # 2. Physics 6-DOF step with Drag Feedforward and SO(3) controller
        new_state, telem = drone.step_acceleration(
            a_cmd=a_cmd,
            dt=dt,
            yaw_desired=yaw_target,
        )

        # Numerical acceleration achieved
        actual_a = (new_state.vel - prev_vel) / dt
        prev_vel = new_state.vel.copy()

        roll, pitch, yaw = quat_to_euler_deg(new_state.quat)

        history["time"].append(t_slice[i])
        history["p_i"].append(new_state.pos.copy())
        history["v_i"].append(new_state.vel.copy())
        history["quat"].append(new_state.quat.copy())
        history["euler"].append([roll, pitch, yaw])
        history["omega"].append(new_state.omega.copy())
        history["motor_speeds"].append(new_state.motor_speeds.copy())
        history["p_t"].append(p_t.copy())
        history["v_t"].append(v_t.copy())
        history["a_cmd"].append(a_cmd.copy())
        history["thrust"].append(telem["T_des"])
        history["tau"].append(telem["tau_des"].copy())
        history["f_drag"].append(telem["f_drag_world"].copy())
        history["actual_accel"].append(actual_a.copy())

    for k in history:
        history[k] = np.array(history[k])

    return history, dt


def plot_physics_telemetry(
    history: dict[str, np.ndarray], output_file: str = "logs/physics_telemetry.png"
):
    """
    Renders comprehensive 6-DOF physics telemetry:
    - 3D Trajectory with body orientation frames
    - Euler angles (Roll/Pitch tilt during maneuvers)
    - 4 Motor RPMs showing first-order spool lag and differential mixing
    - Forces: Total thrust vs Anisotropic aerodynamic drag
    """
    t = history["time"] - history["time"][0]

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "6-DOF Interceptor Physics Telemetry (SO(3) Geometric Control + 17-State RK4)",
        fontsize=16,
        fontweight="bold",
    )

    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    p_i = history["p_i"]
    p_t = history["p_t"]

    ax1.plot(
        p_t[:, 0],
        p_t[:, 1],
        p_t[:, 2],
        "r--",
        linewidth=1.8,
        label="Target Trajectory",
    )
    ax1.plot(
        p_i[:, 0],
        p_i[:, 1],
        p_i[:, 2],
        "b-",
        linewidth=2.2,
        label="6-DOF Interceptor",
    )

    # Draw quadrotor attitude frames at regular intervals
    step_stride = max(1, len(p_i) // 12)
    arm_len = 0.8  # visualization scale
    for idx in range(0, len(p_i), step_stride):
        pos = p_i[idx]
        R = quat_to_rot_matrix(history["quat"][idx])
        # Body X (Forward) in Red, Body Y (Left) in Green, Body Z (Up/Thrust) in Blue
        ax1.quiver(
            pos[0],
            pos[1],
            pos[2],
            R[0, 0],
            R[1, 0],
            R[2, 0],
            length=arm_len,
            color="red",
            alpha=0.7,
        )
        ax1.quiver(
            pos[0],
            pos[1],
            pos[2],
            R[0, 1],
            R[1, 1],
            R[2, 1],
            length=arm_len,
            color="green",
            alpha=0.7,
        )
        ax1.quiver(
            pos[0],
            pos[1],
            pos[2],
            R[0, 2],
            R[1, 2],
            R[2, 2],
            length=arm_len * 1.5,
            color="blue",
            alpha=0.9,
        )

    ax1.set_title("3D Trajectory & Attitude Frames (RGB = FLU)")
    ax1.set_xlabel("X (East) [m]")
    ax1.set_ylabel("Y (North) [m]")
    ax1.set_zlabel("Z (Up) [m]")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # 2. Attitude Angles (Roll, Pitch, Yaw)
    ax2 = fig.add_subplot(2, 3, 2)
    euler = history["euler"]
    ax2.plot(t, euler[:, 0], "r-", label="Roll (φ)")
    ax2.plot(t, euler[:, 1], "g-", label="Pitch (θ)")
    ax2.plot(t, euler[:, 2], "b-", label="Yaw (ψ)")
    ax2.set_title("Euler Attitude (Bank Angles)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angle [deg]")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 3. Motor Speeds (4 Rotors)
    ax3 = fig.add_subplot(2, 3, 3)
    rpm = history["motor_speeds"] * (60.0 / (2.0 * np.pi))  # rad/s to RPM
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["M1 (BR)", "M2 (FR)", "M3 (BL)", "M4 (FL)"]
    for m in range(4):
        ax3.plot(t, rpm[:, m], color=colors[m], label=labels[m], alpha=0.85)
    ax3.set_title("Rotor Speeds (Spool Lag & Differential Torque)")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Speed [RPM]")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", ncol=2)

    # 4. Velocities
    ax4 = fig.add_subplot(2, 3, 4)
    v_i_mag = np.linalg.norm(history["v_i"], axis=1)
    v_t_mag = np.linalg.norm(history["v_t"], axis=1)
    ax4.plot(t, v_i_mag, "b-", linewidth=2.0, label="Interceptor Speed")
    ax4.plot(t, v_t_mag, "r--", linewidth=1.8, label="Target Speed")
    ax4.set_title("Linear Speeds (World Frame)")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Speed [m/s]")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    # 5. Forces: Thrust vs Drag
    ax5 = fig.add_subplot(2, 3, 5)
    drag_mag = np.linalg.norm(history["f_drag"], axis=1)
    ax5.plot(t, history["thrust"], "m-", linewidth=2.0, label="Rotor Thrust (T)")
    ax5.plot(
        t,
        drag_mag,
        "k--",
        linewidth=1.8,
        label="Anisotropic Drag (|F_drag|)",
    )
    ax5.axhline(
        1.0 * 9.81,
        color="gray",
        linestyle=":",
        label="Hover Weight (mg = 9.81 N)",
    )
    ax5.set_title("Forces: Thrust & Drag Feedforward")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Force [N]")
    ax5.grid(True, linestyle="--", alpha=0.6)
    ax5.legend()

    # 6. Interceptor-Target Distance
    ax6 = fig.add_subplot(2, 3, 6)
    dist = np.linalg.norm(p_t - p_i, axis=1)
    ax6.plot(t, dist, "indigo", linewidth=2.0, label="Relative Separation")
    ax6.axhline(
        1.5,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="HOCBF Safety Limit (1.5 m)",
    )
    ax6.set_title("Pursuit Range & Safety Barrier")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Distance [m]")
    ax6.grid(True, linestyle="--", alpha=0.6)
    ax6.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    print(f"Telemetry dashboard saved to: {output_file}")


def render_physics_animation(
    history: dict[str, np.ndarray],
    dt: float,
    output_gif: str = "logs/physics_animation.gif",
    fps: int = 25,
):
    """
    Renders an animated GIF showing the 6-DOF quadrotor banking and pitching in 3D
    with its 4 rotor arms in the X-configuration.
    """
    p_i = history["p_i"]
    p_t = history["p_t"]
    quats = history["quat"]

    # Subsample to target fps
    sim_fps = 1.0 / dt
    step_skip = max(1, int(round(sim_fps / fps)))

    p_i_sub = p_i[::step_skip]
    p_t_sub = p_t[::step_skip]
    quats_sub = quats[::step_skip]
    n_frames = len(p_i_sub)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Spatial bounding box
    all_pos = np.vstack([p_i, p_t])
    min_xyz = np.min(all_pos, axis=0) - 2.0
    max_xyz = np.max(all_pos, axis=0) + 2.0

    # Static trajectory lines
    ax.plot(
        p_t[:, 0],
        p_t[:, 1],
        p_t[:, 2],
        "r--",
        linewidth=1.2,
        alpha=0.6,
        label="Target Track",
    )
    ax.plot(
        p_i[:, 0],
        p_i[:, 1],
        p_i[:, 2],
        "b-",
        linewidth=1.2,
        alpha=0.6,
        label="Interceptor Track",
    )

    # Dynamic elements
    (interceptor_point,) = ax.plot([], [], [], "bo", markersize=6, label="Center of Mass")
    (target_point,) = ax.plot([], [], [], "ro", markersize=7, label="Target Drone")

    # Quadrotor X-frame arms (Front-Right to Back-Left, Front-Left to Back-Right)
    (arm1_line,) = ax.plot([], [], [], "k-", linewidth=1.0)
    (arm2_line,) = ax.plot([], [], [], "k-", linewidth=1.0)
    # Rotor disks (markers at the 4 motor locations)
    (rotors_scatter,) = ax.plot([], [], [], "c^", markersize=1, label="Propellers")
    # Thrust vector (quiver)
    thrust_quiver = [None]

    # Arm geometry in FLU body frame (45 deg arms, L = 0.15m scaled for visibility)
    vis_scale = 1.2  # Scaled slightly so attitude is clearly visible in the wide 3D plot
    arm_L = 0.15 * vis_scale
    l_arm = arm_L * (np.sqrt(2.0) / 2.0)
    # 1=BR, 2=FR, 3=BL, 4=FL
    motor_body_pos = np.array(
        [
            [-l_arm, -l_arm, 0.0],  # M1 (BR)
            [l_arm, -l_arm, 0.0],  # M2 (FR)
            [-l_arm, l_arm, 0.0],  # M3 (BL)
            [l_arm, l_arm, 0.0],  # M4 (FL)
        ]
    )

    ax.set_xlim([min_xyz[0], max_xyz[0]])
    ax.set_ylim([min_xyz[1], max_xyz[1]])
    ax.set_zlim([min_xyz[2], max_xyz[2]])
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Y (North) [m]")
    ax.set_zlabel("Z (Up) [m]")
    ax.set_title("6-DOF Quadrotor Banking Dynamics (X-Frame, FLU)", fontweight="bold")
    ax.legend(loc="upper left")

    def init():
        interceptor_point.set_data([], [])
        interceptor_point.set_3d_properties([])
        target_point.set_data([], [])
        target_point.set_3d_properties([])
        arm1_line.set_data([], [])
        arm1_line.set_3d_properties([])
        arm2_line.set_data([], [])
        arm2_line.set_3d_properties([])
        rotors_scatter.set_data([], [])
        rotors_scatter.set_3d_properties([])
        return (
            interceptor_point,
            target_point,
            arm1_line,
            arm2_line,
            rotors_scatter,
        )

    def update(frame):
        pos_i = p_i_sub[frame]
        pos_t = p_t_sub[frame]
        q = quats_sub[frame]
        R = quat_to_rot_matrix(q)

        # Center points
        interceptor_point.set_data([pos_i[0]], [pos_i[1]])
        interceptor_point.set_3d_properties([pos_i[2]])

        target_point.set_data([pos_t[0]], [pos_t[1]])
        target_point.set_3d_properties([pos_t[2]])

        # Rotate motor positions to world frame
        m_world = pos_i + (R @ motor_body_pos.T).T

        # Arm 1 connects M4(FL) to M1(BR)
        arm1_line.set_data([m_world[3, 0], m_world[0, 0]], [m_world[3, 1], m_world[0, 1]])
        arm1_line.set_3d_properties([m_world[3, 2], m_world[0, 2]])

        # Arm 2 connects M2(FR) to M3(BL)
        arm2_line.set_data([m_world[1, 0], m_world[2, 0]], [m_world[1, 1], m_world[2, 1]])
        arm2_line.set_3d_properties([m_world[1, 2], m_world[2, 2]])

        # Rotors
        rotors_scatter.set_data(m_world[:, 0], m_world[:, 1])
        rotors_scatter.set_3d_properties(m_world[:, 2])

        # Thrust vector (Blue arrow pointing out of drone top along body +Z)
        if thrust_quiver[0] is not None:
            thrust_quiver[0].remove()
        z_thrust = R[:, 2] * 1.5
        thrust_quiver[0] = ax.quiver(
            pos_i[0],
            pos_i[1],
            pos_i[2],
            z_thrust[0],
            z_thrust[1],
            z_thrust[2],
            color="blue",
            linewidth=2.0,
            arrow_length_ratio=0.3,
        )

        return (
            interceptor_point,
            target_point,
            arm1_line,
            arm2_line,
            rotors_scatter,
        )

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=False
    )

    print(f"Rendering animation GIF ({n_frames} frames)...")
    anim.save(output_gif, writer="pillow", fps=fps)
    plt.close()
    print(f"Animation saved to: {output_gif}")


def main():
    parser = argparse.ArgumentParser(description="6-DOF Quadrotor Physics Visualizer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_data/neurobem/processed_data/merged_2021-02-23-19-04-50_seg_2.csv",
        help="Path to real-world dataset (.csv or .hdf5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip animated GIF generation",
    )
    args = parser.parse_args()

    # Use dataset if it exists, otherwise synthetic
    dataset = args.dataset if os.path.exists(args.dataset) else None

    history, dt = run_physics_simulation(
        dataset_path=dataset, duration=args.duration, output_dir="logs"
    )

    plot_physics_telemetry(history, output_file="logs/physics_telemetry.png")

    if not args.no_gif:
        render_physics_animation(
            history, dt, output_gif="logs/physics_animation.gif", fps=20
        )


if __name__ == "__main__":
    main()
