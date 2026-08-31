import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from net_interception_env.mechanics.guidance_algorithm.guidance_controller import (
    HOCBFGuidanceController,
)
from net_interception_env.dataset_loader import (
    load_neurobem_csv,
    load_midair_trajectory,
)


def run_simulation_and_visualize(
    save_static: bool = True, save_video: bool = True, dataset_path: str = None
):
    """
    Simulates a 3D intercept engagement using the HOCBF-PN controller and
    generates a static trajectory plot and/or an animated video (GIF).
    Saves outputs to the 'logs/' directory.
    """
    controller = HOCBFGuidanceController(
        d_min=1.5, k1=2.0, k2=4.0, N=4.0, a_max_xy=30.0, a_max_z=20.0, solver="clarabel"
    )

    # Check if a real dataset is provided
    if dataset_path:
        print(f"Loading real-world dataset from: {dataset_path}")
        if dataset_path.endswith(".csv"):
            traj = load_neurobem_csv(dataset_path)
        elif dataset_path.endswith(".hdf5"):
            # Just grab the first trajectory for visualization
            import h5py

            with h5py.File(dataset_path, "r") as f:
                traj_names = [k for k in f.keys() if k.startswith("trajectory_")]
            traj = load_midair_trajectory(dataset_path, traj_names[0])
        else:
            raise ValueError("Unsupported dataset format. Use .csv or .hdf5")

        dt = traj.dt
        steps = len(traj.time)

        max_duration = 15.0
        max_steps = int(max_duration / dt)
        start_step = int(30.0 / dt) if steps > int(45.0 / dt) else 0

        if steps > max_steps:
            print(
                "Slicing 15 seconds of flight starting at t=30s for aggressive visualization."
            )
            traj.time = traj.time[start_step : start_step + max_steps]
            traj.position = traj.position[start_step : start_step + max_steps]
            traj.velocity = traj.velocity[start_step : start_step + max_steps]
            traj.acceleration = traj.acceleration[start_step : start_step + max_steps]
            steps = max_steps

        # Spawn the interceptor 15 meters behind and 5 meters below the target's starting position
        start_dir = traj.velocity[0] / (np.linalg.norm(traj.velocity[0]) + 1e-6)
        p_i = traj.position[0] - start_dir * 15.0
        p_i[2] -= 5.0
        v_i = traj.velocity[0] * 0.5

    else:
        print("No dataset provided. Simulating dummy spiral target.")
        dt = 0.05  # 20 Hz
        max_time = 10.0
        steps = int(max_time / dt)
        p_i = np.array([0.0, 0.0, 0.0])
        v_i = np.array([15.0, 0.0, 0.0])

    p_i_hist = []
    p_t_hist = []

    print(f"Running physics simulation at {1/dt:.1f}Hz for {steps} steps...")
    time = 0.0
    for i in range(steps):
        if dataset_path:
            p_t = traj.position[i]
            v_t = traj.velocity[i]
            a_t = traj.acceleration[i]
        else:
            # Target spiral
            p_t = np.array(
                [30.0 + 3.0 * time, 10.0 * np.cos(time), 10.0 * np.sin(time)]
            )
            v_t = np.array([3.0, -10.0 * np.sin(time), 10.0 * np.cos(time)])
            a_t = np.array([0.0, -10.0 * np.cos(time), -10.0 * np.sin(time)])

        p_i_hist.append(p_i.copy())
        p_t_hist.append(p_t.copy())

        a_cmd = controller.compute_command(p_i, v_i, p_t, v_t, a_t)
        v_i += a_cmd * dt
        p_i += v_i * dt
        time += dt

    p_i_hist = np.array(p_i_hist)
    p_t_hist = np.array(p_t_hist)
    final_distance = np.linalg.norm(p_t_hist[-1] - p_i_hist[-1])

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    if save_static:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot3D(
            p_t_hist[:, 0],
            p_t_hist[:, 1],
            p_t_hist[:, 2],
            "r--",
            label="Target (Real-World Data)" if dataset_path else "Target (Spiral)",
        )
        ax.plot3D(
            p_i_hist[:, 0],
            p_i_hist[:, 1],
            p_i_hist[:, 2],
            "b-",
            linewidth=2,
            label="Interceptor (HOCBF-PN)",
        )
        ax.scatter(
            p_t_hist[-1, 0],
            p_t_hist[-1, 1],
            p_t_hist[-1, 2],
            color="red",
            s=100,
            marker="X",
        )
        ax.scatter(
            p_i_hist[-1, 0],
            p_i_hist[-1, 1],
            p_i_hist[-1, 2],
            color="blue",
            s=100,
            marker="o",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(
            f"HOCBF-PN Guidance\nFinal Distance: {final_distance:.2f}m (Safety Boundary: 1.5m)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig("logs/guidance_trajectory.png", dpi=300)
        print("Static trajectory saved to 'logs/guidance_trajectory.png'")
        plt.close(fig)

    if save_video:
        print("Generating animation GIF... (this may take a minute)")

        # Downsample the frames to 20Hz for the GIF to save memory and file size
        fps_target = 20
        step_stride = max(1, int((1.0 / fps_target) / dt))
        gif_frames = list(range(0, steps, step_stride))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Dynamically set axes limits based on the trajectory
        all_pts = np.vstack([p_i_hist, p_t_hist])
        ax.set_xlim([np.min(all_pts[:, 0]) - 5, np.max(all_pts[:, 0]) + 5])
        ax.set_ylim([np.min(all_pts[:, 1]) - 5, np.max(all_pts[:, 1]) + 5])
        ax.set_zlim([np.min(all_pts[:, 2]) - 5, np.max(all_pts[:, 2]) + 5])

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        (target_line,) = ax.plot([], [], [], "r--", label="Target")
        (interceptor_line,) = ax.plot(
            [], [], [], "b-", linewidth=2, label="Interceptor"
        )
        (target_point,) = ax.plot([], [], [], "ro", markersize=8)
        (interceptor_point,) = ax.plot([], [], [], "bo", markersize=8)
        title = ax.set_title("HOCBF-PN Intercept Animation")
        ax.legend(loc="upper left")

        def update(frame_idx):
            frame = gif_frames[frame_idx]
            target_line.set_data(p_t_hist[:frame, 0], p_t_hist[:frame, 1])
            target_line.set_3d_properties(p_t_hist[:frame, 2])
            interceptor_line.set_data(p_i_hist[:frame, 0], p_i_hist[:frame, 1])
            interceptor_line.set_3d_properties(p_i_hist[:frame, 2])
            target_point.set_data([p_t_hist[frame, 0]], [p_t_hist[frame, 1]])
            target_point.set_3d_properties([p_t_hist[frame, 2]])
            interceptor_point.set_data([p_i_hist[frame, 0]], [p_i_hist[frame, 1]])
            interceptor_point.set_3d_properties([p_i_hist[frame, 2]])
            dist = np.linalg.norm(p_t_hist[frame] - p_i_hist[frame])
            title.set_text(
                f"HOCBF-PN Intercept\nTime: {frame*dt:.2f}s | Distance: {dist:.2f}m"
            )
            return target_line, interceptor_line, target_point, interceptor_point, title

        anim = FuncAnimation(
            fig, update, frames=len(gif_frames), interval=1000 / fps_target, blit=False
        )
        anim.save("logs/guidance_animation.gif", writer="pillow", fps=fps_target)
        print("Animation saved to 'logs/guidance_animation.gif'")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate and visualize the HOCBF-PN guidance algorithm."
    )
    parser.add_argument(
        "--no-static", action="store_true", help="Disable saving the static plot."
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Disable saving the animation video."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to NeuroBEM .csv or MidAir .hdf5 file",
    )
    args = parser.parse_args()

    run_simulation_and_visualize(
        save_static=not args.no_static,
        save_video=not args.no_video,
        dataset_path=args.dataset,
    )
