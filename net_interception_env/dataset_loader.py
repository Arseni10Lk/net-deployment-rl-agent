import numpy as np
import h5py
import pandas as pd
from dataclasses import dataclass

@dataclass
class TrajectoryData:
    name: str
    time: np.ndarray        # (N,) array of timestamps in seconds
    position: np.ndarray    # (N, 3) XYZ positions
    velocity: np.ndarray    # (N, 3) XYZ velocities
    acceleration: np.ndarray# (N, 3) XYZ accelerations
    quaternion: np.ndarray | None = None # (N, 4) XYZW quaternions

    @property
    def dt(self):
        return np.mean(np.diff(self.time))

def load_midair_trajectory(hdf5_path: str, trajectory_name: str) -> TrajectoryData:
    """
    Loads a specific trajectory from a MidAir sensor_records.hdf5 file.
    MidAir ground truth is recorded at 100Hz.
    """
    with h5py.File(hdf5_path, 'r') as f:
        grp = f[f"{trajectory_name}/groundtruth"]
        pos = grp['position'][:]
        vel = grp['velocity'][:]
        acc = grp['acceleration'][:]
        quat = grp['attitude'][:]  # Usually WXYZ or XYZW, MidAir specifies WXYZ typically, but we store it as is

    # Generate synthetic time array assuming 100Hz
    num_samples = pos.shape[0]
    time = np.arange(num_samples) * 0.01

    return TrajectoryData(
        name=f"midair_{trajectory_name}",
        time=time,
        position=pos,
        velocity=vel,
        acceleration=acc,
        quaternion=quat
    )

def load_midair_all(hdf5_path: str) -> list[TrajectoryData]:
    """Loads all available trajectories from a MidAir HDF5 file."""
    trajectories = []
    with h5py.File(hdf5_path, 'r') as f:
        traj_names = [k for k in f.keys() if k.startswith("trajectory_")]
    
    for name in traj_names:
        trajectories.append(load_midair_trajectory(hdf5_path, name))
    return trajectories

def load_neurobem_csv(csv_path: str) -> TrajectoryData:
    """
    Loads trajectory data from a NeuroBEM processed CSV file.
    """
    df = pd.read_csv(csv_path)
    
    time = df['t'].values
    pos = df[['pos x', 'pos y', 'pos z']].values
    vel = df[['vel x', 'vel y', 'vel z']].values
    # The 'acc' column in NeuroBEM is raw body-frame IMU data (includes 9.81 gravity and rotates with the drone).
    # We must compute true world-frame kinematic acceleration by differentiating the world-frame velocity.
    acc = np.zeros_like(vel)
    dt = np.mean(np.diff(time))
    acc[1:] = (vel[1:] - vel[:-1]) / dt
    acc[0] = acc[1] # Copy first frame
    quat = df[['quat x', 'quat y', 'quat z', 'quat w']].values

    name = csv_path.split("/")[-1].replace(".csv", "")

    return TrajectoryData(
        name=f"neurobem_{name}",
        time=time,
        position=pos,
        velocity=vel,
        acceleration=acc,
        quaternion=quat
    )

if __name__ == "__main__":
    # Quick tests
    print("Testing NeuroBEM Loader...")
    nb_traj = load_neurobem_csv("dataset_data/neurobem/processed_data/merged_2021-02-03-13-43-38_seg_1.csv")
    print(f"Loaded: {nb_traj.name} | Samples: {len(nb_traj.time)} | Duration: {nb_traj.time[-1]:.2f}s | dt: {nb_traj.dt:.4f}s")
    
    print("\nTesting MidAir Loader...")
    ma_trajs = load_midair_all("dataset_data/mid_air_data/MidAir/Kite_training/sunny/sensor_records.hdf5")
    print(f"Loaded {len(ma_trajs)} trajectories from MidAir HDF5.")
    print(f"First traj: {ma_trajs[0].name} | Samples: {len(ma_trajs[0].time)} | Duration: {ma_trajs[0].time[-1]:.2f}s | dt: {ma_trajs[0].dt:.4f}s")
