import numpy as np
import matplotlib.pyplot as plt
from net_interception_env.mechanics.guidance_algorithm.guidance_controller import HOCBFGuidanceController

def simulate_and_visualize():
    # Initialize the controller
    controller = HOCBFGuidanceController(
        d_min=1.5, 
        k1=2.0, 
        k2=4.0, 
        N=4.0, 
        a_max_xy=30.0, 
        a_max_z=20.0, 
        solver='clarabel'
    )
    
    dt = 0.02 # 50 Hz simulation
    max_time = 18.0
    steps = int(max_time / dt)
    
    # Initial Interceptor State
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([15.0, 0.0, 0.0])
    
    # Store history for plotting
    p_i_hist = []
    p_t_hist = []
    
    time = 0.0
    for _ in range(steps):
        # Target flies a spiraling evasion maneuver
        p_t = np.array([30.0 + 3.0 * time, 10.0 * np.cos(time), 10.0 * np.sin(time)])
        v_t = np.array([3.0, -10.0 * np.sin(time), 10.0 * np.cos(time)])
        a_t = np.array([0.0, -10.0 * np.cos(time), -10.0 * np.sin(time)])
        
        p_i_hist.append(p_i.copy())
        p_t_hist.append(p_t.copy())
        
        # Get control command from HOCBF-PN
        a_cmd = controller.compute_command(p_i, v_i, p_t, v_t, a_t)
        
        # Euler integration for interceptor physics
        v_i += a_cmd * dt
        p_i += v_i * dt
        
        time += dt
        
    p_i_hist = np.array(p_i_hist)
    p_t_hist = np.array(p_t_hist)
    
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot3D(p_t_hist[:,0], p_t_hist[:,1], p_t_hist[:,2], 'r--', label='Target (Spiral Evasion)')
    ax.plot3D(p_i_hist[:,0], p_i_hist[:,1], p_i_hist[:,2], 'b-', linewidth=2, label='Interceptor (HOCBF-PN)')
    
    # Mark final positions
    ax.scatter(p_t_hist[-1,0], p_t_hist[-1,1], p_t_hist[-1,2], color='red', s=100, marker='X')
    ax.scatter(p_i_hist[-1,0], p_i_hist[-1,1], p_i_hist[-1,2], color='blue', s=100, marker='o')
    
    # Formatting
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    final_distance = np.linalg.norm(p_t_hist[-1] - p_i_hist[-1])
    ax.set_title(f'HOCBF-PN Guidance Tracking\nFinal Standoff Distance: {final_distance:.2f}m (Target Safety Boundary: 1.5m)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('net_interception_env/mechanics/guidance_algorithm/guidance_trajectory.png', dpi=300)
    print("Trajectory plotted successfully! Saved to 'guidance_trajectory.png'")

if __name__ == '__main__':
    simulate_and_visualize()
