import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from net_interception_env.mechanics.guidance_algorithm.guidance_controller import HOCBFGuidanceController

def simulate_and_visualize_video():
    controller = HOCBFGuidanceController(
        d_min=1.5, k1=2.0, k2=4.0, N=4.0, a_max_xy=30.0, a_max_z=20.0, solver='clarabel'
    )
    
    dt = 0.05 # Lower framerate for smaller GIF file (20 fps)
    max_time = 10.0
    steps = int(max_time / dt)
    
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([15.0, 0.0, 0.0])
    
    p_i_hist = []
    p_t_hist = []
    
    time = 0.0
    for _ in range(steps):
        # Target spiral
        p_t = np.array([30.0 + 3.0 * time, 10.0 * np.cos(time), 10.0 * np.sin(time)])
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
    
    # Create Animation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set static limits based on trajectory bounds
    ax.set_xlim([0, 60])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Plot elements
    target_line, = ax.plot([], [], [], 'r--', label='Target')
    interceptor_line, = ax.plot([], [], [], 'b-', linewidth=2, label='Interceptor')
    target_point, = ax.plot([], [], [], 'ro', markersize=8)
    interceptor_point, = ax.plot([], [], [], 'bo', markersize=8)
    
    title = ax.set_title("HOCBF-PN Intercept Animation")
    ax.legend(loc="upper left")
    
    def update(frame):
        # Update lines
        target_line.set_data(p_t_hist[:frame, 0], p_t_hist[:frame, 1])
        target_line.set_3d_properties(p_t_hist[:frame, 2])
        
        interceptor_line.set_data(p_i_hist[:frame, 0], p_i_hist[:frame, 1])
        interceptor_line.set_3d_properties(p_i_hist[:frame, 2])
        
        # Update points
        target_point.set_data([p_t_hist[frame, 0]], [p_t_hist[frame, 1]])
        target_point.set_3d_properties([p_t_hist[frame, 2]])
        
        interceptor_point.set_data([p_i_hist[frame, 0]], [p_i_hist[frame, 1]])
        interceptor_point.set_3d_properties([p_i_hist[frame, 2]])
        
        dist = np.linalg.norm(p_t_hist[frame] - p_i_hist[frame])
        title.set_text(f"HOCBF-PN Intercept\nTime: {frame*dt:.2f}s | Distance: {dist:.2f}m")
        return target_line, interceptor_line, target_point, interceptor_point, title

    # Must retain reference to anim to prevent garbage collection issues
    anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    
    # Save as GIF using pillow
    anim.save('net_interception_env/mechanics/guidance_algorithm/guidance_animation.gif', writer='pillow', fps=20)
    print("Animation saved to guidance_animation.gif")

if __name__ == '__main__':
    simulate_and_visualize_video()
