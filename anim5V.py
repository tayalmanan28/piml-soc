import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle, Ellipse

def animate_trajectories(v1_x, v1_y, v2_x, v2_y,v3_x, v3_y, v4_x, v4_y,v5_x, v5_y, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5, interval=10, marker_size=0.25):
    """
    Creates an accelerated animation of trajectories for three vehicles with increased marker size.

    Parameters:
    - vehicle1: np.ndarray of shape (T1, 2), where T1 is the number of time steps for vehicle 1.
    - vehicle2: np.ndarray of shape (T2, 2), where T2 is the number of time steps for vehicle 2.
    - vehicle3: np.ndarray of shape (T3, 2), where T3 is the number of time steps for vehicle 3.
    - interval: int, the time in milliseconds between frames (lower = faster animation).
    - marker_size: float, the radius of the markers for the vehicles.
    """
    # Combine all vehicle trajectories
    vehicle1 = np.column_stack((v1_x, v1_y))
    vehicle2 = np.column_stack((v2_x, v2_y))
    vehicle3 = np.column_stack((v3_x, v3_y))
    vehicle4 = np.column_stack((v4_x, v4_y))
    vehicle5 = np.column_stack((v5_x, v5_y))
    trajectories = [vehicle1, vehicle2, vehicle3, vehicle4, vehicle5]
    num_vehicles = len(trajectories)
    num_frames = max(len(traj) for traj in trajectories)

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Vehicle Trajectories")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Initialize scatter plots for vehicles
    scatters = [ax.plot([], [], 'o', markersize = 42, label=f'Vehicle {i+1}')[0] for i in range(num_vehicles)]

    # Initialize paths for vehicles
    paths = [ax.plot([], [], '-', alpha=0.5)[0] for _ in range(num_vehicles)]

    # Function to initialize the animation
    def init():
        for scatter, path in zip(scatters, paths):
            scatter.set_data([], [])
            path.set_data([], [])
        return scatters + paths

    # Function to update the animation at each frame
    def update(frame):
        for i, (scatter, path, trajectory) in enumerate(zip(scatters, paths, trajectories)):
            if frame < len(trajectory):
                # Update scatter position
                scatter.set_data(trajectory[frame, 0], trajectory[frame, 1])
                # Update path to include current frame
                path.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
        return scatters + paths

    currentAxis1 = plt.gca()
    currentAxis1.add_patch(Circle((gx_1[0], gy_1[0]), 0.05, facecolor = 'blue', alpha=1))
    currentAxis2 = plt.gca()
    currentAxis2.add_patch(Circle((gx_2[0], gy_2[0]), 0.05, facecolor = 'orange', alpha=1))
    currentAxis3 = plt.gca()
    currentAxis3.add_patch(Circle((gx_3[0], gy_3[0]), 0.05, facecolor = 'green', alpha=1))
    currentAxis4 = plt.gca()
    currentAxis4.add_patch(Circle((gx_4[0], gy_4[0]), 0.05, facecolor = 'red', alpha=1))
    currentAxis5 = plt.gca()
    currentAxis5.add_patch(Circle((gx_5[0], gy_5[0]), 0.05, facecolor = 'cyan', alpha=1))

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=interval
    )

    # # Add legend
    # ax.legend()

    # Save the animation if a filename is provided
    ani.save('plots/MVC/animation_5.gif', fps=30, writer='pillow')
    print(f"Animation saved")

    # Show the animation
    plt.show()

# Example usage (this section can be moved to another file):
if __name__ == "__main__":
    # Example: Simulating large trajectories
    time_steps = 801
    x1 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    y1 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    x2 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    y2 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    x3 = np.cumsum(np.random.randn(time_steps, 1), axis=0)*0 + 0.125
    y3 = np.cumsum(np.random.randn(time_steps, 1), axis=0)*0 -1

    # Call the function to animate trajectories with faster speed and larger markers
    animate_trajectories(x1, y1, x2, y2, x3, y3,  interval=10, marker_size=0.25)
