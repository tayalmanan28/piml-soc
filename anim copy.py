
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_trajectories(v1_x, v1_y, v2_x, v2_y,v3_x, v3_y, interval=10, marker_size=0.25):
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
    trajectories = [vehicle1, vehicle2, vehicle3]
    num_vehicles = len(trajectories)
    num_frames = max(len(traj) for traj in trajectories)

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_title("Vehicle Trajectories")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Initialize scatter plots for vehicles
    scatters = [ax.plot([], [], 'o', label=f'Vehicle {i+1}')[0] for i in range(num_vehicles)]

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

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=500
    )

    # Add legend
    ax.legend()

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
    x3 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    y3 = np.cumsum(np.random.randn(time_steps, 1), axis=0)

    # Call the function to animate trajectories with faster speed and larger markers
    animate_trajectories(x1, y1, x2, y2, x3, y3, interval=10, marker_size=0.25)
