import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle

def animate_trajectories(v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, v4_x, v4_y, v5_x, v5_y, 
                         gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5, 
                         interval=10, marker_size=0.25, save_interval=200, background_color='#E3E3E3'):
    """
    Creates an accelerated animation of trajectories for five vehicles with increased marker size.
    Saves snapshots of the animation after every `save_interval` points.

    Parameters:
    - v1_x, v1_y, ..., v5_x, v5_y: np.ndarray, x and y coordinates for each vehicle.
    - gx_1, gy_1, ..., gx_5, gy_5: float, goal positions for each vehicle.
    - interval: int, the time in milliseconds between frames (lower = faster animation).
    - marker_size: float, the radius of the markers for the vehicles.
    - save_interval: int, save a snapshot after every `save_interval` points.
    - background_color: str, the custom background color for the plot (default is dark gray).
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

    # Define colors for each vehicle
    colors = ['#FF9500', '#0091FF', '#0D948F', '#CDB3FF', '#6B6B6B']

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    # ax.set_title("Vehicle Trajectories", color='white')  # Set title color to contrast with background
    # ax.set_xlabel("X-axis", color='white')  # Set x-axis label color
    # ax.set_ylabel("Y-axis", color='white')  # Set y-axis label color

    # Set custom background color
    ax.set_facecolor(background_color)  # Background color for the axes
    fig.patch.set_facecolor(background_color)  # Background color for the figure

    # Set tick colors to contrast with the background
    ax.tick_params(axis='x', colors='white')  # X-axis tick color
    ax.tick_params(axis='y', colors='white')  # Y-axis tick color

    # Initialize scatter plots for vehicles
    scatters = [ax.plot([], [], 'o', markersize=16, color=colors[i], label=f'Vehicle {i+1}')[0] 
                for i in range(num_vehicles)]

    # Initialize paths for vehicles with broader lines
    paths = [ax.plot([], [], '-', color=colors[i], alpha=1, linewidth=2.5)[0] for i in range(num_vehicles)]

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

        # Save snapshot after every `save_interval` points
        if frame % save_interval == 0:
            plt.savefig(f'plots/MAN/MPPI_SF_{frame}.png', facecolor=fig.get_facecolor())  # Save with custom background
            print(f"Snapshot saved at frame {frame}")

        return scatters + paths

    # Add goals with the same color as the corresponding vehicle
    goals = [(gx_1[0], gy_1[0]), (gx_2[0], gy_2[0]), (gx_3[0], gy_3[0]), 
             (gx_4[0], gy_4[0]), (gx_5[0], gy_5[0])]
    for i, (gx, gy) in enumerate(goals):
        ax.add_patch(Circle((gx, gy), 0.02, facecolor=colors[i], alpha=1))

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=interval
    )

    # Save the animation with the custom background color
    ani.save('plots/MAN/mppi_sf.gif', fps=30, writer='pillow', savefig_kwargs={'facecolor': background_color})
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
    x3 = np.cumsum(np.random.randn(time_steps, 1), axis=0) * 0 + 0.125
    y3 = np.cumsum(np.random.randn(time_steps, 1), axis=0) * 0 - 1
    x4 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    y4 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    x5 = np.cumsum(np.random.randn(time_steps, 1), axis=0)
    y5 = np.cumsum(np.random.randn(time_steps, 1), axis=0)

    # Define goal positions
    gx_1, gy_1 = [0.5], [0.5]
    gx_2, gy_2 = [-0.5], [0.5]
    gx_3, gy_3 = [0.0], [0.0]
    gx_4, gy_4 = [0.5], [-0.5]
    gx_5, gy_5 = [-0.5], [-0.5]

    # Call the function to animate trajectories with faster speed and larger markers
    animate_trajectories(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, 
                         gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5, 
                         interval=10, marker_size=0.25, save_interval=200, background_color='#1E1E1E')