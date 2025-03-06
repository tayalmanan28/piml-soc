import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle

def animate_trajectories(vehicle_trajectories, goal_positions, traj_number, interval=10, marker_size=0.25, save_interval=200, background_color='#E3E3E3'):
    """
    Creates an accelerated animation of trajectories for a given number of vehicles with increased marker size.
    Saves snapshots of the animation after every `save_interval` points.

    Parameters:
    - vehicle_trajectories: list of np.ndarray, each containing the x and y coordinates for each vehicle.
    - goal_positions: list of tuples, each containing the goal positions (gx, gy) for each vehicle.
    - interval: int, the time in milliseconds between frames (lower = faster animation).
    - marker_size: float, the radius of the markers for the vehicles.
    - save_interval: int, save a snapshot after every `save_interval` points.
    - background_color: str, the custom background color for the plot (default is dark gray).
    """
    num_vehicles = len(vehicle_trajectories)
    num_frames = max(len(traj) for traj in vehicle_trajectories)

    # Define colors for each vehicle
    colors = ['#FF9500', '#0091FF', '#0D948F', '#CDB3FF', '#6B6B6B', '#850F67', '#FF5733', '#33FF57', '#3357FF', '#F333FF']

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Set custom background color
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    # Set tick colors to contrast with the background
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Initialize scatter plots for vehicles
    scatters = [ax.plot([], [], 'o', markersize=16, color=colors[i % len(colors)], label=f'Vehicle {i+1}')[0] 
                for i in range(num_vehicles)]

    # Initialize paths for vehicles with broader lines
    paths = [ax.plot([], [], '-', color=colors[i % len(colors)], alpha=1, linewidth=2.5)[0] for i in range(num_vehicles)]

    # Function to initialize the animation
    def init():
        for scatter, path in zip(scatters, paths):
            scatter.set_data([], [])
            path.set_data([], [])
        return scatters + paths

    # Function to update the animation at each frame
    def update(frame):
        for i, (scatter, path, trajectory) in enumerate(zip(scatters, paths, vehicle_trajectories)):
            if frame < len(trajectory):
                # Update scatter position
                scatter.set_data(trajectory[frame, 0], trajectory[frame, 1])
                # Update path to include current frame
                path.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])

        # Save snapshot after every `save_interval` points
        if frame % save_interval == 0:
            plt.savefig(f'plots/MAN/traj_{traj_number}_frame_{frame}.png', facecolor=fig.get_facecolor())
            print(f"Snapshot saved at frame {frame}")

        return scatters + paths

    # Add goals with the same color as the corresponding vehicle
    for i, (gx, gy) in enumerate(goal_positions):
        ax.add_patch(Circle((gx, gy), 0.02, facecolor=colors[i % len(colors)], alpha=1))

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=interval
    )

    # Save the animation with the custom background color
    ani.save(f'plots/MAN/traj_{traj_number}_animation.gif', fps=30, writer='pillow', savefig_kwargs={'facecolor': background_color})
    print(f"Animation saved")

    # Show the animation
    plt.show()

# Example usage (this section can be moved to another file):
if __name__ == "__main__":
    # Example: Simulating large trajectories
    time_steps = 801
    num_vehicles = 6  # Specify the number of vehicles

    # Generate random trajectories for each vehicle
    vehicle_trajectories = [np.column_stack((np.cumsum(np.random.randn(time_steps, 1), np.cumsum(np.random.randn(time_steps, 1))))) 
                            for _ in range(num_vehicles)]

    # Define goal positions for each vehicle
    goal_positions = [(0.5, 0.5), (-0.5, 0.5), (0.0, 0.0), (0.5, -0.5), (-0.5, -0.5), (0.0, 0.5)]

    # Call the function to animate trajectories with faster speed and larger markers
    animate_trajectories(vehicle_trajectories, goal_positions, interval=10, marker_size=0.25, save_interval=200, background_color='#1E1E1E')