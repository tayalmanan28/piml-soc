import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

def animate_trajectory(x, y, boat_image_path='plots/Boat2D/boat.jpg', output_gif='plots/Boat2D/trajectory_animation.gif'):
    """
    Animates a boat moving along a given trajectory.

    Parameters:
        x (array-like): X-coordinates of the trajectory.
        y (array-like): Y-coordinates of the trajectory.
        boat_image_path (str): Path to the boat image file.
        output_gif (str): Name of the output GIF file.
    """
    # Load the boat image
    boat_img = mpimg.imread(boat_image_path)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-3, 2)  # Set x-axis limits dynamically
    ax.set_ylim(-2, 2)  # Set y-axis limits dynamically
    ax.set_title("Boat Trajectory Animation")

    # Plot the trajectory
    ax.plot(x, y, 'g--', alpha=0.5, label='Trajectory')

    # Add circles (as in your original code)
    ax.add_patch(Circle((-0.5, 0.5), 0.4, facecolor='orange', alpha=1))
    ax.add_patch(Circle((-1.0, -1.2), 0.5, facecolor='orange', alpha=1))
    ax.add_patch(Circle((1.5, 0), 0.025, facecolor='cyan', alpha=1))

    # Initialize the boat image
    boat = ax.imshow(boat_img, extent=(x[0] - 0.1, x[0] + 0.1, y[0], y[0] + 0.2), alpha=1)

    # Function to update the boat position for each frame
    def update(frame):
        # Update the boat's position
        boat.set_extent([x[frame] - 0.1, x[frame] + 0.1, y[frame], y[frame] + 0.2])
        return boat,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)

    # Save the animation as a GIF
    ani.save(output_gif, writer='pillow', fps=60)

    # Show the animation
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example trajectory data
    t = np.linspace(0, 10, 100)  # Time vector
    x = np.sin(t)  # Example x trajectory
    y = np.cos(t)  # Example y trajectory

    # Call the function to animate the trajectory
    animate_trajectory(x, y, boat_image_path='boat.png', output_gif='boat_trajectory_animation.gif')