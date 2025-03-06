import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from unicycle import UnicycleEnv

# Constants (same as in the training script)
NUM_AGENTS = 5
STATE_DIM = 4  # (x, y, gx, gy) per agent
ACTION_DIM = 2  # 2D action space per agent
DT = 0.0025  # Time step
MAX_STEPS = 400  # Maximum steps per episode
COLLISION_DISTANCE = 0.2  # Distance threshold for collision
COLLISION_PENALTY = 10000  # Penalty for collision

# Load the trained model
model = PPO.load("Baselines/CPPO/ppo_unicycle")

# Load initial points from a CSV file
# CSV format: x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5
initial_points = pd.read_csv("plots/Track/Traj_points.csv").values[:,0:8]  # Ensure the CSV file matches the format

# Function to simulate the environment and collect trajectories
def simulate_trajectory(initial_state):
    env = UnicycleEnv()  # Use the same environment class as in training
    obstacles = env.obstacles
    obs, _ = env.reset()
    env.states = initial_state  # Set the initial state

    positions = []  # To store agent positions over time
    distances = []  # To store distances to goals over time
    collision_occurred = False  # Flag to track collisions
    dt = 0.0025
    cost = 0

    for _ in range(MAX_STEPS):
        # Get action from the learned policy
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # 
        x, y, theta, v, goal_x, goal_y, g_vx, g_vy = obs
        cost += np.sqrt((x - goal_x)**2 + (y - goal_y)**2)*dt

        # Check for collisions
        for obstacle in obstacles:
            distance_to_obstacle = np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2)
            if distance_to_obstacle < COLLISION_DISTANCE:
                cost += 100

        if terminated or truncated:
            break
    
    cost += np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

    return cost

# # Plot trajectories
# def plot_trajectories(positions, initial_state):
#     goals = initial_state[2 * NUM_AGENTS:].reshape(NUM_AGENTS, 2)  # Extract goals

#     plt.figure(figsize=(10, 10))
#     for i in range(NUM_AGENTS):
#         # Plot agent trajectory
#         plt.plot(positions[:, i, 0], positions[:, i, 1], label=f"Agent {i+1}")
#         # Plot start and goal points
#         plt.scatter(initial_state[2 * i], initial_state[2 * i + 1], color="red", marker="o", s=100)
#         plt.scatter(goals[i, 0], goals[i, 1], color="green", marker="x", s=100)

#     plt.xlabel("X Position")
#     plt.ylabel("Y Position")
#     plt.title("Agent Trajectories")
#     plt.legend()
#     plt.grid()
#     # plt.show()
#     plt.savefig("plots/MAN/ddpg.png",dpi=1200) 

# Evaluate on all initial points and save results to CSV
results = []  # To store trajectory costs and collision flags

for idx, initial_state in enumerate(initial_points):
    print(f"Evaluating trajectory {idx + 1}...")
    cost = simulate_trajectory(initial_state)
    # cost = calculate_trajectory_cost(distances, collision_occurred)
    print(f"Trajectory {idx + 1} Cost: {cost}")
    results.append([cost])
    # plot_trajectories(positions, initial_state)

# Save results to a CSV file
results_df = pd.DataFrame(results, columns=["Cost",])
results_df.to_csv("plots/Track/rollout_costs_cppo.csv", index=False)
print("Trajectory costs saved to 'trajectory_costs.csv'.")