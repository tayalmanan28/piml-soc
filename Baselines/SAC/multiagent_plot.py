import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from multiagent import *
from anim5V import animate_trajectories

# Constants (same as in the training script)
NUM_AGENTS = 5
STATE_DIM = 4  # (x, y, gx, gy) per agent
ACTION_DIM = 2  # 2D action space per agent
DT = 0.0025  # Time step
MAX_STEPS = 400  # Maximum steps per episode
COLLISION_DISTANCE = 0.1  # Distance threshold for collision
COLLISION_PENALTY = 10000  # Penalty for collision

# Load the trained model
model = SAC.load("Baselines/SAC/sac_multiagent")

# Load initial points from a CSV file
# CSV format: x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5
# initial_points = pd.read_csv("MAN/Traj_points.csv").values[:,0:20]  # Ensure the CSV file matches the format
initial_points = np.array(([0.5, 0.0, 0.155, 0.475,-0.405, 0.294,-0.405,-0.294,0.155,-0.475, -0.405, 0.0,-0.125, -0.385, 0.327,-0.237, 0.327,0.237,-0.125, 0.385],
                            ))
print(initial_points.shape)


# Function to simulate the environment and collect trajectories
def simulate_trajectory(initial_state):
    env = MultiAgentEnv()  # Use the same environment class as in training
    obs, _ = env.reset()
    env.state = initial_state  # Set the initial state

    trajectory = []  # To store agent positions over time
    distances = []  # To store distances to goals over time
    collision_occurred = False  # Flag to track collisions
    cost = 0

    for _ in range(MAX_STEPS):
        # Get action from the learned policy
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(obs)

    #     # Store positions and distances
    #     cost += np.mean(np.linalg.norm(obs[:2 * NUM_AGENTS].reshape(NUM_AGENTS, 2) - obs[2 * NUM_AGENTS:].reshape(NUM_AGENTS, 2), axis=1))*DT
    #     positions.append(obs[:2 * NUM_AGENTS].reshape(NUM_AGENTS, 2))  # Agent positions
    #     # distances.append(np.linalg.norm(obs[:2 * NUM_AGENTS].reshape(NUM_AGENTS, 2) - obs[2 * NUM_AGENTS:].reshape(NUM_AGENTS, 2), axis=1))

    #     # Check for collisions
    #     for i in range(NUM_AGENTS):
    #         for j in range(i + 1, NUM_AGENTS):
    #             if np.linalg.norm(positions[-1][i] - positions[-1][j]) < COLLISION_DISTANCE:
    #                 cost += 100

    #     if terminated or truncated:
    #         break
    # cost += np.mean(np.linalg.norm(obs[:2 * NUM_AGENTS].reshape(NUM_AGENTS, 2) - obs[2 * NUM_AGENTS:].reshape(NUM_AGENTS, 2), axis=1))

    trajectory = np.array(trajectory)
    print(trajectory)
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5 = trajectory.T
    animate_trajectories(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5)

    return cost#np.array(positions), np.array(distances), collision_occurred

# Function to calculate trajectory cost
def calculate_trajectory_cost(distances, collision_occurred):
    # Cost = sum of (total distance from goal / 5) * dt
    trajectory_cost = np.sum(np.mean(distances, axis=1)) * DT + np.mean(distances[-1])
    # Add collision penalty if any collision occurred
    if collision_occurred:
        trajectory_cost += COLLISION_PENALTY
    return trajectory_cost

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
results_df = pd.DataFrame(results, columns=["Cost"])
results_df.to_csv("MAN/rollout_costs_SAC.csv", index=False)
# print("Trajectory costs saved to 'trajectory_costs.csv'.")