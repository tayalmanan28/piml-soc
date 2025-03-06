import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.patches import Circle
import pandas as pd

def csv_to_array(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Convert the DataFrame to a NumPy array (optional)
    array = df.to_numpy()
    
    return array  # Return both the DataFrame and the NumPy array

# Parameters
num_agents = 1
num_obstacles = 2
dim = 2  # 2D plane
delta_o = np.array([0.4, 0.5])  # Safety distance for obstacle avoidance
dt = 0.005  # Time step
total_time = 2  # Total simulation time

# Initial positions and goals
x = np.array([-2.58, 0.77]).reshape(num_agents, dim)#np.random.rand(num_agents, dim) * 10
goals = np.array([1.5, 0.0]).reshape(num_agents, dim)#np.random.rand(num_agents, dim) * 10

# print(x, goals)
# Obstacle positions
obstacles = np.array([-0.5, 0.5, -1.0, -1.2]).reshape(num_obstacles, dim)

# Function to compute nominal control input
def compute_nominal_control(x, goals):
    diff = goals - x
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    return diff / (norm + 1e-10)

# Function to compute B^{ij} for obstacle avoidance
def compute_B(x, i, j):
    B = np.zeros((num_agents, dim))
    B[i] = 2 * (x[i] - obstacles[j])
    return B.flatten()

# Function to solve QP for safe control
def solve_qp(u_nom, x, obstacles):
    u = cp.Variable((num_agents, dim))
    objective = cp.Minimize(cp.sum_squares(u - u_nom))
    
    constraints = []
    
    # Obstacle avoidance constraints
    for i in range(num_agents):
        for j in range(num_obstacles):
            B_ij = compute_B(x, i, j)
            constraints.append(B_ij @ (u.flatten() + np.array([2.0 - 0.5*x[i][1]**2, 0.0])) + np.linalg.norm(x[i] - obstacles[j])**2 - delta_o[j]**2 >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return u.value

states = csv_to_array("plots/Boat2D/Traj_points.csv")
# Simulation loop
time_steps = int(total_time / dt)
h_values = []

for i in range(len(states)):
    # Store trajectories
    x = states[i][0:2].reshape(num_agents, dim)
    trajectories = np.zeros((num_agents, time_steps, dim))

    for t in range(time_steps):
        u_nom = compute_nominal_control(x, goals)
        # print(u_nom)
        u_safe = solve_qp(u_nom, x, obstacles)
        
        # Update positions
        x += (u_safe + np.array([2.0 - 0.5*x[0][1]**2, 0.0])) * dt
        trajectories[:, t, :] = x
        
        # Compute NBF values
        # h_inter = min([np.linalg.norm(x[i] - x[j])**2 - delta_c**2 for i in range(num_agents) for j in range(i + 1, num_agents)])
        h_obs = min([np.linalg.norm(x[i] - obstacles[j])**2 - delta_o[j]**2 for i in range(num_agents) for j in range(num_obstacles)])
        h_total = h_obs#min(h_inter, h_obs)
        h_values.append(h_total)

    # Final trajectory visualization
    plt.figure()
    for i in range(num_agents):
        plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], label=f'Agent {i+1}')

    currentAxis = plt.gca()
    currentAxis.add_patch(Circle((-0.5, 0.5), 0.4, facecolor='orange', alpha=1))
    currentAxis.add_patch(Circle((-1.0, -1.2), 0.5, facecolor='orange', alpha=1))
    plt.scatter(goals[:, 0], goals[:, 1], c='green', label='Goals')
    plt.scatter(obstacles[:, 0], obstacles[:, 1], c='red', label='Obstacles')
    plt.xlim(-3, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plt.title('Agent Trajectories')
    plt.savefig("plots/Boat2D/cbf.png",dpi=1200)
# plt.show()