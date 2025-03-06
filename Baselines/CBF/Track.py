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
num_obstacles = 5
dim = 4  # 2D plane
u_dim = 2
delta_o = 0.2  # Safety distance for obstacle avoidance
dt = 0.005  # Time step
total_time = 2  # Total simulation time

# Initial positions and goals
# x = np.array([-2.58, 0.77]).reshape(num_agents, dim)#np.random.rand(num_agents, dim) * 10
goals = np.array([1.5, 0.0, 0, 0]).reshape(num_agents, dim)#np.random.rand(num_agents, dim) * 10

# print(x, goals)
# Obstacle positions
obstacles = np.array([0.5, 0.5,-0.5, 0.5, -0.5,-0.5, 0.5, -0.5, 0.0, 0.0]).reshape(num_obstacles, 2)

# Function to compute nominal control input
def compute_nominal_control(x, goals):
    
    return np.array([0.1, 0.1])

# Function to compute B^{ij} for obstacle avoidance
def compute_dh(x, i, j):
    dh = np.zeros((num_agents, dim))
    dh[i][0] = x[i][2]*np.cos(x[i][3]) + 20*(x[i][0] - obstacles[j][0])
    dh[i][1] = x[i][2]*np.sin(x[i][3]) + 20*(x[i][1] - obstacles[j][1])
    dh[i][2] = (x[i][0] - obstacles[j][0])*np.cos(x[i][3]) + (x[i][1] - obstacles[j][1])*np.sin(x[i][3])
    dh[i][3] = x[i][2]*(-(x[i][0] - obstacles[j][0])*np.sin(x[i][3]) + (x[i][1] - obstacles[j][1])*np.cos(x[i][3]))
    # dh[i] = 2 * (x[i] - obstacles[j])
    return dh.flatten()

def f(x, i):
    f_x = np.zeros((num_agents, dim))
    f_x[i][0] = x[i][2]*np.cos(x[i][3])
    f_x[i][1] = x[i][2]*np.sin(x[i][3])
    f_x[i][2] = 0.0
    f_x[i][3] = 0.0
    return f_x.flatten()

def g(x, i):
    g_x = np.zeros((dim, u_dim))
    g_x[2][0] = 1.0
    g_x[3][1] = 1.0
    return g_x

# Function to solve QP for safe control
def solve_qp(u_nom, x, obstacles):
    u = cp.Variable((num_agents, u_dim))
    objective = cp.Minimize(cp.sum_squares(u - u_nom))
    
    constraints = []
    
    # Obstacle avoidance constraints
    for i in range(num_agents):
        for j in range(num_obstacles):
            dh_ij = compute_dh(x, i, j)
            f_x = f(x, i)
            g_x = g(x, i)
            constraints.append(dh_ij @ (f_x + g_x@u.flatten()) + x[i][2]*((x[i][0] - obstacles[j][0])*np.cos(x[i][3]) + (x[i][1] - obstacles[j][1])*np.sin(x[i][3])) + 10*(np.linalg.norm(x[i][0:2] - obstacles[j])**2 - delta_o**2) >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return u.value

states = csv_to_array("plots/Track/Traj_points.csv")
# Simulation loop
time_steps = int(total_time / dt)
h_values = []

for i in range(len(states)):
    # Store trajectories
    x = states[i][0:4].reshape(num_agents, dim)
    # x = np.array([-0.8, 0.5, 1.3,-0.65]).reshape(num_agents, dim)#-0.8, 0.5, 1.3,-0.65,-0.2, -0.2, -0.1, 0.0
    trajectories = np.zeros((num_agents, time_steps, dim))

    for t in range(time_steps):
        u_nom = compute_nominal_control(x, goals)
        # print(u_nom)
        u_safe = solve_qp(u_nom, x, obstacles)
        # print(u_safe)
        
        # Update positions
        f_x = f(x, 0).reshape(num_agents, dim)
        g_x = g(x, 0).reshape(num_agents, dim, u_dim)
        x += (f_x + g_x@u_safe.flatten()) * dt
        trajectories[:, t, :] = x
        
        # Compute NBF values
        # h_inter = min([np.linalg.norm(x[i] - x[j])**2 - delta_c**2 for i in range(num_agents) for j in range(i + 1, num_agents)])
        # h_obs = min([np.linalg.norm(x[i][0:2] - obstacles[j])**2 - delta_o**2 for i in range(num_agents) for j in range(num_obstacles)])
        # h_total = h_obs#min(h_inter, h_obs)
        # h_values.append(h_total)

    # Final trajectory visualization
    plt.figure()
    for i in range(num_agents):
        plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], label=f'Agent {i+1}')

    currentAxis = plt.gca()
    currentAxis.add_patch(Circle(( 0.5, 0.5), 0.2, facecolor='orange', alpha=1))
    currentAxis.add_patch(Circle(( 0.5,-0.5), 0.2, facecolor='orange', alpha=1))
    currentAxis.add_patch(Circle((-0.5,-0.5), 0.2, facecolor='orange', alpha=1))
    currentAxis.add_patch(Circle((-0.5, 0.5), 0.2, facecolor='orange', alpha=1))
    currentAxis.add_patch(Circle(( 0.0, 0.0), 0.2, facecolor='orange', alpha=1))
    plt.scatter(goals[:, 0], goals[:, 1], c='green', label='Goals')
    # print(h_values)
    # plt.scatter(obstacles[:, 0], obstacles[:, 1], c='red', label='Obstacles')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.title('Agent Trajectories')
    plt.savefig("plots/Track/cbf.png",dpi=1200)
# plt.show()