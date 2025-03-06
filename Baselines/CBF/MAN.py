import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Parameters
num_agents = 5
num_obstacles = 0
dim = 2  # 2D plane
delta_c = 0.09  # Safety distance for inter-agent collisions
delta_o = 0.1  # Safety distance for obstacle avoidance
dt = 0.005  # Time step
total_time = 2  # Total simulation time

# Initial positions and goals
x = np.array([0.5, 0.0, 0.155, 0.475,-0.405, 0.294,-0.405,-0.294,0.155,-0.475]).reshape(num_agents, dim)#np.random.rand(num_agents, dim) * 10
goals = np.array([-0.405, 0.0,-0.125, -0.385, 0.327,-0.237, 0.327,0.237,-0.125, 0.385]).reshape(num_agents, dim)#np.random.rand(num_agents, dim) * 10

# print(x, goals)
# Obstacle positions
obstacles = np.random.rand(num_obstacles, dim)*0 + 1.5

# Function to compute nominal control input
def compute_nominal_control(x, goals):
    diff = goals - x
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    return diff / (norm + 1e-10)

# Function to compute A^{ij} for inter-agent collisions
def compute_A(x, i, j):
    A = np.zeros((num_agents, dim))
    A[i] = 2 * (x[i] - x[j])
    A[j] = -2 * (x[i] - x[j])
    return A.flatten()

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
    
    # Inter-agent collision avoidance constraints
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            A_ij = compute_A(x, i, j)
            constraints.append(A_ij @ u.flatten() + 1000*(np.linalg.norm(x[i] - x[j])**2 - (delta_c + 0.1)**2)**3 >= 0.0)
    
    # Obstacle avoidance constraints
    for i in range(num_agents):
        for j in range(num_obstacles):
            B_ij = compute_B(x, i, j)
            constraints.append(B_ij @ u.flatten() + np.linalg.norm(x[i] - obstacles[j])**2 - delta_o**2 >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return u.value

# Simulation loop
time_steps = int(total_time / dt)
h_values = []

# Store trajectories
trajectories = np.zeros((num_agents, time_steps, dim))

for t in range(time_steps):
    u_nom = compute_nominal_control(x, goals)
    # print(u_nom)
    u_safe = solve_qp(u_nom, x, obstacles)
    
    # Update positions
    x += u_safe * dt
    trajectories[:, t, :] = x
    
    # Compute NBF values
    h_inter = min([np.linalg.norm(x[i] - x[j])**2 - delta_c**2 for i in range(num_agents) for j in range(i + 1, num_agents)])
    # h_obs = min([np.linalg.norm(x[i] - obstacles[j])**2 - delta_o**2 for i in range(num_agents) for j in range(num_obstacles)])
    h_total = h_inter#min(h_inter, h_obs)
    h_values.append(h_total)
    
    # Visualization
    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], c='blue', label='Agents')
    plt.scatter(goals[:, 0], goals[:, 1], c='green', label='Goals')
    plt.scatter(obstacles[:, 0], obstacles[:, 1], c='red', label='Obstacles')
    for i in range(num_agents):
        plt.plot(trajectories[i, :t+1, 0], trajectories[i, :t+1, 1], 'k--')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.pause(0.01)

# Plot NBF values over time
plt.figure()
plt.plot(np.arange(0, total_time, dt), h_values)
plt.xlabel('Time')
plt.ylabel('NBF Value')
plt.title('NBF Value Over Time')
plt.grid()
plt.show()
print(h_values)

# Final trajectory visualization
plt.figure()
for i in range(num_agents):
    plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], label=f'Agent {i+1}')
plt.scatter(goals[:, 0], goals[:, 1], c='green', label='Goals')
plt.scatter(obstacles[:, 0], obstacles[:, 1], c='red', label='Obstacles')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.legend()
plt.title('Agent Trajectories')
plt.show()