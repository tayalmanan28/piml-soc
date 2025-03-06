import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Dynamics: Single-integrator for planar agents
def single_integrator_dynamics(x, u, dt):
    return x + u * dt

# Barrier function for inter-agent collision avoidance
def inter_agent_barrier(x_i, x_j, delta):
    return np.linalg.norm(x_i - x_j)**2 - delta**2

# Barrier function for obstacle avoidance
def obstacle_barrier(x_i, o_k, delta):
    return np.linalg.norm(x_i - o_k)**2 - delta**2

# Solve QP using cvxopt
def solve_qp_cvxopt(x, u_nominal, agents, obstacles, delta, gamma=1.0):
    num_agents = len(agents)
    dim = 2  # 2D planar system
    u_dim = len(x)

    # Cost function: minimize deviation from nominal control
    P = 2 * np.eye(u_dim)
    q = -2 * u_nominal(x)

    # Constraint matrices
    G_list = []
    h_list = []

    # Inter-agent collision avoidance constraints
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            x_i = x[i * dim:(i + 1) * dim]
            x_j = x[j * dim:(j + 1) * dim]
            A_ij = 2 * (x_i - x_j)
            h_ij = inter_agent_barrier(x_i, x_j, delta)

            G_row = np.zeros(u_dim)
            G_row[i * dim:(i + 1) * dim] = A_ij
            G_row[j * dim:(j + 1) * dim] = -A_ij
            G_list.append(G_row)
            h_list.append(gamma * h_ij)

    # Obstacle avoidance constraints
    for i in range(num_agents):
        x_i = x[i * dim:(i + 1) * dim]
        for o_k in obstacles:
            A_ok = 2 * (x_i - o_k)
            h_ok = obstacle_barrier(x_i, o_k, delta)

            G_row = np.zeros(u_dim)
            G_row[i * dim:(i + 1) * dim] = A_ok
            G_list.append(G_row)
            h_list.append(gamma * h_ok)

    # Convert lists to cvxopt matrices
    G = matrix(np.vstack(G_list))
    h = matrix(h_list)
    P = matrix(P)
    q = matrix(q)

    # Solve the QP
    sol = solvers.qp(P, q, G, h)

    if sol['status'] != 'optimal':
        raise ValueError("QP solver failed")

    return np.array(sol['x']).flatten()

# Simulation parameters
def simulate_and_plot():
    num_agents = 1
    agents = [0]
    obstacles = [np.array([0.0, 5.0])]#[]
    delta = 0.5
    dt = 0.01
    steps = 1000

    # Initial positions
    x = np.array([0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0])

    # Nominal control policy
    def u_nominal(x):
        x_g = np.array([3.0, 3.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0])
        return x_g - x

    trajectories = [x.copy()]

    for _ in range(steps):
        try:
            u_safe = solve_qp_cvxopt(x, u_nominal, agents, obstacles, delta)
            x = single_integrator_dynamics(x, u_safe, dt)
            trajectories.append(x.copy())
        except ValueError as e:
            print("QP solver failed:", e)
            break

    print(u_nominal(x))
    trajectories = np.array(trajectories)

    # Plot trajectories
    plt.figure(figsize=(8, 8))
    for i in range(num_agents):
        plt.plot(
            trajectories[:, i * 2],
            trajectories[:, i * 2 + 1],
            label=f"Agent {i + 1}",
        )
        plt.scatter(
            trajectories[0, i * 2],
            trajectories[0, i * 2 + 1],
            label=f"Start {i + 1}",
        )

    # Plot obstacles
    for o_k in obstacles:
        circle = plt.Circle(o_k, delta, color="red", alpha=0.5)
        plt.gca().add_artist(circle)
        plt.scatter(o_k[0], o_k[1], color="red", label="Obstacle")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectories of Agents with Nonsmooth CBF")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.savefig("plots/MAN/cbf.png",dpi=1200)
    # plt.show()

# Run the simulation
simulate_and_plot()