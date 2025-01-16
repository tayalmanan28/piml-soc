import numpy as np
import matplotlib.pyplot as plt

# Dynamics of the 2D boat
def boat_dynamics(x, u1, u2, dt):
    x1_dot = u1 + 2 - 0.5 * x[1] ** 2
    x2_dot = u2#np.sqrt(1-u1*u1)
    return x + dt * np.array([x1_dot, x2_dot])

# Cost function
def compute_cost(x, goal, obstacles, u1, u2):
    # Distance to the goal
    goal_cost = np.linalg.norm(x - goal)

    u_cost = 0# np.linalg.norm(np.array([u1, u2])) -1

    # Obstacle cost
    obstacle_cost = 0
    for obs in obstacles:
        obs1 = max(0.4 - np.linalg.norm(x - obs["center"], np.inf), 0)
        obs2 = max(0.2 - max(abs(x[0] - obs["rect_center"][0]), (1/5) * abs(x[1] - obs["rect_center"][1])), 0)
        obstacle_cost += obs1 + obs2

    return 1*goal_cost + 10*obstacle_cost + 100*u_cost

# MPPI controller
def mppi_controller(x0, goal, obstacles, num_samples=100, horizon=50, dt=0.025, lam=0.010):
    # Initialize control samples
    u1s = np.random.uniform(-1, 1, (num_samples, horizon))
    u2s = np.random.uniform(-1, 1, (num_samples, horizon))
    # thetas = np.random.uniform(-np.pi, np.pi, (num_samples, horizon))

    # Initialize costs
    costs = np.zeros(num_samples)

    # Simulate trajectories and compute costs
    for i in range(num_samples):
        x = x0.copy()
        cost = 0
        for t in range(horizon):
            u1 = u1s[i, t]
            u2 = u2s[i, t]
            x = boat_dynamics(x, u1, u2, dt)
            cost += compute_cost(x, goal, obstacles, u1, u2)
        costs[i] = cost

    # Compute weights
    min_cost = np.min(costs)
    weights = np.exp(-lam * (costs - min_cost))
    weights /= np.sum(weights)

    # Compute optimal control
    optimal_u1 = np.sum(u1s.T * weights, axis=1)
    optimal_u2 = np.sum(u2s.T * weights, axis=1)
    return optimal_u1[0], optimal_u2[0]  # Return the first control input

# Simulation parameters
state_space = [(-3, 2), (-2, 2)]
v = 1.0
dt = 0.025
x_goal = np.array([1.5, 0])
obstacles = [
    {"center": np.array([-0.5, 0.5]), "rect_center": np.array([-1, -1.5])}
]

# Random initial state
# x0 = np.array([
#     np.random.uniform(state_space[0][0], state_space[0][1]),
#     np.random.uniform(state_space[1][0], state_space[1][1])
# ])


start_positions =  [np.array([-2.44,-1.157]),
                    np.array([-2.20, 1.890]),
                    np.array([-1.33, 1.510]),
                    np.array([ 0.69, 1.100]),
                    np.array([-1.93, 0.060]),
                    np.array([-2.33,-0.520]),
                    np.array([0.068, 0.870]),
                    np.array([-0.52,-0.140]),
                    np.array([-0.63,-1.210]),
                    np.array([-2.58, 0.770])]
    
#[np.array([-2, -1]), np.array([-1, 1]), np.array([0, 0]), np.array([1, -1]), np.array([1.5, 1.5])]
trajectories = []

for start in start_positions:
    trajectory = [start]
    x = start.copy()

    num_steps = 800
    for _ in range(num_steps):
        u1, u2 = mppi_controller(x, x_goal, obstacles, num_samples=100, horizon=50, dt=dt)
        x = boat_dynamics(x, u1, u2, dt)
        trajectory.append(x)
        
        # Check if goal is reached
        if np.linalg.norm(x - x_goal) < 0.1:
            print("Goal reached!")
            # break

    print(len(trajectory))
    trajectories.append(np.array(trajectory))


plt.figure(figsize=(8, 8))
for idx, traj in enumerate(trajectories):
    plt.plot(traj[:, 0], traj[:, 1], label=f"Start: {start_positions[idx]}")

# Add goal
plt.scatter(x_goal[0], x_goal[1], color='red', label='Goal', marker='*', s=150)

# Add obstacles
obs1 = plt.Rectangle((-0.9, 0.1), 0.8, 0.8, color='orange', alpha=0.3, label='Obstacle 1')
obs2 = plt.Rectangle((-1.2, -2.3), 0.4, 2, color='orange', alpha=0.3, label='Obstacle 2')
plt.gca().add_patch(obs1)
plt.gca().add_patch(obs2)

plt.title("MPPI Controller")
plt.xlim(state_space[0])
plt.ylim(state_space[1])
plt.xlabel('x1')
plt.ylabel('x2')
# plt.title('Trajectories with Obstacles')
plt.savefig("Boat2D/mppi_traj_plot.png",dpi=1200)  
plt.legend()
# plt.grid()
plt.show()

# x0 = np.array([-2.20, 1.890])

# # Simulation
# x_traj = [x0]
# x = x0.copy()
# num_steps = 100
# for _ in range(num_steps):
#     theta = mppi_controller(x, x_goal, v, obstacles, num_samples=100, horizon=25, dt=dt)
#     x = boat_dynamics(x, v, theta, dt)
#     x_traj.append(x)
    
#     # Check if goal is reached
#     if np.linalg.norm(x - x_goal) < 0.1:
#         print("Goal reached!")
#         break

# # Visualization
# x_traj = np.array(x_traj)
# plt.figure(figsize=(8, 6))
# plt.plot(x_traj[:, 0], x_traj[:, 1], label="Trajectory")
# plt.scatter(x_goal[0], x_goal[1], color="green", label="Goal")

# # Plot obstacles
# obs1 = plt.Rectangle((-0.9, 0.1), 0.8, 0.8, color='orange', alpha=0.3, label='Obstacle 1')
# obs2 = plt.Rectangle((-1.2, -2.3), 0.4, 2, color='orange', alpha=0.3, label='Obstacle 2')
# plt.gca().add_patch(obs1)
# plt.gca().add_patch(obs2)

# plt.xlim(state_space[0])
# plt.ylim(state_space[1])
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.legend()
# plt.title("2D Boat MPPI Controller")
# # plt.grid()
# plt.show()

