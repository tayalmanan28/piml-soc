import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Dynamics of the 2D boat
def boat_dynamics(x, v, theta, dt):
    x1_dot = 2 - 0.5 * x[1] ** 2
    x2_dot = v * np.sin(theta)
    return x + dt * np.array([x1_dot, x2_dot])

# Control Barrier Function (CBF) for the circular obstacle
def cbf_circle(x, center, radius):
    return np.linalg.norm(x - center)**2 - radius**2

# Control Barrier Function (CBF) for the rectangular obstacle
def cbf_rectangle(x, center, bounds):
    dx = abs(x[0] - center[0]) - bounds[0]
    dy = abs(x[1] - center[1]) - bounds[1]
    return max(dx, dy)

# Combined CBF constraint
def cbf_constraints(x, u, obstacles, dt, v):
    constraints = []
    for obs in obstacles:
        if "circle" in obs:
            h = cbf_circle(x, obs["circle"]["center"], obs["circle"]["radius"])
            h_dot = 2 * (x - obs["circle"]["center"]) @ np.array([
                v * np.cos(u[0]),
                v * np.sin(u[0])
            ])
            constraints.append(h_dot + h / dt)

        if "rectangle" in obs:
            h = cbf_rectangle(x, obs["rectangle"]["center"], obs["rectangle"]["bounds"])
            h_dot = 0  # Simplifying assumption: h_dot is approximated as static
            constraints.append(h_dot + h / dt)

    return constraints

# CBF-based controller
def cbf_controller(x, goal, v, obstacles, dt):
    def objective(u):
        # Minimize the distance to the goal
        next_x = boat_dynamics(x, v, u[0], dt)
        return np.linalg.norm(next_x - goal)

    def constraint(u):
        # Ensure CBF constraints are satisfied
        return np.array(cbf_constraints(x, u, obstacles, dt, v))

    u0 = np.array([0.0])  # Initial guess for theta
    bounds = [(-np.pi, np.pi)]  # Bounds for theta

    cons = [{"type": "ineq", "fun": lambda u: constraint(u)}]

    result = minimize(objective, u0, bounds=bounds, constraints=cons)
    return result.x[0] if result.success else 0.0

# Simulation parameters
state_space = [(-3, 2), (-2, 2)]
v = 1.0
dt = 0.1
x_goal = np.array([1.5, 0])
obstacles = [
    {"circle1": {"center": np.array([-0.5, 0.5]), "radius": 0.4}},
    {"circle2": {"center": np.array([-1.0, -1.2]), "radius": 0.5}}
]

# Random initial state
x0 = np.array([
    np.random.uniform(state_space[0][0], state_space[0][1]),
    np.random.uniform(state_space[1][0], state_space[1][1])
])


x0 = np.array([-2.44,-1.157])
# Simulation
x_traj = [x0]
x = x0.copy()
num_steps = 100
for _ in range(num_steps):
    theta = cbf_controller(x, x_goal, v, obstacles, dt)
    x = boat_dynamics(x, v, theta, dt)
    x_traj.append(x)

    # Check if goal is reached
    if np.linalg.norm(x - x_goal) < 0.1:
        print("Goal reached!")
        break

# Visualization
x_traj = np.array(x_traj)
plt.figure(figsize=(8, 6))
plt.plot(x_traj[:, 0], x_traj[:, 1], label="Trajectory")
plt.scatter(x_goal[0], x_goal[1], color="green", label="Goal")

# Plot obstacles
obs1 = plt.Circle((-0.5, 0.5), 0.4, color='orange', alpha=0.3, label='Obstacle 1')
obs2 = plt.Circle((-1.0, -1.2), 0.5, color='orange', alpha=0.3, label='Obstacle 2')
plt.gca().add_patch(obs1)
plt.gca().add_patch(obs2)

plt.xlim(state_space[0])
plt.ylim(state_space[1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("2D Boat CBF Controller")
plt.grid()
plt.show()
