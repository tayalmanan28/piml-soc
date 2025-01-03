import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from gym.spaces import Box
import matplotlib.pyplot as plt

# Define the Boat2D environment class
class Boat2DEnvironment:
    def __init__(self):
        self.dt = 0.025  # Time step
        self.state = np.zeros(2)  # [x1, x2]
        self.goal = np.array([1.5, 0])  # Goal position
        self.bounds = [(-3, 2), (-2, 2)]  # Environment bounds
        self.action_space = Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.time = 0

    def reset(self):
        self.state = np.array([np.random.uniform(-3,2), np.random.uniform(-2,2)])
        self.time = 0
        return self.state

    def dynamics(self, state, action):
        theta = action[0]  # Only control theta
        v = 1.0  # Fixed velocity
        x1, x2 = state

        x1_dot = v * np.cos(theta) + 2 - 0.5 * x2**2
        x2_dot = v * np.sin(theta)

        x1_new = np.clip(x1 + x1_dot * self.dt, self.bounds[0][0], self.bounds[0][1])
        x2_new = np.clip(x2 + x2_dot * self.dt, self.bounds[1][0], self.bounds[1][1])
        self.time += self.dt
        # print(self.time)

        return np.array([x1_new, x2_new])

    def compute_constraints(self, state):
        x1, x2 = state
        obs1 = max(0.4 - np.max(np.abs([x1 - (-0.5), x2 - 0.5])), 0)
        obs2 = max(0.2 - max(abs(x1 - (-1)), (1 / 5) * abs(x2 - (-1.5))), 0)
        return obs1 + obs2

    def step(self, action):
        next_state = self.dynamics(self.state, action)
        distance_to_goal = np.linalg.norm(next_state - self.goal)
        reward = -distance_to_goal  # Reward is negative distance to the goal

        if self.compute_constraints(next_state) > 0:
            reward -= 1000  # Increased penalty for violating constraints

        done = self.time >= 2#distance_to_goal < 0.1  # Episode ends if close to the goal

        if distance_to_goal < 0.1:
            reward += 100  # Large reward for reaching the goal

        self.state = next_state
        return next_state, reward, done, {}

# Define PPO components
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.fc(x)
        std = torch.exp(self.log_std).clamp(min=1e-3, max=1.0)  # Strictly clamp std
        return mu, std

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class ConstrainedPPO:
    def __init__(self, env, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.env = env
        self.policy = PolicyNetwork(input_dim=2, action_dim=1)
        self.value = ValueNetwork(input_dim=2)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def compute_advantages(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = []
        gae = 0
        for delta in reversed(deltas):
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

    def update(self, trajectories):
        states = torch.tensor(np.vstack([traj["states"] for traj in trajectories]), dtype=torch.float32)
        actions = torch.tensor(np.vstack([traj["actions"] for traj in trajectories]), dtype=torch.float32)
        rewards = torch.tensor(np.concatenate([traj["rewards"] for traj in trajectories]), dtype=torch.float32)
        dones = torch.tensor(np.concatenate([traj["dones"] for traj in trajectories]), dtype=torch.float32)

        values = self.value(states).squeeze()
        if values.dim() == 0:  # Ensure `values` is at least 1D
            values = values.unsqueeze(0)

        next_values = torch.cat([values[1:], torch.tensor([0.0], dtype=torch.float32)])
        advantages = self.compute_advantages(rewards, values, next_values, dones)

        for _ in range(10):  # Multiple PPO updates
            mu, std = self.policy(states)
            std = std.clamp(min=1e-3, max=1.0)  # Ensure stability
            cov_matrix = torch.diag_embed(std**2) + 1e-3 * torch.eye(std.size(-1))

            try:
                dist = MultivariateNormal(mu, cov_matrix)
                log_probs = dist.log_prob(actions)
            except ValueError as e:
                print("MultivariateNormal distribution error:", e)
                continue

            with torch.no_grad():
                old_log_probs = log_probs.clone()

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        value_loss = ((rewards + self.gamma * next_values - values)**2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, max_episodes=1000):
        for episode in range(max_episodes):
            state = self.env.reset()
            done = False
            trajectory = {"states": [], "actions": [], "rewards": [], "dones": []}
            

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mu, std = self.policy(state_tensor)

                std = std.clamp(min=1e-3, max=1.0)  # Clamp std
                dist = MultivariateNormal(mu, torch.diag(std) + 1e-3 * torch.eye(std.size(-1)))
                action = dist.sample().squeeze().numpy()

                next_state, reward, done, _ = self.env.step([action])

                trajectory["states"].append(state)
                trajectory["actions"].append([action])
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)

                state = next_state

            print("Episode:", episode, "Reward:", reward)
            self.update([trajectory])

        torch.save(self.policy.state_dict(), "Boat2D/policy_model.pth")
        torch.save(self.value.state_dict(), "Boat2D/value_model.pth")

# Visualize trajectories
def visualize_trajectories(env, policy):
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
        state1 = env.reset()
        env.state = start
        trajectory = [start.copy()]
        done = False

        while not done:
            state_tensor = torch.tensor(env.state, dtype=torch.float32).unsqueeze(0)
            mu, std = policy(state_tensor)
            action = mu.detach().numpy().squeeze()
            next_state, _, done, _ = env.step([action])
            trajectory.append(next_state.copy())

        print(len(trajectory))
        trajectories.append(np.array(trajectory))

    
    plt.figure(figsize=(8, 8))
    for idx, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], label=f"Start: {start_positions[idx]}")

    # Add goal
    plt.scatter(env.goal[0], env.goal[1], color='red', label='Goal', marker='*', s=150)

    # Add obstacles
    obs1 = plt.Rectangle((-0.9, 0.1), 0.8, 0.8, color='orange', alpha=0.3, label='Obstacle 1')
    obs2 = plt.Rectangle((-1.2, -2.3), 0.4, 2, color='orange', alpha=0.3, label='Obstacle 2')
    plt.gca().add_patch(obs1)
    plt.gca().add_patch(obs2)

    plt.title("CPPO Trajectories")
    plt.xlim(env.bounds[0])
    plt.ylim(env.bounds[1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.title('Trajectories with Obstacles')
    plt.savefig("Boat2D/cppo_traj_plot.png",dpi=1200)  
    plt.legend()
    # plt.grid()
    plt.show()

# Initialize and train the agent
env = Boat2DEnvironment()
agent = ConstrainedPPO(env)
agent.train(max_episodes=200000)

# Load trained model for visualization
policy = PolicyNetwork(input_dim=2, action_dim=1)
policy.load_state_dict(torch.load("Boat2D/policy_model.pth"))
visualize_trajectories(env, policy)
