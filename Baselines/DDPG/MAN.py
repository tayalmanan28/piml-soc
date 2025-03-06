import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Constants
NUM_AGENTS = 5
STATE_DIM = 4  # (x, y, gx, gy) per agent
ACTION_DIM = 2  # 2D action space per agent
DT = 0.0025  # Time step
MAX_STEPS = 400  # Maximum steps per episode
COLLISION_DISTANCE = 0.1  # Distance threshold for collision
COLLISION_PENALTY = -100  # Penalty for collision

# Multi-Agent Environment using Gymnasium
class MultiAgentEnv(gym.Env):
    def __init__(self):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = NUM_AGENTS
        self.state_dim = STATE_DIM * NUM_AGENTS
        self.action_dim = ACTION_DIM * NUM_AGENTS
        self.max_steps = MAX_STEPS
        self.dt = DT

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # Initialize random states and goals for each agent
        np.random.seed(seed)
        self.states = np.random.uniform(-1, 1, (self.num_agents, STATE_DIM))
        self.current_step = 0

        # Arrange states in the specified order: (x1, y1, x2, y2, ..., gx_1, gy_1, gx_2, gy_2, ...)
        self.states = np.hstack([self.states[:, :2].flatten(), self.states[:, 2:].flatten()])
        return self.states.astype(np.float32), {}

    def step(self, actions):
        # Reshape actions to (num_agents, ACTION_DIM)
        actions = actions.reshape(self.num_agents, ACTION_DIM)
        print(self.states)

        # Update states: only the agent's position (x, y) is affected by actions
        positions = self.states[:2 * self.num_agents].reshape(self.num_agents, 2)
        goals = self.states[2 * self.num_agents:].reshape(self.num_agents, 2)

        # Apply actions to positions
        new_positions = positions + actions * self.dt

        # Check for collisions
        collision_penalty = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = np.linalg.norm(new_positions[i] - new_positions[j])
                if distance < COLLISION_DISTANCE:
                    collision_penalty += COLLISION_PENALTY

        # Update states
        self.states[:2 * self.num_agents] = new_positions.flatten()

        # Calculate rewards: negative distance to individual goals
        rewards = -np.linalg.norm(new_positions - goals, axis=1)
        total_reward = np.sum(rewards) + collision_penalty

        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return (
            self.states.astype(np.float32),  # Observation
            total_reward,  # Sum of rewards for all agents
            done,  # Episode ends if max steps reached
            False,  # Truncated (not used here)
            {},  # Additional info
        )

if __name__ =="__main__":

    # Define action noise for exploration
    env = MultiAgentEnv()
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create the DDPG model
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="Baselines/DDPG/ddpg_multiagent_tensorboard/",
        gamma=0.99,
        batch_size=128,
        buffer_size=int(1e6),
        learning_rate=1e-4,
    )

    # Train the model
    model.learn(total_timesteps=1000000, log_interval=10)

    # Save the model
    model.save("Baselines/DDPG/ddpg_multiagent")