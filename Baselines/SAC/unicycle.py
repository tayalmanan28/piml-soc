import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

def csv_to_array(csv_file_path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Convert the DataFrame to a NumPy array (optional)
        array = df.to_numpy()
        
        return df, array  # Return both the DataFrame and the NumPy array

class UnicycleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(UnicycleEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-2, -2]), high=np.array([2, 2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1, -1, -math.pi, 0, -0.9, -0.9, -0.4, -0.4]), 
                                            high=np.array([1, 1, math.pi, 2, 0.9, 0.9, 0.4, 0.4]), dtype=np.float32)
        
        # Initial state
                        # [x, y, theta, v, goal_x, goal_y, gvx, gvy]
        self.state = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-math.pi, math.pi),  np.random.uniform(0, 2),
                               np.random.uniform(-0.9, 0.9), np.random.uniform(-0.9, 0.9), np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)])  
        self.obstacles = [np.array([0., 0]), np.array([-0.5, 0.5]), np.array([-0.5, -0.5]), np.array([0.5, -0.5]), np.array([0.5, 0.5])]  # List of obstacle positions
        
        # Parameters
        self.max_steps = 400
        self.current_step = 0
        self.goal_threshold = 0.05
        self.collision_threshold = 0.2

        # Rendering setup
        self.render_mode = render_mode
        self.screen_size = 500
        self.scale = self.screen_size / 2  # Scale to convert world coordinates to screen coordinates
        self.screen = None
        self.clock = None

        # Initialize pygame if render_mode is 'human'
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Unicycle Environment")
            self.clock = pygame.time.Clock()  # Corrected initialization

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        self.state = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-math.pi, math.pi),  np.random.uniform(0, 2),
                               np.random.uniform(-0.9, 0.9), np.random.uniform(-0.9, 0.9), np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)]) 
        # self.state = self.state
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        # Unpack the state
        x, y, theta, v, goal_x, goal_y, g_vx, g_vy = self.state
        a, w = action  # Linear and angular velocity

        # Update the state
        dt = 0.0025
        x += (v * np.cos(theta) * dt)
        y += (v * np.sin(theta) * dt)
        v += (a * dt)
        theta += (w * dt)
        goal_x += (g_vx * dt)
        goal_y += (g_vy * dt)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-pi, pi]

        # Update the state
        self.state = np.clip(np.array([x, y, theta, v, goal_x, goal_y, g_vx, g_vy]), a_min=np.array([-1, -1, -math.pi, 0, -0.9, -0.9, -0.4, -0.4]), 
                                            a_max=np.array([1, 1, math.pi, 2, 0.9, 0.9, 0.4, 0.4]))

        # Calculate reward
        distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        reward = -distance_to_goal  # Reward is negative distance to goal

        # Check for collision with obstacles
        for obstacle in self.obstacles:
            distance_to_obstacle = np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2)
            if distance_to_obstacle < self.collision_threshold:
                reward -= 100  # Large penalty for collision
                # done = True
                # return self.state, reward, done, False, {}

        # Check if goal is reached
        if distance_to_goal < self.goal_threshold:
            reward += 100  # Large reward for reaching the goal
            # done = True
            # return self.state, reward, done, False, {}

        # Check if max steps reached
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.state, reward, done, False, {}

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption("Unicycle Environment")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()  # Corrected initialization

        self.screen.fill((255, 255, 255))  # Clear screen with white

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),  # Red color
                self._world_to_screen(obstacle),
                int(self.collision_threshold * self.scale),
            )

        # Draw goal
        goal_x, goal_y = self.state[4], self.state[5]
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),  # Green color
            self._world_to_screen((goal_x, goal_y)),
            int(self.goal_threshold * self.scale),
        )

        # Draw unicycle
        x, y, theta = self.state[0], self.state[1], self.state[2]
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),  # Blue color
            self._world_to_screen((x, y)),
            5,  # Radius of the unicycle
        )

        # Draw orientation line
        end_x = x + 0.05 * np.cos(theta)
        end_y = y + 0.05 * np.sin(theta)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # Black color
            self._world_to_screen((x, y)),
            self._world_to_screen((end_x, end_y)),
            2,
        )

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates."""
        x, y = world_pos
        screen_x = int((x + 1) * self.scale)  # Shift and scale
        screen_y = int((1 - y) * self.scale)  # Flip y-axis
        return (screen_x, screen_y)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ =="__main__":

    # Create the environment with render_mode='human'
    env = UnicycleEnv()
    env = make_vec_env(lambda: env, n_envs=1)

    # Initialize the PPO agent
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="Baselines/SAC/sac_unicycle_tensorboard/",  # Log training data for TensorBoard
    )

    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("Baselines/SAC/sac_unicycle_1")