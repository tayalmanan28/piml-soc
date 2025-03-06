import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame

class BoatEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(BoatEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-3, -2]), 
                                            high=np.array([2, 2]), dtype=np.float32)
        
        # Initial state
        self.state = np.array([0, 0])  # [x, y]
        self.obstacles = [np.array([-0.5, 0.5]), np.array([-1, -1.2])]  # List of obstacle positions
        
        # Parameters
        self.max_steps = 800
        self.current_step = 0
        self.goal_threshold = 0.1
        self.collision_threshold = [0.4, 0.5] 

        # Rendering setup
        self.render_mode = render_mode
        self.screen_size = 500
        self.scale = self.screen_size / 5 # Scale to convert world coordinates to screen coordinates
        self.screen = None
        self.clock = None

        # Initialize pygame if render_mode is 'human'
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Boat Environment")
            self.clock = pygame.time.Clock()  # Corrected initialization

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        self.state = np.array([np.random.uniform(-3,2), np.random.uniform(-2,2)])
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        # Unpack the state
        x, y = self.state
        u1, u2 = action  # Linear and angular velocity
        u = np.linalg.norm(action)
        u1 = u1/u
        u2 = u2/u

        # Update the state
        dt = 0.025
        x += u1 + 2 - 0.5 * y * y * dt
        y += u2

        # Update the state
        self.state = np.array([x, y])

        # Calculate reward
        distance_to_goal = np.sqrt((x - 1.5)**2 + (y - 0.0)**2)
        reward = -distance_to_goal  # Reward is negative distance to goal
        
        obstacle_num = 0
        # Check for collision with obstacles
        for obstacle in self.obstacles:
            distance_to_obstacle = np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2)
            if distance_to_obstacle < self.collision_threshold[obstacle_num]:
                reward -= 100  # Large penalty for collision
                #done = True
                #return self.state, reward, done, False, {}
            obstacle_num += 1

        # Check if goal is reached
        if distance_to_goal < self.goal_threshold:
            reward += 100  # Large reward for reaching the goal
            #done = True
            #return self.state, reward, done, False, {}

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
                pygame.display.set_caption("Boat Environment")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()  # Corrected initialization

        self.screen.fill((255, 255, 255))  # Clear screen with white

        # Draw obstacles
        obstacle_num = 0
        for obstacle in self.obstacles:
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),  # Red color
                self._world_to_screen(obstacle),
                int(self.collision_threshold[obstacle_num] * self.scale),
            )
            obstacle_num += 1

        # Draw goal
        goal_x, goal_y = 1.5, 0.0
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),  # Green color
            self._world_to_screen((goal_x, goal_y)),
            int(self.goal_threshold * self.scale),
        )

        # Draw unicycle
        x, y = self.state[0], self.state[1]
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),  # Blue color
            self._world_to_screen((x, y)),
            5,  # Radius of the unicycle
        )

        # Draw orientation line
        # end_x = x + 0.5 * np.cos(theta)
        # end_y = y + 0.5 * np.sin(theta)
        # pygame.draw.line(
        #     self.screen,
        #     (0, 0, 0),  # Black color
        #     self._world_to_screen((x, y)),
        #     self._world_to_screen((end_x, end_y)),
        #     2,
        # )

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
        screen_x = int((x + 3) * self.scale)  # Shift and scale
        screen_y = int((2 - y) * self.scale)  # Flip y-axis
        return (screen_x, screen_y)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None