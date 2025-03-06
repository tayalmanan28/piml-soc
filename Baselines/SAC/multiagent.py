import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame

class MultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(MultiAgentEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6]), high=np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9]), 
                                            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]), dtype=np.float32)
        
        # Initial state
        self.state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])  # [x, y, theta, goal_x, goal_y]
        #self.obstacles = [np.array([2, 2]), np.array([-2, -2])]  # List of obstacle positions
        
        # Parameters
        self.max_steps = 800
        self.current_step = 0
        self.goal_threshold = 0.05
        self.collision_threshold = 0.1

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
            pygame.display.set_caption("Multiagent Environment")
            self.clock = pygame.time.Clock()  # Corrected initialization

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        self.state = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9), np.random.uniform(-0.9,0.9)])
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        # Unpack the state
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx1, gy1, gx2, gy2, gx3, gy3, gx4, gy4, gx5, gy5 = self.state
        vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4, vx5, vy5  = action  # Linear and angular velocity
        # print(vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4, vx5, vy5)
        # print(self.current_step)
        if x1 < -1.0:
            x1 = -1.0
        elif x1 > 1.0:
            x1 = 1.0
        elif y1 < -1.0:
            y1 = -1.0
        elif y1 > 1.0:
            y1 = 1.0
        elif x2 < -1.0:
            x2 = -1.0
        elif x2 > 1.0:
            x2 = 1.0
        elif y2 < -1.0:
            y2 = -1.0
        elif y2 > 1.0:
            y2 = 1.0
        elif x3 < -1.0:
            x3 = -1.0
        elif x3 > 1.0:
            x3 = 1.0
        elif y3 < -1.0:
            y3 = -1.0
        elif y3 > 1.0:
            y3 = 1.0
        elif x4 < -1.0:
            x4 = -1.0
        elif x4 > 1.0:
            x4 = 1.0
        elif y4 < -1.0:
            y4 = -1.0
        elif y4 > 1.0:
            y4 = 1.0
        elif x5 < -1.0:
            x5 = -1.0
        elif x5 > 1.0:
            x5 = 1.0
        elif y5 < -1.0:
            y5 = -1.0
        elif y5 > 1.0:
            y5 = 1.0

        if gx1 < -0.9:
            gx1 = -0.9
        elif gx1 > 0.9:
            gx1 = 0.9
        elif gy1 < -0.9:
            gy1 = -0.9
        elif gy1 > 0.9:
            gy1 = 0.9
        elif gx2 < -0.9:
            gx2 = -0.9
        elif gx2 > 0.9:
            gx2 = 0.9
        elif gy2 < -0.9:
            gy2 = -0.9
        elif gy2 > 0.9:
            gy2 = 0.9
        elif gx3 < -0.9:
            gx3 = -0.9
        elif gx3 > 0.9:
            gx3 = 0.9
        elif gy3 < -0.9:
            gy3 = -0.9
        elif gy3 > 0.9:
            gy3 = 0.9
        elif gx4 < -0.9:
            gx4 = -0.9
        elif gx4 > 0.9:
            gx4 = 0.9
        elif gy4 < -0.9:
            gy4 = -0.9
        elif gy4 > 0.9:
            gy4 = 0.9
        elif gx5 < -0.9:
            gx5 = -0.9
        elif gx5 > 0.9:
            gx5 = 0.9
        elif gy5 < -0.9:
            gy5 = -0.9
        elif gy5 > 0.9:
            gy5 = 0.9

        # Update the state
        dt = 0.0025
        x1 += (vx1 * dt)
        y1 += (vy1 * dt)
        x2 += (vx2 * dt)
        y2 += (vy2 * dt)
        x3 += (vx3 * dt)
        y3 += (vy3 * dt)
        x4 += (vx4 * dt)
        y4 += (vy4 * dt)
        x5 += (vx5 * dt)
        y5 += (vy5 * dt)

        # Update the state
        self.state = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx1, gy1, gx2, gy2, gx3, gy3, gx4, gy4, gx5, gy5])

        # Calculate reward
        distance_to_goal = (np.sqrt((x1 - gx1)**2 + (y1 - gy1)**2) + np.sqrt((x2 - gx2)**2 + (y2 - gy2)**2) + np.sqrt((x3 - gx3)**2 + (y3 - gy3)**2) + np.sqrt((x4 - gx4)**2 + (y4 - gy4)**2) + np.sqrt((x5 - gx5)**2 + (y5 - gy5)**2))/5
        reward = - distance_to_goal  # Reward is negative distance to goal
        
        # Check for collision with obstacles
        obs12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        obs13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        obs14 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
        obs15 = np.sqrt((x1 - x5)**2 + (y1 - y5)**2)

        obs23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
        obs24 = np.sqrt((x2 - x4)**2 + (y2 - y4)**2)
        obs25 = np.sqrt((x2 - x5)**2 + (y2 - y5)**2)

        obs34 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
        obs35 = np.sqrt((x3 - x5)**2 + (y3 - y5)**2)

        obs45 = np.sqrt((x4 - x5)**2 + (y4 - y5)**2)

        distance_to_obstacle = np.maximum(obs12, np.maximum(obs13, np.maximum(obs14, np.maximum(obs15, np.maximum(obs23, np.maximum(obs24, np.maximum(obs25, np.maximum(obs34, np.maximum(obs35, obs45)))))))))
        if distance_to_obstacle <= self.collision_threshold:
            reward -= 10  # Large penalty for collision

        # Check if goal is reached
        if distance_to_goal <= self.goal_threshold:
            reward += 10  # Large reward for reaching the goal

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
                pygame.display.set_caption("Multiagent Environment")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()  # Corrected initialization

        self.screen.fill((255, 255, 255))  # Clear screen with white

        # Draw obstacles
        # for obstacle in self.obstacles:
        #     pygame.draw.circle(
        #         self.screen,
        #         (255, 0, 0),  # Red color
        #         self._world_to_screen(obstacle),
        #         int(self.collision_threshold * self.scale),
        #     )
        
        color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
        # Draw goal
        for i in range(5):
            goal_x, goal_y = self.state[10+2*i], self.state[11+2*i]
            pygame.draw.circle(
               self.screen,
               color[i],  # Green color
                self._world_to_screen((goal_x, goal_y)),
                int(self.goal_threshold * self.scale/5),
                )
        
        # Draw unicycle
        for i in range(5):
            x, y = self.state[2*i], self.state[2*i]
            pygame.draw.circle(
               self.screen,
               color[i],  # Green color
                self._world_to_screen((x, y)),
                int(self.collision_threshold * self.scale),
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
        screen_x = int((x + 1) * self.scale)  # Shift and scale
        screen_y = int((1 - y) * self.scale)  # Flip y-axis
        return (screen_x, screen_y)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None