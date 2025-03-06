from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from multiagent import MultiAgentEnv
import numpy as np

from anim5V import animate_trajectories

# Create the environment with render_mode='human'
env = make_vec_env(
    lambda: MultiAgentEnv(), 
    n_envs=1,
)

# Initialize the SAC agent
# model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_multiagent_tensorboard/")

# Train the agent
# model.learn(total_timesteps=10000)

# Save the model
# model.save("sac_multiagent_trial")

# # Optional: Load the model
model = SAC.load("Baselines/SAC/sac_multiagent")
# env = make_vec_env(
#     lambda: MultiAgentEnv(render_mode='human'),  # Pass render_mode here
#     n_envs=1,
# )

trajectories = []

# Test the trained agent with rendering
obs = env.reset()
for _ in range(800):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    trajectories.append(obs[0])
    # env.render(mode='human')  # Render the environment
    if dones.any():  # Check if any environment is done
        obs = env.reset()

trajectories = np.array(trajectories)
print(trajectories)
x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5 = trajectories.T
animate_trajectories(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5)

env.close()