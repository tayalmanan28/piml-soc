from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from boat import BoatEnv

# Create the environment with render_mode='human'
env = BoatEnv()
env = make_vec_env(
    lambda: env,
    n_envs=1,
)


# Initialize the SAC agent
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_boat_tensorboard/")

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("sac_boat_1")

env = make_vec_env(
    lambda: BoatEnv(render_mode='human'),  # Pass render_mode here
    n_envs=1,
)

# Optional: Load the model
# model = SAC.load("sac_unicycle")

# Test the trained agent with rendering
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')  # Render the environment
    if dones.any():  # Check if any environment is done
        obs = env.reset()

env.close()