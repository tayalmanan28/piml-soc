from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from unicycle import UnicycleEnv, csv_to_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle, Ellipse

# Create the environment with render_mode='human'
env = make_vec_env(
    lambda: UnicycleEnv(render_mode='human'),  # Pass render_mode here
    n_envs=1,
)

# Initialize the SAC agent
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Baselines/SAC/sac_unicycle_tensorboard/")

_, states = csv_to_array("plots/Track/Traj_points.csv")
# Test the trained agent with rendering
unsafe = 0
unsafe_idx = []
for i in range(len(states)):
    obs = env.reset()
    obs = [states[i][0:4]]
    print(obs)
    rew = 0.0
    unsafe_flag = 0
    x = []
    y = []
    for _ in range(800):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        rew -= rewards*0.0025
        if (dones.any() == True) and ( _ < 398) and (unsafe_flag == 0):
            unsafe = unsafe + 1
            unsafe_idx.append(i)
            unsafe_flag = 1
            print("Unsafe")
        x.append(obs[0][0])
        y.append(obs[0][1]) 
        # print(action)
        # env.render(mode='human')  # Render the environment
        # if dones.any():  # Check if any environment is done
        #     obs = env.reset()    
    print(rew)
    rew = 0.0
    plt.xlim(-3, 2)
    plt.ylim(-2, 2)
    plt.title("VDR Trajectories")
    currentAxis1 = plt.gca()
    currentAxis1.add_patch(Circle((-0.5, 0.5), 0.4, facecolor = 'orange', alpha=1))
    currentAxis2 = plt.gca()
    currentAxis2.add_patch(Circle((-1.0, -1.2), 0.5, facecolor = 'orange', alpha=1))
    currentAxis3 = plt.gca()
    currentAxis3.add_patch(Circle((1.5, 0), 0.025, facecolor = 'cyan', alpha=1))
    plt.scatter(x, y, s=1)
    plt.savefig("plots/Track/sac.png",dpi=1200) 

    env.close()
print(unsafe, unsafe_idx)