import torch
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import math
import pandas as pd
from mppi_utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()

def mppi_cost_func(x, u, goal, obs, device):
    # goalX, goalY, goalR = goal
    obs_cost_ = obs_cost(x, obs, device)
    # for i in range(1, x.shape[1]):
    #     obs_cost_ += obs_cost(x[:, i, :], goal, device)
    cost = torch.zeros(x.shape[0])
    # x1 = x[:, :, 0]
    # x2 = x[:, :, 1]
    # print(x.shape)
    goal_distance = running_cost(x, goal, device)#torch.sqrt((x1 - goalX)**2 + (x2 - goalY)**2)
    # for i in range(1, x.shape[1]):
    #     goal_distance += running_cost(x[:, i, :], goal, device)
    cost = goal_distance + obs_cost_
    # print(cost.shape)
    cost = torch.sum(cost, dim=1)
    return cost

def dynamics(x, u, device):
    dx = torch.zeros_like(x, device=device)
    dx[:, 0] = x[:, 2]*torch.cos(x[:, 3])
    dx[:, 1] = x[:, 2]*torch.sin(x[:, 3])
    dx[:, 2] = u[:, 0]
    dx[:, 3] = u[:, 1]
    dx[:, 4] = x[:, 6]
    dx[:, 5] = x[:, 7]
    return dx

def running_cost(x, goal, device):
    x = x.to(device)
    r = 0.0
    goal_dist = torch.norm(x[..., 0:2] - x[..., 4:6], dim=-1) - r
    # print(x)

    return goal_dist

def obs_cost(x, obs, device):
    # obs1_distance=0.4 - math.sqrt((current_x[0,0] - (-0.5))**2 + (current_x[0, 1] - (0.5))**2)
    # obs2_distance=0.5 - math.sqrt((current_x[0,0] - (-1))**2 + (current_x[0, 1] - (-1.2))**2)
    
    # obs_distance = max(0.0,obs1_distance, obs2_distance)
    x = x.to(device)
    obs_distance = torch.Tensor([0.0]).to(device)
    for i in range(len(obs)):
        obsi_x, obsi_y, obsi_r = obs[i]
        obsi_distance = signed_distance_to_circle(x, torch.tensor([obsi_x, obsi_y, obsi_r]), device)
        obs_distance = torch.max(obsi_distance, obs_distance)

    return  1 *obs_distance

if __name__ =="__main__":
    file_path = "plots/Track/Traj_points.csv"
    dt = 0.0025
    horizon = 1
    total_time_steps = int(horizon / dt)
    planning_horizon = 20
    num_rollouts = 8000
    softmax_lambda = 200.0
    goal = []
    obstacles = [[0.5, 0.5, 0.2], [-0.5, 0.5, 0.2], [-0.5,-0.5, 0.2], [ 0.5,-0.5, 0.2], [ 0.0, 0.0, 0.2]]

    # Load rollout points
    rollout_points = csv_to_tensor(file_path)[0:8].to(device).T
    rollout_points = torch.Tensor(([-0.8, 0.5, 1.3, -0.65, -0.2, -0.2, -0.1, 0.0],
                                  [0.0, -0.5, 1.2, 0.0, 0.75, -0.4, -0.35, 0.35])).to('cuda')
    rollout_costs = torch.zeros(rollout_points.shape[0], device=device)
    for i in range(rollout_points.shape[0]):
        print(f"Rollout {i+1} of {rollout_points.shape[0]}")        
        x0 = rollout_points[i, :].reshape(1, 8)
        rollout_cost, trajectory, _ = run_batch_mppi(device, x0, dynamics, mppi_cost_func, running_cost, obs_cost, total_time_steps, num_rollouts, planning_horizon, dt, goal, obstacles, softmax_lambda, u_dim=2, norm_flag =False, clip_flag =True)
        rollout_costs[i] = rollout_cost.item()

        x, y, v, psi, gx, gy, vgx, vgy = trajectory.T
        x = x.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()
        v = v.to('cpu').detach().numpy()
        psi = psi.to('cpu').detach().numpy()
        gx = gx.to('cpu').detach().numpy()
        gy = gy.to('cpu').detach().numpy()
        vgx = vgx.to('cpu').detach().numpy()
        vgy = vgy.to('cpu').detach().numpy()

        # plt.figure(1)
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.title("MPPI Trajectories")
        # currentAxis1 = plt.gca()
        # currentAxis1.add_patch(Circle((0.5, 0.5), 0.2, facecolor = 'blue', alpha=1))
        # currentAxis2 = plt.gca()
        # currentAxis2.add_patch(Circle((0.5, -0.5), 0.2, facecolor = 'orange', alpha=1))
        # currentAxis3 = plt.gca()
        # currentAxis3.add_patch(Circle((0.0, 0.0), 0.2, facecolor = 'green', alpha=1))
        # currentAxis4 = plt.gca()
        # currentAxis4.add_patch(Circle((-0.5, 0.5), 0.2, facecolor = 'red', alpha=1))
        # currentAxis5 = plt.gca()
        # currentAxis5.add_patch(Circle((-0.5, -0.5), 0.2, facecolor = 'cyan', alpha=1))
        # plt.scatter(x, y, s=1)
        # plt.scatter(gx, gy, s=1.0)
        # plt.scatter(x[0], y[0], s=20)
        # plt.scatter(gx[0], gy[0], s=20)
        # plt.savefig("plots/Track/sac.png",dpi=1200) 
        # pd.DataFrame(x).to_csv('plots/Track/SAC_traj_x'+str(i)+'.csv', index=False, header=["x"])
        # pd.DataFrame(y).to_csv('plots/Track/SAC_traj_y'+str(i)+'.csv', index=False, header=["y"])

    rollout_costs = rollout_costs.cpu()
    # Convert tensors to NumPy and save to CSV
    # pd.DataFrame(rollout_costs.numpy()).to_csv('plots/Track/mppi_costs.csv', index=False, header=["rollout_costs"])
    print(f"Time taken is {time.time() - start_time:.2f} seconds")