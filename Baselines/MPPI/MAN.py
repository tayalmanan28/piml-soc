import torch
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import math
import pandas as pd
from mppi_utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
from anim5V import animate_trajectories

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
    dx[:, 0] = u[:, 0]
    dx[:, 1] = u[:, 1]
    dx[:, 2] = u[:, 2]
    dx[:, 3] = u[:, 3]
    dx[:, 4] = u[:, 4]
    dx[:, 5] = u[:, 5]
    dx[:, 6] = u[:, 6]
    dx[:, 7] = u[:, 7]
    dx[:, 8] = u[:, 8]
    dx[:, 9] = u[:, 9]
    return dx

def running_cost(x, goal, device):
    x = x.to(device)
    r = 0.0
    goal_dist_1 = torch.norm(x[..., 0:2] - x[..., 10:12], dim=-1) - r
    goal_dist_2 = torch.norm(x[..., 2:4] - x[..., 12:14], dim=-1) - r
    goal_dist_3 = torch.norm(x[..., 4:6] - x[..., 14:16], dim=-1) - r
    goal_dist_4 = torch.norm(x[..., 6:8] - x[..., 16:18], dim=-1) - r
    goal_dist_5 = torch.norm(x[..., 8:10] -x[..., 18:20], dim=-1) - r
    goal_dist = (goal_dist_1 + goal_dist_2 + goal_dist_3 + goal_dist_4 + goal_dist_5)/5
    # print(x, goal_dist)
    return goal_dist

def obs_cost(x, obs, device):
    x = x.to(device)
    r = 0.1
    obs_dist_12 = -torch.norm(x[..., 0:2] - x[..., 2:4], dim=-1) + r
    obs_dist_13 = -torch.norm(x[..., 0:2] - x[..., 4:6], dim=-1) + r
    obs_dist_14 = -torch.norm(x[..., 0:2] - x[..., 6:8], dim=-1) + r
    obs_dist_15 = -torch.norm(x[..., 0:2] - x[..., 8:10], dim=-1) + r
    
    obs_dist_23 = -torch.norm(x[..., 2:4] - x[..., 4:6], dim=-1) + r
    obs_dist_24 = -torch.norm(x[..., 2:4] - x[..., 6:8], dim=-1) + r
    obs_dist_25 = -torch.norm(x[..., 2:4] - x[..., 8:10], dim=-1) + r
    
    obs_dist_34 = -torch.norm(x[..., 4:6] - x[..., 6:8], dim=-1) + r
    obs_dist_35 = -torch.norm(x[..., 4:6] - x[..., 8:10], dim=-1) + r
    
    obs_dist_45 = -torch.norm(x[..., 6:8] - x[..., 8:10], dim=-1) + r
    
    obs_dist_1 = torch.max(obs_dist_15,torch.max(obs_dist_14, torch.max(obs_dist_12, obs_dist_13)))
    obs_dist_2 = torch.max(obs_dist_25, torch.max(obs_dist_24, obs_dist_23))
    obs_dist_3 = torch.max(obs_dist_34, obs_dist_35)

    obs_dist = torch.max(torch.Tensor([0.0]).to(device),torch.max(obs_dist_1,torch.max(obs_dist_2, torch.max(obs_dist_3, obs_dist_45))))
    return  0.5 *obs_dist

if __name__ =="__main__":
    file_path = "plots/MAN/Traj_points.csv"
    dt = 0.0025
    horizon = 2
    total_time_steps = int(horizon / dt)
    planning_horizon = 40
    num_rollouts = 8000
    softmax_lambda = 200.0
    goal = []
    obstacles = []

    # Load rollout points
    rollout_points = csv_to_tensor(file_path)[0:20].to(device).T[0:2]
    rollout_points = torch.Tensor(([0.5, 0.0, 0.155, 0.475,-0.405, 0.294,-0.405,-0.294,0.155,-0.475, -0.405, 0.0,-0.125, -0.385, 0.327,-0.237, 0.327,0.237,-0.125, 0.385],
                            )).to('cuda')
    rollout_costs = torch.zeros(rollout_points.shape[0], device=device)
    for i in range(rollout_points.shape[0]):
        print(f"Rollout {i+1} of {rollout_points.shape[0]}")        
        x0 = rollout_points[i, :].reshape(1, 20)
        rollout_cost, trajectory, _ = run_batch_mppi(device, x0, dynamics, mppi_cost_func, running_cost, obs_cost, total_time_steps, num_rollouts, planning_horizon, dt, goal, obstacles, softmax_lambda, u_dim=10, norm_flag =True)
        rollout_costs[i] = rollout_cost.item()

        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5 = trajectory.T
        x1 = x1.to('cpu').detach().numpy()
        y1 = y1.to('cpu').detach().numpy()
        x2 = x2.to('cpu').detach().numpy()
        y2 = y2.to('cpu').detach().numpy()
        x3 = x3.to('cpu').detach().numpy()
        y3 = y3.to('cpu').detach().numpy()
        x4 = x4.to('cpu').detach().numpy()
        y4 = y4.to('cpu').detach().numpy()
        x5 = x5.to('cpu').detach().numpy()
        y5 = y5.to('cpu').detach().numpy()
        gx_1 = gx_1.to('cpu').detach().numpy()
        gy_1 = gy_1.to('cpu').detach().numpy()
        gx_2 = gx_2.to('cpu').detach().numpy()
        gy_2 = gy_2.to('cpu').detach().numpy()
        gx_3 = gx_3.to('cpu').detach().numpy()
        gy_3 = gy_3.to('cpu').detach().numpy()
        gx_4 = gx_4.to('cpu').detach().numpy()
        gy_4 = gy_4.to('cpu').detach().numpy()
        gx_5 = gx_5.to('cpu').detach().numpy()
        gy_5 = gy_5.to('cpu').detach().numpy()

        animate_trajectories(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5)

        plt.figure(1)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.title("MPPI Trajectories")
        currentAxis1 = plt.gca()
        currentAxis1.add_patch(Circle((x0[0][0], x0[0][1]), 0.01, facecolor = 'blue', alpha=1))
        currentAxis2 = plt.gca()
        currentAxis2.add_patch(Circle((x0[0][2], x0[0][3]), 0.01, facecolor = 'orange', alpha=1))
        currentAxis3 = plt.gca()
        currentAxis3.add_patch(Circle((x3[0], y3[0]), 0.01, facecolor = 'green', alpha=1))
        currentAxis4 = plt.gca()
        currentAxis4.add_patch(Circle((x0[0][6], x0[0][7]), 0.01, facecolor = 'red', alpha=1))
        currentAxis5 = plt.gca()
        currentAxis5.add_patch(Circle((x0[0][8], x0[0][9]), 0.01, facecolor = 'cyan', alpha=1))

        currentAxis1 = plt.gca()
        currentAxis1.add_patch(Circle((gx_1[0], gy_1[0]), 0.01, facecolor = 'blue', alpha=1))
        currentAxis2 = plt.gca()
        currentAxis2.add_patch(Circle((gx_2[0], gy_2[0]), 0.01, facecolor = 'orange', alpha=1))
        currentAxis3 = plt.gca()
        currentAxis3.add_patch(Circle((gx_3[0], gy_3[0]), 0.01, facecolor = 'green', alpha=1))
        currentAxis4 = plt.gca()
        currentAxis4.add_patch(Circle((gx_4[0], gy_4[0]), 0.01, facecolor = 'red', alpha=1))
        currentAxis5 = plt.gca()
        currentAxis5.add_patch(Circle((gx_5[0], gy_5[0]), 0.01, facecolor = 'cyan', alpha=1))
        plt.scatter(x1, y1, s=0.1)
        plt.scatter(x2, y2, s=0.1)
        plt.scatter(x3, y3, s=0.1)
        plt.scatter(x4, y4, s=0.1)
        plt.scatter(x5, y5, s=0.1)
        plt.savefig("plots/MAN/mppi.png",dpi=1200) 

    rollout_costs = rollout_costs.cpu()
    # Convert tensors to NumPy and save to CSV
    pd.DataFrame(rollout_costs.numpy()).to_csv('plots/MAN/mppi_costs.csv', index=False, header=["rollout_costs"])
    print(f"Time taken is {time.time() - start_time:.2f} seconds")