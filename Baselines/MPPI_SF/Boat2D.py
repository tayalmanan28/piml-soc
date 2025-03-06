import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle, Ellipse
import math
import pandas as pd
import numpy as np
from mppi_utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
from CBF_Boat import *

def mppi_cost_func(x, u, goal, obs, device):
    goalX, goalY, goalR = goal
    obs1_x, obs1_y, obs1_r = obs[0]
    obs2_x, obs2_y, obs2_r = obs[1]
    cost = torch.zeros(x.shape[0])
    x1 = x[:, :, 0]
    x2 = x[:, :, 1]
    goal_distance = torch.sqrt((x1 - goalX)**2 + (x2 - goalY)**2)
    obs1_distance = signed_distance_to_circle(x, torch.tensor([obs1_x, obs1_y, obs1_r]), device)
    obs2_distance = signed_distance_to_circle(x, torch.tensor([obs2_x, obs2_y, obs2_r]), device)
    obs_distance = torch.max(obs1_distance, obs2_distance)

    cost = goal_distance + 1e12 * obs_distance
    # print(cost.shape)
    cost = torch.sum(cost, dim=1)
    return cost

def dynamics(x, u, device):
    dx = torch.zeros_like(x, device=device)
    dx[:, 0] = u[:, 0] + 2 - 0.5 * x[:, 1] ** 2
    dx[:, 1] = u[:, 1]
    return dx

def running_cost(x, goal, device):
    goalX, goalY, goalR = goal
    x = x.to('cpu').detach()
    goal_dist = math.sqrt((x[0,0] - goalX)**2 + (x[0, 1] - goalY)**2)
    # print(x, goal_dist)
    return torch.Tensor([goal_dist]).to(device)

def obs_cost(current_x, obs, device):
    obs1_distance=0.4 - math.sqrt((current_x[0,0] - (-0.5))**2 + (current_x[0, 1] - (0.5))**2)
    obs2_distance=0.5 - math.sqrt((current_x[0,0] - (-1))**2 + (current_x[0, 1] - (-1.2))**2)
    
    obs_distance = max(0.0,obs1_distance, obs2_distance)

    return torch.Tensor([obs_distance]).to(device)

if __name__ =="__main__":
    file_path = "plots/Boat2D/Traj_points.csv"
    dt = 0.0025
    horizon = 2
    total_time_steps = int(horizon / dt)
    planning_horizon = 20
    num_rollouts = 800
    softmax_lambda = 200.0
    goal = [1.5, 0, 0.25]
    obstacles = [[-0.5, 0.5, 0.4], [-1, -1.2, 0.5]]

    # Load rollout points
    rollout_points = csv_to_tensor(file_path)[0:2].to(device).T
    rollout_points = torch.Tensor(( 
                            [-2.07,-1.43],
                            [-2.58, 0.670]
                            )).to('cuda')
    rollout_costs = torch.zeros(rollout_points.shape[0], device=device)
    for i in range(rollout_points.shape[0]):
        print(f"Rollout {i+1} of {rollout_points.shape[0]}")        
        x0 = rollout_points[i, :].reshape(1, 2)
        rollout_cost, trajectory, _ = run_batch_mppi(device, x0, dynamics, mppi_cost_func, running_cost, obs_cost, solve_qp, total_time_steps, num_rollouts, planning_horizon, dt, goal, obstacles, softmax_lambda)
        rollout_costs[i] = rollout_cost.item()

        x, y = trajectory.T
        x = x.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()

        plt.figure(1)
        plt.xlim(-3, 2)
        plt.ylim(-2, 2)
        plt.title("MPPI Trajectories")
        currentAxis1 = plt.gca()
        currentAxis2 = plt.gca()
        currentAxis3 = plt.gca()
        currentAxis1.add_patch(Circle((-0.5, 0.5), 0.4, facecolor = 'orange', alpha=1))
        currentAxis2.add_patch(Circle((-1.0,-1.2), 0.5, facecolor = 'orange', alpha=1))
        currentAxis3.add_patch(Circle(( 1.5, 0.0), 0.025, facecolor = 'cyan', alpha=1))
        plt.scatter(x, y, s=0.1)
        plt.savefig("plots/Boat2D/mppi_sf.png",dpi=1200) 
        pd.DataFrame(x).to_csv('plots/Boat2D/mppi_sf_traj_x'+str(i)+'.csv', index=False, header=["x"])
        pd.DataFrame(y).to_csv('plots/Boat2D/mppi_sf_traj_y'+str(i)+'.csv', index=False, header=["y"])

    rollout_costs = rollout_costs.cpu()
    # Convert tensors to NumPy and save to CSV
    # pd.DataFrame(rollout_costs.numpy()).to_csv('plots/Boat2D/rollout_costs_sf.csv', index=False, header=["rollout_costs"])
    
    print(f"Time taken is {time.time() - start_time:.2f} seconds")