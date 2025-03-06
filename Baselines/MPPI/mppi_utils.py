import torch
import time
import math
import pandas as pd
import numpy as np

def csv_to_tensor(file_path):
    """Load a CSV file and convert it to a PyTorch tensor."""
    df = pd.read_csv(file_path)
    np_array = df.to_numpy(dtype=np.float32)
    tensor = torch.tensor(np_array).T
    return tensor

def signed_distance_to_circle(points, circle, device):
    cx, cy = circle[0], circle[1]
    radius = circle[2]
    
    dp = points[..., 0:2]*1.0
    dp[..., 0] = dp[..., 0]-(cx)
    dp[..., 1] = dp[..., 1]-(cy)
    dist = torch.norm(dp[..., 0:2], dim=-1) - radius
    
    signed_distance = torch.where(dist<=0, -dist, 0*dist)
    return signed_distance

def dyn_propogate(x, u, dynamics, dt, device):
    dx = dynamics(x, u, device)
    x_next = x + dx * dt
    return x_next

def batch_rollout_traj(x0, u, dynamics, dt, device):
    #           num_rollouts, planning_horizon, shape of x
    x = torch.zeros(u.shape[0], u.shape[1], x0.shape[1], device=device)
    x[:, 0, :] = x0
    for i in range(1, u.shape[1]):
        x[:, i, :] = dyn_propogate(x[:, i-1, :], u[:, i-1, :], dynamics, dt, device)
    return x

def mppi_control(u, weights, device):
    w = torch.zeros(u.shape, device=device)
    w[:, :, :] = weights.unsqueeze(1).unsqueeze(2)
    u_new = torch.sum(w * u, dim=0) / torch.sum(w, dim=0)
    return u_new    

def normalize_samples(samples):
    magnitude = torch.norm(samples, dim=-1)
    mask = magnitude > 1
    normalized_samples = torch.where(mask.unsqueeze(-1), samples / magnitude.unsqueeze(-1), samples)
    return normalized_samples

def clip_samples(u):
    clipped_u = torch.clamp(u, min=-2, max=2)
    # print("U",u, clipped_u)
    return clipped_u

def run_batch_mppi(device, x0, dynamics, mppi_cost_func, running_cost, obs_cost, total_time_steps, num_rollouts, planning_horizon, dt, goal, obs, softmax_lambda, u_dim=2, norm_flag =False, clip_flag =False):
    running_costs = torch.zeros(total_time_steps, device=device)
    traj = [x0]
    control_traj = []
    x_dim = x0.shape[1]
    # print('x_dim',x_dim)
    mean_u = torch.zeros(num_rollouts, planning_horizon, u_dim, device=device)
    for step in range(0, total_time_steps):
        u = torch.normal(mean_u)
        if norm_flag == True:
            u = normalize_samples(u)*2

        if clip_flag == True:
            u = clip_samples(u)
        current_x = traj[-1]
        # print(current_x.shape, u.shape)
        # running_cost = math.sqrt((current_x[0,0] - goalX)**2 + (current_x[0, 1] - goalY)**2)
        running_costs[step] = running_cost(torch.Tensor(current_x), goal, device).to('cpu').detach() + 1e10*obs_cost(torch.Tensor(current_x), obs, device).to('cpu').detach()#.numpy()#obs_distance*1e12
        next_x = batch_rollout_traj(current_x, u, dynamics, dt, device)
        cost = mppi_cost_func(next_x, None, goal, obs, device)
        best_u = mppi_control(u, torch.softmax(-softmax_lambda * cost, dim=0), device)
        best_x = dyn_propogate(current_x, best_u[0, :].reshape(1, u_dim), dynamics, dt, device)
        traj.append(best_x)
        control_traj.append(best_u[0, :].reshape(1, u_dim))

    terminal_cost= running_cost(torch.Tensor(current_x), goal, device).to('cpu').detach()
    traj = torch.stack(traj, dim=1).reshape(total_time_steps + 1, x_dim)
    control_traj = torch.stack(control_traj, dim=1).reshape(total_time_steps, u_dim)
    return torch.trapezoid(running_costs, dx=dt).to('cpu').detach()+terminal_cost, traj, control_traj
