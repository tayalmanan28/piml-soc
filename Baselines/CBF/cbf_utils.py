import torch
import pandas as pd
import numpy as np

def signed_distance_to_circle(points, circle, device):
    cx, cy = circle[0], circle[1]
    radius = circle[2]
    
    dp = points[..., 0:2]*1.0
    dp[..., 0] = dp[..., 0]-(cx)
    dp[..., 1] = dp[..., 1]-(cy)
    dist = torch.norm(dp[..., 0:2], dim=-1)**2 - radius**2
    
    # signed_distance = torch.where(dist<=0, -dist, 0*dist)
    return dist

def csv_to_tensor(file_path):
    """Load a CSV file and convert it to a PyTorch tensor."""
    df = pd.read_csv(file_path)
    np_array = df.to_numpy(dtype=np.float32)
    tensor = torch.tensor(np_array).T
    return tensor

def dyn_propogate(x, u, f, g, dt, device):
    dx = f(x, device) + g(x, device)*u
    x_next = x + dx * dt
    return x_next

def calc_safe_u(x, h, d_h, goal_cost, f, g, gamma, u_dim, device):
    u_ref = 0.1*goal_cost(x).to(device)
    u_safe = 0*u_ref
    x_dim = x.shape
    # print(h.shape)
    h = h.reshape((-1, 1, 1))
    d_h = d_h.reshape((-1, 1, x_dim))
    f_x = f(x, device).reshape((-1, x_dim, 1)).to(device)
    g_x = g(x, u_dim, device).reshape((-1, x_dim, u_dim)).to(device)
    # print(f_x.shape, g_x.shape, d2_h.shape, x.shape, h.shape)
    A = d_h@f_x
    B = d_h@g_x
    
    psi = A + torch.bmm(B,u_ref) + gamma*h 
    u_safe = - psi/B
    u_safe[psi>=0] = 0
    u = u_ref + u_safe
    
    return u

def normalize_samples(samples):
    magnitude = torch.norm(samples, dim=-1)
    mask = magnitude > 1
    normalized_samples = torch.where(mask.unsqueeze(-1), samples / magnitude.unsqueeze(-1), samples)
    return normalized_samples

def clip_samples(u):
    clipped_u = torch.clamp(u, min=-2, max=2)
    # print("U",u, clipped_u)
    return clipped_u

def run_batch_cbf(device, x0, h, d_h, f, g, running_cost, obs_cost, total_time_steps, dt, goal, obs, u_dim=2, norm_flag =False, clip_flag =False):
    running_costs = torch.zeros(total_time_steps, device=device)
    traj = [x0]
    control_traj = []
    x_dim = x0.shape[1]
    u = torch.zeros(u_dim, device=device)
    for step in range(0, total_time_steps):
        current_x = traj[-1]
        # print(current_x.shape, u.shape)
        # running_cost = math.sqrt((current_x[0,0] - goalX)**2 + (current_x[0, 1] - goalY)**2)
        running_costs[step] = running_cost(torch.Tensor(current_x), goal, device).to('cpu').detach() + obs_cost(torch.Tensor(current_x), obs, device).to('cpu').detach()#.numpy()#obs_distance*1e12
        
        safe_u = calc_safe_u(current_x, h, d_h, running_cost, f, g, gamma=1, u_dim=u_dim)
        next_x = dyn_propogate(current_x, safe_u, f, g, dt, device)
        traj.append(next_x)
        control_traj.append(safe_u.reshape(1, u_dim))

    terminal_cost= running_cost(torch.Tensor(current_x), goal, device).to('cpu').detach()
    traj = torch.stack(traj, dim=1).reshape(total_time_steps + 1, x_dim)
    control_traj = torch.stack(control_traj, dim=1).reshape(total_time_steps, u_dim)
    return torch.trapezoid(running_costs, dx=dt).to('cpu').detach()+terminal_cost, traj, control_traj
