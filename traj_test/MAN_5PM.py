import torch
from tqdm.autonotebook import tqdm
from dynamics.MANAug5PM import MANAug5PM
import os
import numpy as np
from utils import modules
from final_value import opt_value_func_mesh as opt_value
from animation.anim_MAN import animate_trajectories

init_points = []
actual_cost = []
count = 0

def traj_test(model, initial_state, dynamics, traj_num, dt = 0.0025, tMax = 2.0, tMin = 0):
    policy = model
    state_trajs = torch.zeros(1, int((tMax-tMin)/dt + 1), dynamics.state_dim)
    ctrl_trajs = torch.zeros(1, int((tMax-tMin)/dt), dynamics.control_dim)
    dstb_trajs = torch.zeros(1, int((tMax-tMin)/dt), dynamics.disturbance_dim)
    ham_trajs = torch.zeros(1, int((tMax-tMin)/dt))


    state_trajs[:, 0, :] = initial_state
    pbar_pos = 0
    traj_time_list = [0.0]
    cost = 0.0
    for k in tqdm(range(int((tMax-tMin)/dt)), desc='Trajectory Propagation', position=pbar_pos, leave=False):
        traj_time = tMax - k*dt
        traj_time_list.append(1.0-traj_time)
        traj_times = torch.full((1, ), traj_time)
        cost = cost + dynamics.l_x(state_trajs[:, k])*dt
        
        traj_coords = torch.cat((traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)
        traj_policy_results = policy({'coords': dynamics.coord_to_input(traj_coords.cuda())})
        traj_dvs = dynamics.io_to_dv(traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

        ctrl_trajs[:, k] = dynamics.optimal_control(traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        dstb_trajs[:, k] = dynamics.optimal_disturbance(traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        ham_trajs[:, k] = dynamics.hamiltonian(traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        
        next_state_ = dynamics.equivalent_wrapped_state(state_trajs[:, k].cuda(
        ) + dt*dynamics.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda(), dstb_trajs[:, k].cuda()))

        next_state_ = torch.clamp(next_state_, torch.tensor(dynamics.state_test_range(
        )).cuda()[..., 0], torch.tensor(dynamics.state_test_range()).cuda()[..., 1])

        state_trajs[:, k+1] = next_state_
        if (traj_time <= dt):
            cost = cost + dynamics.l_x(state_trajs[:, k+1])
            print("Terminal_Cost", dynamics.l_x(state_trajs[:, k+1]))
        pbar_pos +=1
    
    traj = state_trajs[0].T
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5, z = traj
    cost_fn = dynamics.cost_fn(state_trajs)
    print("Avoid cost", cost_fn)

    # Conversion to numpy arrays
    def to_numpy(tensor):
        return tensor.to('cpu').detach().numpy()

    variables = {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3,
        'x4': x4, 'y4': y4, 'x5': x5, 'y5': y5,
        'gx_1': gx_1, 'gy_1': gy_1, 'gx_2': gx_2, 'gy_2': gy_2,
        'gx_3': gx_3, 'gy_3': gy_3, 'gx_4': gx_4, 'gy_4': gy_4,
        'gx_5': gx_5, 'gy_5': gy_5
    }

    # Convert all variables to numpy
    for name, tensor in variables.items():
        variables[name] = to_numpy(tensor)

    # Create vehicle trajectories list
    num_vehicles = 5  # Update this if your vehicle count changes
    vehicle_trajectories = [
        np.column_stack((variables[f'x{i+1}'], variables[f'y{i+1}']))
        for i in range(num_vehicles)
    ]

    # Create goal positions list
    goal_positions = [
        (variables[f'gx_{i+1}'][0], variables[f'gy_{i+1}'][0])
        for i in range(num_vehicles)
    ]

    z = z.to('cpu').detach().numpy()
    cost = cost.to('cpu').detach().numpy()
    print("Actual Cost",cost)

    if cost_fn < 0 and z[0] < 10:
        init_points.append(initial_state.to('cpu').detach().numpy())
        actual_cost.append(cost)

    # Call the generalized animation function
    animate_trajectories(vehicle_trajectories=vehicle_trajectories, goal_positions=goal_positions, traj_number=traj_num)

if __name__ =="__main__":

    dyn = MANAug5PM()
    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=256, num_hidden_layers=3)
    model_path = os.path.join('runs/MANAug5PM', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    times = torch.Tensor([2.0])
                        #    x1   y1   x2    y2     x3      y3     x4     y4      x5   y5      gx1  gy1    gx2     gy2   gx3    gy3     gx4    gy4   gx5    gy5   z
    states = torch.Tensor(([0.5, 0.0, 0.155, 0.475,-0.405, 0.294,-0.405,-0.294,0.155,-0.475, -0.405, 0.0,-0.125, -0.385, 0.327,-0.237, 0.327,0.237,-0.125, 0.385, 0.0],
                            )).to('cuda')
    
    for i in range(len(states)): #range(0,1): #
        act_state = states[i][0:20]
        # print(act_state)
        
        z_range = torch.Tensor([0, 4.3])

        Z = opt_value(model, dyn, times, act_state, z_range, resolution=1, num_z=210, delta=-0.02)

        states[i][21] = Z
        print("Optimal Z", Z)
        traj_test(model, states[i], dynamics=dyn, traj_num=i)