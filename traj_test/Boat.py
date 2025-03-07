import torch
from tqdm.autonotebook import tqdm
from dynamics.Boat2DAug import Boat2DAug
import os
from utils import modules
from final_value import opt_value_func_mesh as opt_value
from utils.animation.anim_Boat import animate_trajectory

def traj_test(model, initial_state, traj_num, dynamics, dt = 0.0025, tMax = 2, tMin = 0):
    policy = model
    state_trajs = torch.zeros(1, int((tMax-tMin)/dt + 1), dynamics.state_dim)
    ctrl_trajs = torch.zeros(1, int((tMax-tMin)/dt), dynamics.control_dim)
    dstb_trajs = torch.zeros(1, int((tMax-tMin)/dt), dynamics.disturbance_dim)
    ham_trajs = torch.zeros(1, int((tMax-tMin)/dt))

    state_trajs[:, 0, :] = initial_state
    pbar_pos = 0
    traj_time_list = [0.0]
    Z = initial_state[2]
    Z = Z.to('cpu').detach().numpy()
    cost = 0.0
    for k in tqdm(range(int((tMax-tMin)/dt)), desc='Trajectory Propagation', position=pbar_pos, leave=False):
        traj_time = tMax - k*dt
        traj_time_list.append(2.0-traj_time)
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

        if (traj_time <= dt):
            cost = cost + dynamics.l_x(state_trajs[:, k+1])
            print("Terminal_Cost", dynamics.l_x(state_trajs[:, k+1]))
            print("Total_Cost", cost)

        state_trajs[:, k+1] = next_state_
        pbar_pos +=1

    traj = state_trajs[0].T
    x, y, z = traj
    cost_fn = dynamics.cost_fn(state_trajs)
    if cost_fn > 0:
        print("Avoid Cost",cost_fn)

    x = x.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()
    z = z.to('cpu').detach().numpy()

    animate_trajectory(x, y, traj_num=traj_num)

if __name__ =="__main__":

    dyn = Boat2DAug()

    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=256, num_hidden_layers=3)
    model_path = os.path.join('runs/Boat2DAug', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    times = torch.Tensor([2.0])
    states = torch.Tensor(( 
                            [-2.07,-1.43, 0.0],
                            [-2.20, 1.890, 0.0],
                            [-2.34,-1.653, 0.0],
                            [-2.07,-1.425, 0.0],
                            [-2.67,-1.825, 0.0],
                            [-1.33, 1.510, 0.0],
                            [ 0.69, 1.100, 0.0],
                            [-1.93, 0.060, 0.0],
                            [-2.33,-0.520, 0.0],
                            [0.068, 0.870, 0.0],
                            [-0.52,-0.140, 0.0],
                            [-0.63,-1.210, 0.0],
                            [-2.58, 0.670, 0.0]
                            )).to('cuda')
    
    for i in range(len(states)):
        act_state = states[i][0:2]
        z_range = torch.Tensor([0, 14.86])
        Z = opt_value(model, dyn, times, act_state, z_range, resolution=1, num_z=210, delta=-0.02)
        states[i][2] = Z
        print("Optimal Z", Z)
        traj_test(model, states[i], dynamics=dyn, traj_num = i)