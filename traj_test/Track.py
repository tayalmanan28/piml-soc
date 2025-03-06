import torch
from tqdm.autonotebook import tqdm
from dynamics.Track8DAug import Track8DAug_norm
import os
from utils import modules
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from final_value import opt_value_func_mesh as opt_value

def traj_test(model, initial_state, dynamics, dt = 0.0025, tMax = 6.0, tMin = 0):
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
        traj_time = 1
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

        state_trajs[:, k+1] = next_state_
        pbar_pos +=1

    traj = state_trajs[0].T
    x, y, v, psi, gx, gy, vgx, vgy, z = traj
    cost_fn = dynamics.cost_fn(state_trajs)
    print("Avoid cost",cost_fn)

    x = x.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()
    v = v.to('cpu').detach().numpy()
    psi = psi.to('cpu').detach().numpy()
    gx = gx.to('cpu').detach().numpy()
    gy = gy.to('cpu').detach().numpy()
    vgx = vgx.to('cpu').detach().numpy()
    vgy = vgy.to('cpu').detach().numpy()

    z = z.to('cpu').detach().numpy()
    print("Actual Cost",cost)

    plt.figure(2)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("VDR Trajectories")
    currentAxis = plt.gca()
    currentAxis.add_patch(Circle(( 0.5, 0.5), 0.2, facecolor = '#6B6B6B', alpha=1))
    currentAxis.add_patch(Circle(( 0.5,-0.5), 0.2, facecolor = '#6B6B6B', alpha=1))
    currentAxis.add_patch(Circle(( 0.0, 0.0), 0.2, facecolor = '#6B6B6B', alpha=1))
    currentAxis.add_patch(Circle((-0.5, 0.5), 0.2, facecolor = '#6B6B6B', alpha=1))
    currentAxis.add_patch(Circle((-0.5,-0.5), 0.2, facecolor = '#6B6B6B', alpha=1))
    plt.scatter(x, y, s=1, color = '#FF9500')
    plt.scatter(gx, gy, s=1.0, color='#740cad')
    plt.savefig("plots/Track/traj_plot_paper.png",dpi=1200)       

if __name__ =="__main__":

    dyn = Track8DAug_norm()
    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=256, num_hidden_layers=3)
    model_path = os.path.join('runs/Track8DAug', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    # initial_state = torch.Tensor([-0.63, -1.21, 2.7809]).to('cuda')
    # traj_test(model, initial_state, dynamics=dyn)
    times = torch.Tensor([1.0])
                        #    x1    z
    states = torch.Tensor(([-0.80, 0.40, 0.50,-0.03,-0.20,-0.20,-0.10, 0.00, 0],
                           [ 0.00,-0.50, 0.60, 0.00, 0.75,-0.40,-0.15, 0.15, 0],
                           [ 0.53, 0.20, 0.60, 1.20, 0.70, 0.76,-0.20, 0.00, 0],
                           [-0.90,-0.50, 0.00,-1.50,-0.50,-0.50, 0.08,-0.08, 0],
                           [-0.80,-0.50, 1.23,-1.20,-0.70,-0.30, 0.12,-0.12, 0],
                            )).to('cuda')

    for i in range(len(states)):
        act_state = states[i][0:8]
        z_range = torch.Tensor([0, 4.3])
        Z = opt_value(model, dyn, times, act_state, z_range, resolution=1, num_z=210, delta=-0.0)
        states[i][8] = Z
        print("Optimal Z", Z)
        traj_test(model, states[i], dynamics=dyn)