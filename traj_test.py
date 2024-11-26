import torch
from tqdm.autonotebook import tqdm
from dynamics.dynamics import Boat2DAug
import os
from utils import modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle

def traj_test(model, initial_state, dynamics, dt = 0.0025, tMax = 10, tMin = 0):
    policy = model
    state_trajs = torch.zeros(1, int(
        (tMax-tMin)/dt + 1), dynamics.state_dim)
    ctrl_trajs = torch.zeros(1, int(
        (tMax-tMin)/dt), dynamics.control_dim)
    dstb_trajs = torch.zeros(1, int(
        (tMax-tMin)/dt), dynamics.disturbance_dim)
    ham_trajs = torch.zeros(1, int((tMax-tMin)/dt))


    state_trajs[:, 0, :] = initial_state
    pbar_pos = 0
    for k in tqdm(range(int((tMax-tMin)/dt)), desc='Trajectory Propagation', position=pbar_pos, leave=False):
        traj_time = tMax - k*dt
        traj_times = torch.full((1, ), traj_time)
        
        traj_coords = torch.cat(
            (traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)
        traj_policy_results = policy(
            {'coords': dynamics.coord_to_input(traj_coords.cuda())})
        traj_dvs = dynamics.io_to_dv(
            traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

        # TODO: I do not think there is actually any reason to store these trajs? Could save space by removing these.

        ctrl_trajs[:, k] = dynamics.optimal_control(
            traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        dstb_trajs[:, k] = dynamics.optimal_disturbance(
            traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        ham_trajs[:, k] = dynamics.hamiltonian(
            traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
        
        next_state_ = dynamics.equivalent_wrapped_state(state_trajs[:, k].cuda(
        ) + dt*dynamics.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda(), dstb_trajs[:, k].cuda()))
        next_state_ = torch.clamp(next_state_, torch.tensor(dynamics.state_test_range(
        )).cuda()[..., 0], torch.tensor(dynamics.state_test_range()).cuda()[..., 1])

        state_trajs[:, k+1] = next_state_
        pbar_pos +=1

    traj = state_trajs[0].T
    x, y, z = traj

    print(z, ctrl_trajs[0])

    x = x.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()
    z = z.to('cpu').detach().numpy()

    

    plt.xlim(-3, 2)
    plt.ylim(-2, 2)
    currentAxis1 = plt.gca()
    currentAxis1.add_patch(Rectangle((-0.7, 0.3), 0.4, 0.4, facecolor = 'orange', alpha=1))
    currentAxis2 = plt.gca()
    currentAxis2.add_patch(Rectangle((-1.1, -2), 0.2, 1, facecolor = 'orange', alpha=1))
    currentAxis3 = plt.gca()
    currentAxis3.add_patch(Circle((1.5, 0), 0.25, facecolor = 'cyan', alpha=1))
    plt.scatter(x, y, s=1)
    plt.savefig("traj_plot.png",dpi=1200)    
    plt.show()    



if __name__ =="__main__":

    dyn = Boat2DAug()

    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
    model_path = os.path.join('runs/Boat2D_s0_Aug', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    initial_state = torch.Tensor([-2,-1.5, 10]).to('cuda')
    traj_test(model, initial_state, dynamics=dyn)



