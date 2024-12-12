import torch
from tqdm.autonotebook import tqdm
from dynamics.Boat2DAug import Boat2DAug
import os
from utils import modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle, Ellipse

def traj_test(model, initial_state, dynamics, dt = 0.0025, tMax = 2, tMin = 0):
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
        # print()

    traj = state_trajs[0].T
    x, y, z = traj

    x = x.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()
    z = z.to('cpu').detach().numpy()
    print(traj, ctrl_trajs[0].T)

    plt.xlim(-3, 2)
    plt.ylim(-2, 2)
    plt.title("ExactBC Trajectories")
    currentAxis1 = plt.gca()
    currentAxis1.add_patch(Rectangle((-0.9, 0.1), 0.8, 0.8, facecolor = 'orange', alpha=1))
    currentAxis2 = plt.gca()
    currentAxis2.add_patch(Rectangle((-1.2, -2.3), 0.4, 2, facecolor = 'orange', alpha=1))
    currentAxis3 = plt.gca()
    currentAxis3.add_patch(Circle((1.5, 0), 0.025, facecolor = 'cyan', alpha=1))
    plt.scatter(x, y, s=0.1)
    plt.savefig("Boat2D/traj_plot.png",dpi=1200)    
    # plt.show()    

if __name__ =="__main__":

    dyn = Boat2DAug()
    print("Yoo")

    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=256, num_hidden_layers=3)
    model_path = os.path.join('runs/Boat2DAug_BC', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    # initial_state = torch.Tensor([-0.63, -1.21, 2.7809]).to('cuda')
    # traj_test(model, initial_state, dynamics=dyn)
    # ExactBC
    states = torch.Tensor(([-2.44,-1.157, 10.0539],
                            [-2.20, 1.89, 6.2035],
                            [-1.33, 1.51, 3.8504],
                            [ 0.69, 1.10, 2.1391],
                            [-1.93, 0.06, 20.0],
                            [-2.33,-0.52, 10.9096],
                            [0.068, 0.87, 1.9252],
                            [-0.52,-0.14, 2.3530],
                            [-0.63,-1.21, 2.7809],
                            [ 1.80,-1.50, 2.9948],
                            [1.1570, -1.3048, 2.5670],
                            [-2.58, 0.77, 5.3478])).to('cuda')
    
    ## Vanilla DR
    # states = torch.Tensor(([-2.44,-1.157, 7.4870],
    #                         [-2.20, 1.89, 11.7652],
    #                         [-1.33, 1.51, 5.5617],
    #                         [ 0.69, 1.10, 2.9948],
    #                         [-1.93, 0.06, 4.0643],
    #                         [-2.33,-0.52, 5.1339],
    #                         [0.068, 0.87, 2.3530],
    #                         [-0.52,-0.14, 1.9252],
    #                         [-0.63,-1.21, 3.2087],
    #                         [-2.58, 0.77, 6.2035])).to('cuda')

    for i in range(len(states)):
        traj_test(model, states[i], dynamics=dyn)



# Point 1 -2.44, -1.157, 89
# Point 2 -2.20, 1.89, 3
# Point 3 -1.33, 1.51, 0
# Point 4 0.69, 1.10, 13
# Point 5 -1.93, 0.06, 9
# Point 6 -2.33, -0.52, 35
# point 7 0.068,0.87, 74
# Point 8 -0.52,-0.138, 72
# Point 9 -0.63, -1.21, 28