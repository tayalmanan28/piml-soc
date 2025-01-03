import torch
from tqdm.autonotebook import tqdm
from dynamics.MVC6DAug import MVC6DAug
import os
import numpy as np
from utils import modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle, Ellipse
from final_value_MVC6D import opt_value_func_mesh as opt_value
import math

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
    traj_time_list = [0.0]
    z_range = torch.Tensor([0, 2.9])
    Z = initial_state[6]
    Z = Z.to('cpu').detach().numpy()
    Act_Z = [Z]
    for k in tqdm(range(int((tMax-tMin)/dt)), desc='Trajectory Propagation', position=pbar_pos, leave=False):
        traj_time = tMax - k*dt
        traj_time_list.append(2.0-traj_time)
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

        # print(next_state_)
        
        

        # if k%100 == 0:
        #     x_, y_, z_ = next_state_[0]
        #     x_range_ = [x_, x_]#[-3,2]# 
        #     y_range_ = [y_, y_]#[-2,2]#

        #     Z = opt_value(model, dyn, times, x_range_, y_range_, z_range, resolution=1, num_z=210)
        #     Z = Z.to('cpu').detach().numpy()
            
        #     # next_state_[0][2] = torch.Tensor(Z).to('cuda')
        
        # Act_Z.append(Z)


        state_trajs[:, k+1] = next_state_
        pbar_pos +=1
        # print()

    traj = state_trajs[0].T
    x1, y1, x2, y2, th1, th2, z = traj

    x1 = x1.to('cpu').detach().numpy()
    y1 = y1.to('cpu').detach().numpy()
    x2 = x2.to('cpu').detach().numpy()
    y2 = y2.to('cpu').detach().numpy()
    th1 = th1.to('cpu').detach().numpy()
    th2 = th2.to('cpu').detach().numpy()


    z = z.to('cpu').detach().numpy()
    # print(traj, ctrl_trajs[0].T, x2, y2)

    # traj_t = np.array(traj_time_list)
    # act_z = np.array(Act_Z)

    # plt.figure(1)

    # plt.scatter(traj_t, z, s=0.1)
    # plt.scatter(traj_t, act_z, s=0.1)
    # plt.savefig("MVC9D/z_time_plot.png",dpi=1200) 

    plt.figure(2)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Exact BC Trajectories")
    # currentAxis1 = plt.gca()
    # currentAxis1.add_patch(Rectangle((-0.9, 0.1), 0.8, 0.8, facecolor = 'orange', alpha=1))
    # currentAxis2 = plt.gca()
    # currentAxis2.add_patch(Rectangle((-1.2, -2.3), 0.4, 2, facecolor = 'orange', alpha=1))
    currentAxis1 = plt.gca()
    currentAxis1.add_patch(Circle((-0.50, -0.5), 0.025, facecolor = 'cyan', alpha=1))
    currentAxis2 = plt.gca()
    currentAxis2.add_patch(Circle((0.5, 0.5), 0.025, facecolor = 'cyan', alpha=1))
    # currentAxis3 = plt.gca()
    # currentAxis3.add_patch(Circle((1.0, -1.0), 0.025, facecolor = 'cyan', alpha=1))
    plt.scatter(x1, y1, s=0.1)
    plt.scatter(x2, y2, s=0.1)
    plt.savefig("MVC9D/traj_plot.png",dpi=1200)    
    # plt.show()    

if __name__ =="__main__":

    dyn = MVC6DAug()
    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=256, num_hidden_layers=3)
    model_path = os.path.join('runs/MVC6DAug_BC', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    # initial_state = torch.Tensor([-0.63, -1.21, 2.7809]).to('cuda')
    # traj_test(model, initial_state, dynamics=dyn)
    times = torch.Tensor([2.0])

    states = torch.Tensor(( [0.15, 0.15, -0.15, -0.15, math.pi, 0, 0.0],
                        #    [-0.3, 0, 0.3,-0.9, 0.3, 0.9, 0, 0, 0, 0.0],
                            )).to('cuda')
    
    for i in range(len(states)): #range(0,1): #
        x1, y1, x2, y2, th1, th2, z = states[i]
        z_range = torch.Tensor([0, 8.5])

        Z = opt_value(model, dyn, times, x1, y1, x2, y2, th1, th2, z_range, resolution=1, num_z=210)
        states[i][6] = Z 
        print("Optimal Z",Z)

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