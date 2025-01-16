import torch
from tqdm.autonotebook import tqdm
from dynamics.Track7DAug import Track7DAug
import os
import numpy as np
from utils import modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle, Ellipse
from final_value import opt_value_func_mesh as opt_value
from anim5V import animate_trajectories

def traj_test(model, initial_state, dynamics, dt = 0.0025, tMax = 1.0, tMin = 0):
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
    cost = 0.0
    for k in tqdm(range(int((tMax-tMin)/dt)), desc='Trajectory Propagation', position=pbar_pos, leave=False):
        traj_time = tMax - k*dt
        traj_time_list.append(1.0-traj_time)
        traj_times = torch.full((1, ), traj_time)
        cost = cost + dynamics.l_x(state_trajs[:, k])*dt
        
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
        if (traj_time <= dt):
            cost = cost + dynamics.l_x(state_trajs[:, k+1])
            print("Terminal_Cost", dynamics.l_x(state_trajs[:, k+1]))
        pbar_pos +=1
        # print()

    traj = state_trajs[0].T
    x, y, v, psi, delta, gx, gy, vgx, vgy, z = traj
    cost_fn = dynamics.cost_fn(state_trajs)
    print("Avoid cost",cost_fn)

    x = x.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()
    v = v.to('cpu').detach().numpy()
    psi = psi.to('cpu').detach().numpy()
    delta = delta.to('cpu').detach().numpy()
    gx = gx.to('cpu').detach().numpy()
    gy = gy.to('cpu').detach().numpy()
    vgx = vgx.to('cpu').detach().numpy()
    vgy = vgy.to('cpu').detach().numpy()

    # animate_trajectories(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, gx_4, gy_4, gx_5, gy_5)


    z = z.to('cpu').detach().numpy()
    # print(traj, ctrl_trajs[0].T, x2, y2)
    #print(ctrl_trajs[0].T)
    # print(z)
    print("Actual Cost",cost)

    #traj_t = np.array(traj_time_list)
    # act_z = np.array(Act_Z)

    # plt.figure(1)

    #plt.scatter(traj_t, z, s=0.1)
    # plt.scatter(traj_t, act_z, s=0.1)
    # plt.savefig("MVC9D/z_time_plot.png",dpi=1200) 

    plt.figure(2)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title("VDR Trajectories")
    # currentAxis1 = plt.gca()
    # currentAxis1.add_patch(Rectangle((-0.9, 0.1), 0.8, 0.8, facecolor = 'orange', alpha=1))
    # currentAxis2 = plt.gca()
    # currentAxis2.add_patch(Rectangle((-1.2, -2.3), 0.4, 2, facecolor = 'orange', alpha=1))
    # currentAxis1 = plt.gca()
    # currentAxis1.add_patch(Circle((initial_state[0], initial_state[1]), 0.01, facecolor = 'blue', alpha=1))
    # currentAxis2 = plt.gca()
    # currentAxis2.add_patch(Circle((initial_state[2], initial_state[3]), 0.01, facecolor = 'orange', alpha=1))
    # currentAxis3 = plt.gca()
    # currentAxis3.add_patch(Circle((initial_state[4], initial_state[5]), 0.01, facecolor = 'green', alpha=1))
    # currentAxis4 = plt.gca()
    # currentAxis4.add_patch(Circle((initial_state[6], initial_state[7]), 0.01, facecolor = 'red', alpha=1))
    # currentAxis5 = plt.gca()
    # currentAxis5.add_patch(Circle((initial_state[8], initial_state[9]), 0.01, facecolor = 'cyan', alpha=1))

    currentAxis1 = plt.gca()
    currentAxis1.add_patch(Circle((-6, -3), 1.0, facecolor = 'blue', alpha=1))
    currentAxis2 = plt.gca()
    currentAxis2.add_patch(Circle((-0.3, 0.6), 1.0, facecolor = 'orange', alpha=1))
    # currentAxis3 = plt.gca()
    # currentAxis3.add_patch(Circle((-3, 4.5), 1.0, facecolor = 'green', alpha=1))
    currentAxis4 = plt.gca()
    currentAxis4.add_patch(Circle((3, 6.9), 1.0, facecolor = 'red', alpha=1))
    # currentAxis5 = plt.gca()
    # currentAxis5.add_patch(Circle((4.5, 0.3), 1.0, facecolor = 'cyan', alpha=1))
    plt.scatter(x, y, s=0.1)
    plt.scatter(gx, gy, s=0.1)
    plt.savefig("plots/Track/traj_plot.png",dpi=1200)    
    # plt.show()    

if __name__ =="__main__":

    dyn = Track7DAug()
    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=256, num_hidden_layers=3)
    model_path = os.path.join('runs/Track7DAug_VDR', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    # initial_state = torch.Tensor([-0.63, -1.21, 2.7809]).to('cuda')
    # traj_test(model, initial_state, dynamics=dyn)
    times = torch.Tensor([1.0])
                        #    x1    z
    states = torch.Tensor(([-2.5, 0.0, 2, 0.5, 0, 0, 2, -2.0, 1.0, 0],
                           [2.5, -7.5, 1.5, 0.0, 0, 5, -7.5, 0.0,-2.0, 0],
                           [2.5, 5.0, 1.5, 0.0, 0, 5, 5, 0.0, 1.5, 0],
                           [0.0,-5.0, 5,0.0, 0, 5, -5, 1.5, 0.0, 0],
                            )).to('cuda')
    
    for i in range(len(states)): #range(0,1): #
        act_state = states[i][0:9]
        # print(act_state)
        
        z_range = torch.Tensor([0, 28.28])

        Z = opt_value(model, dyn, times, act_state, z_range, resolution=1, num_z=210, delta=-0.610)

        states[i][9] = Z
        print("Optimal Z", Z)

    for i in range(len(states)):
        traj_test(model, states[i], dynamics=dyn)

# Point 1 [0.0, 0.0, 0.5, 0.3, 0.5, -0.3, math.pi, -math.pi/3, 2*math.pi/3, -0.3, 0.0, 0.51,-0.4,0.49, 0.4, 0.0]
# Point 2 [0.0, 0.0, 0.0,-0.4, 0.0, 0.4, math.pi, 0.0, 0.0, -0.3, 0.0, 0.51,-0.4,0.49, 0.4, 0.0]
# Point 3 [0.0, 0.0, 0.0, 0.4, 0.0, -0.4, math.pi, 0.0, 0.0, -0.3, 0.0, 0.51,-0.4,0.49, 0.4, 0.0]

# Point 1 [0.0, 0.0, 0.5, 0.3, 0.5,-0.3, math.pi, -math.pi/3, 2*math.pi/3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Point 2 [0.0, 0.0, 0.0,-0.4, 0.0, 0.4, math.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Point 3 [0.0, 0.0, 0.0, 0.4, 0.0,-0.4, math.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Point 1 [0.0, 0.0, 0.5, 0.3, 0.5, -0.3, math.pi, -math.pi/3, 2*math.pi/3, 0.3, 0.0, -0.51,-0.4,-0.49, 0.4, 0.0]
# Point 2 [0.0, 0.0, 0.0,-0.4, 0.0, 0.4, math.pi, 0.0, 0.0, -0.3, 0.0, 0.51,-0.4,0.49, 0.4, 0.0]
# Point 3 [0.0, 0.0, 0.0, 0.4, 0.0, -0.4, math.pi, 0.0, 0.0, -0.3, 0.0, 0.51,-0.4,0.49, 0.4, 0.0]
