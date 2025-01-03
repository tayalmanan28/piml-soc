import torch
from dynamics.MVC9DAugParam import MVC9DAugParam
import os
from utils import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def opt_value_func_mesh(model, dynamics, traj_time, x1, y1, x2, y2, x3, y3, th1, th2, th3, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3, z_range, resolution, num_z, plot_flag= False):
    policy = model.eval()
    z_list = torch.linspace(z_range[0], z_range[1], num_z)
    z_opt = 1000
    
    dataset = []
    delta = -0.13
    
    for i in range(resolution):
        for j in range(resolution):
            z_min_list = []
            for z in z_list:
                traj_coord = torch.cat(
                    (traj_time.to('cuda'), torch.Tensor([x1, y1, x2, y2, x3, y3, th1, th2, th3, gx_1, gy_1, gx_2, gy_2, gx_3, gy_3]).to('cuda'), z.unsqueeze(0).to('cuda')), dim=-1
                ).reshape(1, 17)
                with torch.no_grad():
                    V_hat = policy(
                        {'coords': dynamics.coord_to_input(traj_coord.cuda())})
                    values = dynamics.io_to_value(V_hat['model_in'].detach(),
                                                V_hat['model_out'].squeeze(dim=-1).detach())
                if values <= delta:
                    z_min_list.append(z.item())
                # print(z.item(), values)
            
            # Find the minimum z-value for the current (x, y)
            if z_min_list:
                z_min_list.sort()
                z_opt = z_min_list[0]
            else:
                z_opt = 50  # Handle cases where no valid z-value exists
    

    # if plot_flag == True:
    #     # Save dataset to CSV
    #     df = pd.DataFrame(dataset, columns=['X', 'Y', 'Z'])
    #     df.to_csv("Boat2D/dataset.csv", index=False)
    #     print(f"Dataset saved")
        
    #     # Convert to numpy for plotting
    #     X_np = X.detach().cpu().numpy()
    #     Y_np = Y.detach().cpu().numpy()
    #     Z_np = Z.detach().cpu().numpy()

    #     # Plot the 2D heatmap
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(
    #         Z_np.T,  # Transpose to match the coordinate system of the plot
    #         extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
    #         origin='lower',
    #         cmap='RdYlBu',
    #         aspect='auto'
    #     )
    #     plt.colorbar(label='Z-Value (Optimal)')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('2D Heatmap of Z')
    #     plt.savefig("Boat2D/final_V_heatmap.png", dpi=1200)
    #     # plt.show()

    #     # Plot the 3D mesh-grid
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(projection='3d')
    #     ax.plot_surface(X_np, Y_np, Z_np, cmap='YlOrRd', edgecolor='k', alpha=0.8)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('3D Mesh-Grid Plot')
    #     plt.savefig("Boat2D/final_V_mesh_plot.png", dpi=1200)
    #     # plt.show()

    return z_opt

if __name__ == "__main__":
    times = torch.Tensor([2.0])
    dyn = MVC9DAugParam()
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
        final_layer_factor=1., hidden_features=512, num_hidden_layers=3
    )
    model_path = os.path.join('runs/MVC9DAug_BC_s3', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    states = np.array(([-0.3, 0, 0.3, -0.9, 0.3, 0.9, 0.0, 0.0, 0.0],
                       [1, 0, -1, -0.9, 1, 0.9, 0, 0, 0]))
    # states = np.array(([-2.4285, -1.1558], [0,0]))
    for i in range(len(states)): #range(0,1): #
        resolution = 1  # Resolution for the mesh grid
        x1, y1, x2, y2, x3, y3, th1, th2, th3 = states[i]
        z_range = torch.Tensor([0, 5.6])
        print(1, x2)

        Z = opt_value_func_mesh(model, dyn, times, x1, y1, x2, y2, x3, y3, th1, th2, th3, z_range, resolution, num_z=210)

        print("Optimal Z:", Z)