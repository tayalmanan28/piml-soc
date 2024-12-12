import torch
from dynamics.Boat2DAug import Boat2DAug
import os
from utils import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def opt_value_func_mesh(model, dynamics, traj_time, x_range, y_range, z_range, resolution, num_z):
    policy = model.eval()
    z_list = torch.linspace(z_range[0], z_range[1], num_z)
    
    # Create mesh grid for x and y
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    z_opt = 1000
    
    Z = torch.zeros_like(X)  # To store z-values for the mesh grid
    dataset = []
    delta = -0.14#-0.038
    
    for i in range(resolution):
        for j in range(resolution):
            z_min_list = []
            for z in z_list:
                traj_coord = torch.cat(
                    (traj_time.to('cuda'), torch.Tensor([X[i, j], Y[i, j]]).to('cuda'), z.unsqueeze(0).to('cuda')), dim=-1
                ).reshape(1, 4)
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
                Z[i, j] = z_min_list[0]
            else:
                Z[i, j] = 20  # Handle cases where no valid z-value exists

            if Z[i, j] <= z_opt:
                z_opt = Z[i,j]

            print(i,j)
            dataset.append([X[i, j].item(), Y[i, j].item(), Z[i, j].item()])

    # Save dataset to CSV
    df = pd.DataFrame(dataset, columns=['X', 'Y', 'Z'])
    df.to_csv("Boat2D/dataset.csv", index=False)
    print(f"Dataset saved")

    print("Optimal Z:", z_opt)
    
    # Convert to numpy for plotting
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    Z_np = Z.detach().cpu().numpy()

    # Plot the 2D heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        Z_np.T,  # Transpose to match the coordinate system of the plot
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin='lower',
        cmap='RdYlBu',
        aspect='auto'
    )
    plt.colorbar(label='Z-Value (Optimal)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Heatmap of Z')
    plt.savefig("Boat2D/final_V_heatmap.png", dpi=1200)
    # plt.show()

    # Plot the 3D mesh-grid
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X_np, Y_np, Z_np, cmap='YlOrRd', edgecolor='k', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh-Grid Plot')
    plt.savefig("Boat2D/final_V_mesh_plot.png", dpi=1200)
    # plt.show()

if __name__ == "__main__":
    times = torch.Tensor([2.0])
    dyn = Boat2DAug()
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
        final_layer_factor=1., hidden_features=256, num_hidden_layers=3
    )
    model_path = os.path.join('runs/Boat2DAug_ext_BC', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    states = np.array(([-2.44, -1.157],
    [-2.20, 1.89],
    [-1.33, 1.51],
    [ 0.69, 1.10],
    [-1.93, 0.06],
    [-2.33,-0.52],
    [0.068, 0.87],
    [-0.52,-0.14],
    [-0.63,-1.21],
    [ 1.80,-1.50],
    [1.1570, -1.3048],
    [-2.58, 0.77]))
    # states = np.array(([-2.4285, -1.1558], [0,0]))
    for i in range(0,1): #range(len(states)): #
        resolution = 200  # Resolution for the mesh grid
        x, y = states[i]
        x_range = [-3,2]#[x, x]# 
        y_range = [-2,2]#[y, y]#
        z_range = torch.Tensor([0, 14.76])

        opt_value_func_mesh(model, dyn, times, x_range, y_range, z_range, resolution, num_z=200)