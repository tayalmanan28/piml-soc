'''
This Code is for saving States, Z, V_hat values in a csv file
'''

import torch
from dynamics.Boat2DAug import Boat2DAug
import os
from utils import modules
import pandas as pd

def opt_value_func_mesh(model, dynamics, traj_time, x_range, y_range, z_range, resolution):
    policy = model.eval()
    
    # Create mesh grid for x and y
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    z = torch.linspace(z_range[0], z_range[1], resolution)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    V = torch.zeros_like(X)  # To store z-values for the mesh grid
    dataset = []
    
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                traj_coord = torch.cat(
                    (traj_time.to('cuda'), torch.Tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]]).to('cuda')), dim=-1
                ).reshape(1, 4)
                with torch.no_grad():
                    V_hat = policy(
                        {'coords': dynamics.coord_to_input(traj_coord.cuda())})
                    values = dynamics.io_to_value(V_hat['model_in'].detach(),
                                                V_hat['model_out'].squeeze(dim=-1).detach())
                    V[i, j, k] = values
                    print(i, j, k)

                dataset.append([X[i, j, k].item(), Y[i, j, k].item(), Z[i, j, k].item(), V[i, j, k].item()])

    # Save dataset to CSV
    df = pd.DataFrame(dataset, columns=['X', 'Y', 'Z', 'V'])
    df.to_csv("Boat2D/V_hat_ebc.csv", index=False)
    print(f"Dataset saved")

if __name__ == "__main__":
    times = torch.Tensor([2.0])
    dyn = Boat2DAug()
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
        final_layer_factor=1., hidden_features=256, num_hidden_layers=3
    )
    model_path = os.path.join('runs/Boat2DAug_BC', 'training', 'checkpoints', 'model_final.pth')
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    resolution = 70  # Resolution for the mesh grid
    # x, y = states[i]
    x_range = [-3,2]#[x, x]#[-2.78, -2.78] -2.33, 
    y_range = [-2,2]#[y, y]#[1.0, 1.0]
    z_range = [0, 14.76]

    opt_value_func_mesh(model, dyn, times, x_range, y_range, z_range, resolution)