# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import utils
from utils import modules
import train_BC_estimator as BC_Model

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()

p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.5, required=False, help='End time of simulation')
p.add_argument('--num_nl', type=int, default=128, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--val_x_resolution', type=int, default=200,
                   help='x-axis resolution of validation plot during training')
p.add_argument('--val_y_resolution', type=int, default=200,
                   help='y-axis resolution of validation plot during training')
p.add_argument('--val_z_resolution', type=int, default=5,
                   help='z-axis resolution of validation plot during training')
p.add_argument('--val_time_resolution', type=int, default=3,
                   help='time-axis resolution of validation plot during training')
opt = p.parse_args()

state_mean=[0.0, 0.0, 0.0, 0.0],  # q1, q2, omega1, omega2
state_var=[0.52, 1.04, 4.0, 8.0],  # q1, q2, omega1, omega2
value_mean=0.0,
value_var=3.0,
value_normto=0.02,
deepReach_model='exact'

def state_test_range(self):
        return [
            [-0.52, 0.52],                    # q1
            [-1.04, 1.04],                    # q2
            [-4.0, 4.0],                      # q3
            [-8.0, 8.0],                      # q4
        ]

def boundary_fn(self, state):
        device=state.device
        lx=BC_Model.BCNN(state.cuda()).to(device)
        return lx

model = BC_Model.BCNetwork()

model.load_state_dict(
                torch.load('BC_estimator.pth', map_location='cpu'))
model.eval()
ckpt_dir= './Compass_Walker_Plots'

xs = torch.linspace(-0.52, 0.52, 100)
ys = torch.linspace(-1.04, 1.04, 100)
dq1s = torch.linspace(-4.0, 4.0, 10)
dq2s = torch.linspace(-8.0, 8.0, 10)
coords = torch.cartesian_prod(xs, ys, dq1s, dq2s)

coords= coords
lx2=model(coords).detach().cpu()
print(lx2)
brt_counts= torch.sum(torch.sum(lx2.reshape(100,100,10,10),axis=-1),axis=-1).T.float()
print(brt_counts)
BRT_img = brt_counts.numpy()

lx=BC_Model.boundary_fn(coords)
brt_counts_orig=torch.sum(torch.sum(lx.reshape(100,100,10,10),axis=-1),axis=-1).T.float()
BRT_img_orig = brt_counts_orig.numpy()

max_value = np.amax(BRT_img)
min_value = np.amin(BRT_img)
max_value_orig = np.amax(BRT_img_orig)
min_value_orig = np.amin(BRT_img_orig)
print(max_value)
print(min_value)
print(max_value_orig)
print(min_value_orig)

imshow_kwargs = {
    'vmax': max_value,
    'vmin': min_value,
    'cmap': 'RdYlBu',
    'extent': (-0.52, 0.52, -1.04, 1.04),
    'origin': 'lower',
}
fig= plt.figure()
ax = fig.add_subplot(1, 1, 1)
s1 = ax.imshow(BRT_img, **imshow_kwargs)
fig.colorbar(s1)
fig2= plt.figure(2)
ax2 = fig2.add_subplot(1, 1, 1)
s2 = ax2.imshow(BRT_img_orig, **imshow_kwargs)
fig2.colorbar(s2)
fig.savefig(os.path.join(ckpt_dir, 'Learned_Boundary_Function.png'))
fig2.savefig(os.path.join(ckpt_dir, 'Actual_Boundary_Function.png'))