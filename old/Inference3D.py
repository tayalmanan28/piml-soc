# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import utils
from utils import modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio

p = configargparse.ArgumentParser()

p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.5, required=False, help='End time of simulation')
p.add_argument('--num_nl', type=int, default=128, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--val_x_resolution', type=int, default=201,
                   help='x-axis resolution of validation plot during training')
p.add_argument('--val_y_resolution', type=int, default=201,
                   help='y-axis resolution of validation plot during training')
p.add_argument('--val_z_resolution', type=int, default=5,
                   help='z-axis resolution of validation plot during training')
p.add_argument('--val_time_resolution', type=int, default=1,
                   help='time-axis resolution of validation plot during training')
opt = p.parse_args()

state_mean=torch.Tensor([0.2, 0.35])  # theta, omega
state_var= torch.Tensor([0.4, 0.95])   # theta, omega
value_mean=0
value_var=1.2
value_normto=0.02
deepReach_model='reg'

model = modules.SingleBVPNet(in_features=3, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model_hard = modules.SingleBVPNet(in_features=3, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model.cuda()
model_hard.cuda()

# model_hard = modules.SingleBVPNet(in_features=4, out_features=1, type=opt.model, mode=opt.mode, final_layer_factor=1.,
#                              hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
# model_hard.cuda()

model_dir = '.'
model_path = os.path.join(model_dir, 'Inference_Models', 'model_rimless_vanilla_s3.pth')
model_path_hard = os.path.join(model_dir, 'Inference_Models', 'model_rimless_WP_s2.pth')
#model_path = '/Inference_Models/model_rimless_vanilla.pth'
checkpoint = torch.load(model_path)
checkpoint_hard = torch.load(model_path_hard)
#model.load_state(checkpoint)
model.load_state_dict(checkpoint['model'])
model_hard.load_state_dict(checkpoint_hard['model'])
model.eval()
ckpt_dir= './Inference_Checkpoints'
E=1.132
eps=0.03

true_data = spio.loadmat('data_rimless_wheel_t_30.mat')
#print(true_data['data'][:, :, 30])
Z = np.array(true_data['data'][:, :, 30].T)
Z = Z<=0
#print(Z)

def state_test_range():
        return [
            [-0.2, 0.6],                        # theta
            [-0.6, 1.3],                    # omega
        ]

def plot_config():
        return {
            'state_slices': [0, 0],
            'state_labels': ['theta', 'omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': -1,
        }

def boundary_fn(state): # everything pre-switch state is converted to post-switch first
        lx=torch.abs(torch.sqrt(2*(E-torch.cos(state[..., 0])))-state[..., 1])-eps
        return lx

def input_to_coord(input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * state_var.to('cuda')
                          ) + state_mean.to('cuda')
        return coord

def coord_to_input(coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - state_mean.to('cuda')
                          ) / state_var.to('cuda')
        return input

# convert model io to real value
def io_to_value(input, output):
    if deepReach_model == 'diff':
            return (output * value_var / value_normto) + boundary_fn(input_to_coord(input)[..., 1:])
    elif deepReach_model == 'exact':
            k = 1.0
            exact_BC_factor = k * input[..., 0] + (1-k)
            return (output * exact_BC_factor * value_var / value_normto) + boundary_fn(input_to_coord(input)[..., 1:])
    elif deepReach_model == 'reg':
            return (output * value_var / value_normto) + value_mean
    else:
            raise NotImplementedError


x_resolution = opt.val_x_resolution
y_resolution = opt.val_y_resolution
plot_configuration = plot_config()
state_slices = plot_configuration['state_slices']
x_axis_idx = plot_configuration['x_axis_idx']
y_axis_idx = plot_configuration['y_axis_idx']
state_range = state_test_range()
x_min, x_max = state_range[x_axis_idx]
y_min, y_max = state_range[y_axis_idx]
time_resolution = opt.val_time_resolution
tMax = opt.tMax
times = torch.linspace(opt.tMax, opt.tMax, time_resolution)
xs = torch.linspace(x_min, x_max, x_resolution)
ys = torch.linspace(y_min, y_max, y_resolution)
xys = torch.cartesian_prod(xs, ys)
fig = plt.figure(figsize=(6, 5*len(times)))
fig2 = plt.figure(figsize=(6, 5*len(times)))
means_vanilla=np.zeros((len(times)))
std_devs_vanilla=np.zeros(len(times))
means_ebc=np.zeros(len(times))
std_devs_ebc=np.zeros(len(times))

for i in range(len(times)):
    coords = torch.zeros(
        x_resolution*y_resolution, 3)
    coords[:, 0] = times[i]
    coords[:, 1:] = torch.tensor(plot_configuration['state_slices'])
    coords[:, 1 + plot_configuration['x_axis_idx']] = xys[:, 0]
    coords[:, 1 + plot_configuration['y_axis_idx']] = xys[:, 1]

    with torch.no_grad():
        model_results = model(
            {'coords': coord_to_input(coords.cuda())})
        
    values = io_to_value(model_results['model_in'].detach(
            ), model_results['model_out'].squeeze(dim=-1).detach())
    
    deepReach_model='exact'
    with torch.no_grad():
        model_results_hard = model_hard(
            {'coords': coord_to_input(coords.cuda())})
        
    values_hard = io_to_value(model_results_hard['model_in'].detach(
            ), model_results_hard['model_out'].squeeze(dim=-1).detach())

    ax = fig.add_subplot(len(times), 1, 1 + i)
    ax.set_title('t = %0.2f' % (times[i]))
    BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    #print("Vanilla",np.min(BRT_img), np.max(BRT_img))
    means_vanilla[i] = np.mean(values.detach().cpu().numpy())
    std_devs_vanilla[i] = np.std(values.detach().cpu().numpy())
    BRT_img_hard = values_hard.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    #print("Exact_BC",np.min(BRT_img_hard), np.max(BRT_img_hard))
    means_ebc[i] = np.mean(values_hard.detach().cpu().numpy())
    std_devs_ebc[i] = np.std(values_hard.detach().cpu().numpy())
    # BRT_img = np.abs(BRT_img_hard - BRT_img_vanilla)
    max_value = np.amax(BRT_img)
    min_value = np.amin(BRT_img)
    van_bool = np.array((BRT_img<=0.0), dtype=bool)
    exact_bool = np.array((BRT_img_hard<=0.0), dtype=bool)
    #print(van_bool)
    #print(exact_bool)
    van_bool = np.logical_and(van_bool, Z)
    exact_bool = np.logical_and(exact_bool, Z)
    #print(BRT_img_hard<=0)
    print(np.sum(Z+~Z))
    print(np.sum(Z))
    print(np.sum(exact_bool))
    print(np.sum(van_bool))
    print(np.sum(exact_bool)/np.sum(Z))
    print(np.sum(van_bool)/np.sum(Z))
    # We'll also create a grey background into which the pixels will fade
    greys = np.full((*BRT_img.shape, 3), 70, dtype=np.uint8)
    imshow_kwargs = {
        'vmax': max_value,
        'vmin': min_value,
        'cmap': 'RdYlBu',
        'extent': (x_min, x_max, y_min, y_max),
        'origin': 'lower',
        'aspect': 'auto',
    }
    ax.imshow(greys)
    ax.imshow((BRT_img<=0.0), cmap='bwr',
            origin='lower', extent=(x_min, x_max, y_min, y_max), aspect='auto')

    ax2 = fig2.add_subplot(len(times), 1, 1 + i)
    ax2.set_title('t = %0.2f' % (times[i]))
    ax2.imshow(1*(BRT_img_hard<=0.0), cmap='bwr',
            origin='lower', extent=(x_min, x_max, y_min, y_max), aspect='auto')

fig.savefig(os.path.join(ckpt_dir, 'Vanilla_BRT.png'))
fig2.savefig(os.path.join(ckpt_dir, 'Exact_BRT.png'))
plt.figure(figsize=(8, 6))
plt.errorbar(times, means_vanilla, yerr=std_devs_vanilla, fmt='o-', ecolor='r', capsize=5, capthick=2)
plt.xlabel('Timestamps')
plt.ylabel('Value Mean and Standard Deviation')
plt.title('Range of Values for Each Timestamp')
plt.grid(True)
plt.savefig('vanilla_range.png', dpi=300, bbox_inches='tight')
plt.figure(figsize=(8, 6))
plt.errorbar(times, means_ebc, yerr=std_devs_ebc, fmt='o-', ecolor='r', capsize=5, capthick=2)
plt.xlabel('Timestamps')
plt.ylabel('Value Mean and Standard Deviation')
plt.title('Range of Values for Each Timestamp')
plt.grid(True)
plt.savefig('ebc_range.png', dpi=300, bbox_inches='tight')
