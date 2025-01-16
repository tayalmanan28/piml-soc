from .dynamics import Dynamics
import torch
import math
from utils.modules import BCNetwork
import os
import numpy as np
from utils import modules

class Track7DAug(Dynamics):
    def __init__(self):
        self.w_max = 2
        self.a_max = 10
        self.L = 1
        self.z_mean = 14.5
        self.z_var = 14.6
        self.vx_g = 1
        self.vy_g = 1

        super().__init__(
            loss_type='brt_aug_hjivi', set_mode='reach',
            state_dim=10, input_dim=11, control_dim=2, disturbance_dim=0,
            state_mean=[0, 0, 5, 0, 0, 0, 0, 0, 0, self.z_mean],
            state_var=[10, 10, 5, math.pi, math.pi / 6, 9, 9, 2, 2, self.z_var],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model='reg',
            exact_factor=1.0
        )

    def control_range(self, state):
        return [[-self.a_max, self.a_max], [-self.w_max, self.w_max]]

    def state_test_range(self):
        return [
            [-10, 10],
            [-10, 10],
            [0, 10],
            [-math.pi, math.pi],
            [-math.pi/6, math.pi/6],
            [-9, 9],
            [-9, 9],
            [-2, 2],
            [-2, 2],
            [self.z_mean - self.z_var, self.z_mean + self.z_var],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 3] = (
            wrapped_state[..., 3] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # Dynamics of the Dubins5D Car:
    #         \dot{x}_0 = x_2 * cos(x_3)
    #         \dot{x}_1 = x_2 * sin(x_3)
    #         \dot{x}_2 = acc
    #         \dot{x}_3 = x_4
    #         \dot{x}_4 = alpha 

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] * torch.cos(state[..., 3]) 
        dsdt[..., 1] = state[..., 2] * torch.sin(state[..., 3]) 
        dsdt[..., 2] = control[..., 0] 
        dsdt[..., 3] = state[..., 2]/self.L * torch.tan(state[..., 4]) 
        dsdt[..., 4] = control[..., 1] 
        dsdt[..., 5] = state[..., 7]
        dsdt[..., 6] = state[..., 8]
        dsdt[..., 9] = -self.l_x(state)
        return dsdt
    
    def l_x(self, state):
        dist1 = state[..., 0:2]*1.0
        dist1[..., 0] = dist1[..., 0]- state[..., 5]
        dist1[..., 1] = dist1[..., 1]- state[..., 6]
        distance_1 = torch.norm(dist1[..., 0:2], dim=-1) 

        return (distance_1)

    def avoid_fn(self, state):
        # Defining a Rectangular Target
        # return torch.max(torch.abs(state[..., 0]) - 2, torch.abs(state[..., 1]) - 3)

        # Defining a Circular Target
        # obstacles (x,y,r): (-2,-1,0.6), (-0.1,0.2,0.5), (-1,1.5,0.9), (1,2.3,0.4), (1.5,0.1,0.9) * 3 #, (0.7,-1.6,0.6), (1,3,0.4)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(-6.0)
        dp1[..., 1] = dp1[..., 1]-(-3.0)
        dist1 = -torch.norm(dp1[..., 0:2], dim=-1) + 1.0

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-0.3)
        dp2[..., 1] = dp2[..., 1]-(0.6)
        dist2 = -torch.norm(dp2[..., 0:2], dim=-1) + 1.0

        # dp3 = state[..., 0:2]*1.0
        # dp3[..., 0] = dp3[..., 0]-(-3.0)
        # dp3[..., 1] = dp3[..., 1]-(4.5)
        # dist3 = -torch.norm(dp3[..., 0:2], dim=-1) + 1.0

        dp4 = state[..., 0:2]*1.0
        dp4[..., 0] = dp4[..., 0]-(3)
        dp4[..., 1] = dp4[..., 1]-(6.9)
        dist4 = -torch.norm(dp4[..., 0:2], dim=-1) + 1.0

        # dp5 = state[..., 0:2]*1.0
        # dp5[..., 0] = dp5[..., 0]-(4.5)
        # dp5[..., 1] = dp5[..., 1]-(0.3)
        # dist5 = -torch.norm(dp5[..., 0:2], dim=-1) + 1.0

        # dp6 = state[..., 0:2]*1.0
        # dp6[..., 0] = dp6[..., 0]-(0.7)
        # dp6[..., 1] = dp6[..., 1]-(-1.6)
        # dist6 = torch.norm(dp6[..., 0:2], dim=-1) - 0.6

        # dp7 = state[..., 0:2]*1.0
        # dp7[..., 0] = dp7[..., 0]-(1)
        # dp7[..., 1] = dp7[..., 1]-(3)
        # dist7 = torch.norm(dp7[..., 0:2], dim=-1) - 0.4

        lx = (torch.max(torch.max(dist1, dist2), dist4))#(torch.max(torch.max(torch.max(torch.max(dist1, dist2), dist3), dist4), dist5))
        return lx
    
    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,9])

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.max(self.avoid_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        
        return ((state[..., 2] * torch.cos(state[..., 3]) * dvds[..., 0]) \
                + (state[..., 2] * torch.sin(state[..., 3]) * dvds[..., 1]) \
                - (self.a_max) * torch.abs(dvds[..., 2])\
                + (state[..., 2] / self.L * torch.tan(state[..., 4]) * dvds[..., 3])\
                - (self.w_max) * torch.abs(dvds[..., 4]) \
                + (state[..., 7]* dvds[..., 5])\
                + (state[..., 8]* dvds[..., 6])\
                - (dvds[..., 9]*self.l_x(state)))

    def optimal_control(self, state, dvds):
        control1 = (-self.a_max*torch.sign(dvds[..., 2]))[..., None]
        control2 = (-self.w_max*torch.sign(dvds[..., 4]))[..., None]
        return torch.cat((control1, control2), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 4, 0, 0, 3, -5, 0, 0, 0],
            'state_labels': ['px', 'py', 'v', r'$\psi$', r'$\delta$', r'$gx$', r'$gy$', r'$vgx$', r'$vgy$',  r'$z$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 9,
        }