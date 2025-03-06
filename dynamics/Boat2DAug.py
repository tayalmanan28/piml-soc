from .dynamics import Dynamics
import torch
import math
from utils.modules import BCNetwork

class Boat2DAug(Dynamics):
    def __init__(self):
        self.goalR = 0.0
        self.avoid_fn_weight = -1
        self.v_max = 1
        super().__init__(
            loss_type='brt_aug_hjivi', set_mode="avoid",
            state_dim=3, input_dim=4, control_dim=2, disturbance_dim=0,
            state_mean=[-0.5, 0, 7.38],
            state_var=[2.5, 2.0, 7.48],#[3.125, 2.5, 7.48],#
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='reg',
            exact_factor=1.0,
        )
        # self.BCNN = BCNetwork()
        # self.BCNN.load_state_dict(
        #                 torch.load('plots/Boat2D/Boat_gx.pth', map_location='cpu'))
        # self.BCNN.eval()
        # self.BCNN.cuda()
        # for param in self.BCNN.parameters():
        #     param.requires_grad = False

    def state_test_range(self):
        return [
            [-3, 2],#[-3.625, 2.625],#
            [-2, 2],#[-2.5  , 2.5  ],#
            [-0.1, 14.86],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # wrapped_state[..., 2] = (
        #     wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    # Boat2D dynamics
    # \dot x    = u_1  + 2 - 0.5*y^2
    # \dot y    = u_2
    # \dot \theta = - norm(x_1-1.5, x_2)
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = control[..., 0] + 2 - 0.5*state[..., 1]*state[..., 1]
        dsdt[..., 1] = control[..., 1]
        dsdt[..., 2] = - self.l_x(state)
        return dsdt

    def reach_fn(self, state):
        return NotImplementedError

    def avoid_fn(self, state):
        # distance between the vehicle and obstacles
        # obstacles (x,y,r): (-2,-1,0.6), (-0.1,0.2,0.5), (-1,1.5,0.9), (1,2.3,0.4), (1.5,0.1,0.9), (0.7,-1.6,0.6), (1,3,0.4)

        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(-0.5)
        dp1[..., 1] = dp1[..., 1]-(0.5)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) -0.4 

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-1)
        dp2[..., 1] = (dp2[..., 1]-(-1.2))
        dist2 = torch.norm(dp2[..., 0:2], dim=-1) - 0.5

        dist = self.avoid_fn_weight * (torch.minimum(dist1, dist2))
        return dist

    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,2]) #self.avoid_fn(state)#

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def l_x(self, state):
        dist = state[..., 0:2]*1.0
        dist[..., 0] = dist[..., 0]-(1.5)
        dist[..., 1] = dist[..., 1]-(0)
        return torch.norm(dist[..., 0:2], dim=-1) -self.goalR

    def cost_fn(self, state_traj):
        return torch.max(self.avoid_fn(state_traj), dim=-1).values #torch.max(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return - torch.norm(dvds[..., 0:2], dim = -1) - dvds[..., 2]*self.l_x(state) + (2 - 0.5*state[..., 1]*state[..., 1])*dvds[..., 0]

    def optimal_control(self, state, dvds):
        opt_u1 = - dvds[..., 0]/torch.norm(dvds[..., 0:2], dim = -1)
        opt_u2 = - dvds[..., 1]/torch.norm(dvds[..., 0:2], dim = -1)
        return torch.cat((opt_u1[..., None], opt_u2[..., None]), dim=-1)
    
    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', 'z'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }