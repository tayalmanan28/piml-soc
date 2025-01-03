from .dynamics import Dynamics
import torch
import math
from utils import diff_operators, quaternion
from utils.modules import BCNetwork

class Quadrotor10DAug(Dynamics):
    def __init__(self):  # simpler quadrotor
        self.d0 = 7
        self.d1 = 4
        self.n0 = 12
        self.g = -9.81
        self.z_mean = 4.2 ##Include terminal cost
        self.z_var = 4.3
        self.goalR = 0.0
        self.collisionR = 0.5
        self.u1_max = math.pi/4
        self.u2_max = math.pi/4
        self.u3_max = 1.0

        super().__init__(
            loss_type='brt_aug_hjivi', set_mode='reach',
            state_dim=11, input_dim=12, control_dim=3, disturbance_dim=0,
            state_mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.z_mean],
            state_var=[4.0, 4.0, 2.0, 1.5, 1.5, 3, 3, 2, 6, 6, self.z_var],
            value_mean= 0,
            value_var=1,
            value_normto=0.02,
            deepReach_model='reg',
            exact_factor=1,
        )

    def state_test_range(self):
        return [
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-1.0, 1.0],
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-2.0, 2.0],
            [-6.0, 6.0],
            [-6.0, 6.0],
            [self.z_mean - self.z_var, self.z_mean + self.z_var]
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        phi = state[..., 3] * 1.0
        theta = state[..., 4] * 1.0
        vx = state[..., 5] * 1.0
        vy = state[..., 6] * 1.0
        vz = state[..., 7] * 1.0
        wx = state[..., 8] * 1.0
        wy = state[..., 9] * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -self.d1*phi + wx
        dsdt[..., 4] = -self.d1*theta + wy
        dsdt[..., 5] = self.g*torch.tan(theta)
        dsdt[..., 6] = self.g*torch.tan(phi)
        dsdt[..., 7] = control[..., 2]
        dsdt[..., 8] = -self.d0*phi + self.n0*control[..., 0]
        dsdt[..., 9] = -self.d0*theta + self.n0*control[..., 1]
        dsdt[..., 10] = -self.l_x(state)

        return dsdt
    
    def l_x(self, state): # to be changed
        dist1 = state[..., 0:2]*1.0
        dist1[..., 0] = dist1[..., 0]- 1.5
        dist1[..., 1] = dist1[..., 1]- 0
        distance_1 = torch.norm(dist1[..., 0:2], dim=-1) - self.goalR

        return distance_1 #torch.maximum(distance_1, torch.maximum(distance_2, distance_3)) #

    def avoid_fn(self, state):
        '''for cylinder with full body collision'''

        px = state[..., 0]
        py = state[..., 1]

        # get full body distance
        dist = torch.norm(
            torch.cat((px[..., None], py[..., None]), dim=-1), dim=-1)
        return -dist + self.collisionR

    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,10])

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.max(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):

        phi = state[..., 3] * 1.0
        theta = state[..., 4] * 1.0
        vx = state[..., 5] * 1.0
        vy = state[..., 6] * 1.0
        vz = state[..., 7] * 1.0
        wx = state[..., 8] * 1.0
        wy = state[..., 9] * 1.0

        # Compute the hamiltonian for the quadrotor
        ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
        ham += dvds[..., 3] * (-self.d1*phi + wx)
        ham += dvds[..., 4] * (-self.d1*theta + wy)
        ham += dvds[..., 5] * (self.g*torch.tan(theta))
        ham += dvds[..., 6] * (self.g*torch.tan(phi))
        ham += dvds[..., 7] * self.u3_max
        ham += -dvds[..., 8] * self.d0*phi + torch.abs(dvds[..., 8]) * self.n0*self.u1_max
        ham += -dvds[..., 9] * self.d0*theta + torch.abs(dvds[..., 9])*self.n0*self.u2_max

        ham += - dvds[..., 10]*self.l_x(state)

        return ham

    def optimal_control(self, state, dvds):
        u1 = -self.u1_max * torch.sign(dvds[..., 8])
        u2 = -self.u2_max * torch.sign(dvds[..., 9])
        u3 = -self.u3_max * torch.sign(dvds[..., 7])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0.0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'phi', 'theta', 'vx', 'vy', 'vz', 'wx', 'wy', 'z_cost'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 10,
        }