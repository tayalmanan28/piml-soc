from .dynamics import Dynamics
import torch
import math

class MANAug4PM(Dynamics):
    def __init__(self):
        self.set_mode = 'reach'
        self.velocity = 0.6
        self.collisionR = 0.25
        self.goalR = 0.0
        self.z_mean = 2.1
        self.z_var = 2.2
        super().__init__(
            loss_type='brt_aug_hjivi', set_mode=self.set_mode, state_dim=17, input_dim=18, control_dim=8, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,self.z_mean,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                1, 1,
                0.9, 0.9,
                0.9, 0.9,
                0.9, 0.9,
                0.9, 0.9,  self.z_var,
            ],
            value_mean=0.0,
            value_var=1.0,
            value_normto=1.0,
            deepReach_model='reg',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-0.9, 0.9], [-0.9, 0.9],
            [-0.9, 0.9], [-0.9, 0.9],
            [-0.9, 0.9], [-0.9, 0.9],
            [-0.9, 0.9], [-0.9, 0.9],
            [self.z_mean - self.z_var, self.z_mean + self.z_var],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # dynamics (per car)
    # \dot x    = v_x
    # \dot y    = v_y
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = control[..., 0]
        dsdt[..., 1] = control[..., 1]
        dsdt[..., 2] = control[..., 2]
        dsdt[..., 3] = control[..., 3]
        dsdt[..., 4] = control[..., 4]
        dsdt[..., 5] = control[..., 5]
        dsdt[..., 6] = control[..., 6]
        dsdt[..., 7] = control[..., 7]
        dsdt[..., 16] = -self.l_x(state)
        return dsdt
    
    def l_x(self, state):
        dist1 = state[..., 0:2]*1.0
        dist1[..., 0] = dist1[..., 0]- state[..., 8]
        dist1[..., 1] = dist1[..., 1]- state[..., 9]
        distance_1 = torch.norm(dist1[..., 0:2], dim=-1) - self.goalR

        dist2 = state[..., 2:4]*1.0
        dist2[..., 0] = dist2[..., 0]- state[..., 10]
        dist2[..., 1] = dist2[..., 1]- state[..., 11]
        distance_2 = torch.norm(dist2[..., 0:2], dim=-1) - self.goalR

        dist3 = state[..., 4:6]*1.0
        dist3[..., 0] = dist3[..., 0]- state[..., 12]
        dist3[..., 1] = dist3[..., 1]- state[..., 13]
        distance_3 = torch.norm(dist3[..., 0:2], dim=-1) - self.goalR

        dist4 = state[..., 6:8]*1.0
        dist4[..., 0] = dist4[..., 0]- state[..., 14]
        dist4[..., 1] = dist4[..., 1]- state[..., 15]
        distance_4 = torch.norm(dist4[..., 0:2], dim=-1) - self.goalR

        return (distance_1 + distance_2 + distance_3 + distance_4)/4 #torch.maximum(distance_1, torch.maximum(distance_2, distance_3)) #

    def avoid_fn(self, state):
        boundary_values_12 = -torch.norm(
            state[..., 0:2] - state[..., 2:4], dim=-1) + self.collisionR
        boundary_values_13 = -torch.norm(
            state[..., 0:2] - state[..., 4:6], dim=-1) + self.collisionR
        boundary_values_14 = -torch.norm(
            state[..., 0:2] - state[..., 6:8], dim=-1) + self.collisionR
        
        boundary_values_23 = -torch.norm(
            state[..., 2:4] - state[..., 4:6], dim=-1) + self.collisionR
        boundary_values_24 = -torch.norm(
            state[..., 2:4] - state[..., 6:8], dim=-1) + self.collisionR
        
        boundary_values_34 = -torch.norm(
            state[..., 4:6] - state[..., 6:8], dim=-1) + self.collisionR
        
        boundary_values_1 = torch.max(boundary_values_14, torch.max(
                    boundary_values_12, boundary_values_13))
        
        boundary_values_2 =  torch.max(
                    boundary_values_24, boundary_values_23)

        boundary_values = torch.max(boundary_values_1,torch.max(boundary_values_2, boundary_values_34))
        
        return boundary_values
    
    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,16])

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.max(self.avoid_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):

        # Compute the hamiltonian for the ego vehicle
        ham =-self.velocity*(torch.abs(dvds[..., 0]) + \
                             torch.abs(dvds[..., 1]) + \
                             torch.abs(dvds[..., 2]) + \
                             torch.abs(dvds[..., 3]) + \
                             torch.abs(dvds[..., 4]) + \
                             torch.abs(dvds[..., 5]) + \
                             torch.abs(dvds[..., 6]) + \
                             torch.abs(dvds[..., 7])) 

        ham = ham - dvds[..., 16]*self.l_x(state)
        return ham

    def optimal_control(self, state, dvds):
        return -1.0*self.velocity*torch.sign(dvds[..., [0, 1, 2, 3, 4, 5, 6, 7]])

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                 0.0, 0.0,
                 0.5,-0.4,
                 0.5, 0.4,
                -0.5,-0.4,
                -0.5, 0.4,
                -0.5, 0.0,
                 0.5,-0.5,
                 0.5, 0.5,  self.z_mean
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$x_4$', r'$y_4$',
                r'$gx_1$', r'$gy_1$',
                r'$gx_2$', r'$gy_2$',
                r'$gx_3$', r'$gy_3$',
                r'$gx_4$', r'$gy_4$',  r'$z$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 16,
        }
