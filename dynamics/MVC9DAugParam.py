from .dynamics import Dynamics
import torch
import math

class MVC9DAugParam(Dynamics):
    def __init__(self):
        self.set_mode = 'reach'
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        self.goalR = 0.0
        self.alpha_time = 1.0
        self.z_mean = 2.1
        self.z_var = 2.1
        self.G_1 = [-0.5, 0.0]
        self.G_2 = [ 0.5,-0.5]
        self.G_3 = [ 0.5, 0.5]
        super().__init__(
            loss_type='brt_aug_hjivi', set_mode=self.set_mode, state_dim=16, input_dim=17, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0,
                0, 0,
                0, 0, 0, 
                0, 0,
                0, 0,
                0, 0,self.z_mean,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor *
                math.pi, self.angle_alpha_factor*math.pi,
                0.9, 0.9,
                0.9, 0.9,
                0.9, 0.9,  self.z_var,
            ],
            value_mean=0.0,
            value_var=1.0,
            value_normto=1.0,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],
            [-0.9, 0.9], [-0.9, 0.9],
            [-0.9, 0.9], [-0.9, 0.9],
            [-0.9, 0.9], [-0.9, 0.9],
            [self.z_mean - self.z_var, self.z_mean + self.z_var],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 6] = (
            wrapped_state[..., 6] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (
            wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 8] = (
            wrapped_state[..., 8] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        dsdt[..., 15] = -self.l_x(state)
        return dsdt
    
    def l_x(self, state): # to be changed
        dist1 = state[..., 0:2]*1.0
        dist1[..., 0] = dist1[..., 0]- state[..., 9]
        dist1[..., 1] = dist1[..., 1]- state[..., 10]
        distance_1 = torch.norm(dist1[..., 0:2], dim=-1) - self.goalR

        dist2 = state[..., 2:4]*1.0
        dist2[..., 0] = dist2[..., 0]- state[..., 11]
        dist2[..., 1] = dist2[..., 1]- state[..., 12]
        distance_2 = torch.norm(dist2[..., 0:2], dim=-1) - self.goalR

        dist3 = state[..., 4:6]*1.0
        dist3[..., 0] = dist3[..., 0]- state[..., 13]
        dist3[..., 1] = dist3[..., 1]- state[..., 14]
        distance_3 = torch.norm(dist3[..., 0:2], dim=-1) - self.goalR

        return (distance_1 + distance_2 + distance_3)/3 #torch.maximum(distance_1, torch.maximum(distance_2, distance_3)) #

    def avoid_fn(self, state):
        boundary_values_12 = -torch.norm(
            state[..., 0:2] - state[..., 2:4], dim=-1) + self.collisionR
        boundary_values_13 = -torch.norm(
            state[..., 0:2] - state[..., 4:6], dim=-1) + self.collisionR
        boundary_values_23 = -torch.norm(
            state[..., 2:4] - state[..., 4:6], dim=-1) + self.collisionR
        boundary_values = torch.max(boundary_values_23, torch.max(
                    boundary_values_12, boundary_values_13))
        return boundary_values
    
    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,15])

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.max(self.avoid_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        dvds[..., 6:9] = dvds[..., 6:9] / self.angle_alpha_factor

        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(self.angle_alpha_factor*state[..., 6]) * dvds[..., 0] + torch.sin(
            self.angle_alpha_factor*state[..., 6]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 6]) + \
            self.velocity*(torch.cos(self.angle_alpha_factor*state[..., 7]) * dvds[..., 2] + torch.sin(
            self.angle_alpha_factor*state[..., 7]) * dvds[..., 3]) - self.omega_max * torch.abs(dvds[..., 7]) + \
            self.velocity*(torch.cos(self.angle_alpha_factor*state[..., 8]) * dvds[..., 4] + torch.sin(
            self.angle_alpha_factor*state[..., 8]) * dvds[..., 5]) - self.omega_max * torch.abs(dvds[..., 8]) 

        ham = ham * self.alpha_time - dvds[..., 15]*self.l_x(state)
        return ham

    def optimal_control(self, state, dvds):
        return -1.0*self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                0, 0,
                0, -0.4,
                0, 0.4,
                math.pi/2, 0.0, 0.0,
               -0.5, 0.0,
                0.5,-0.5,
                0.5, 0.5,  self.z_mean
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
                r'$gx_1$', r'$gy_1$',
                r'$gx_2$', r'$gy_2$',
                r'$gx_3$', r'$gy_3$',  r'$z$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 15,
        }
