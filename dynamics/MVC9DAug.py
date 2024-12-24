from .dynamics import Dynamics
import torch
import math

class MVC9DAug(Dynamics):
    def __init__(self):
        self.set_mode = 'reach'
        self.angle_alpha_factor = 1.2
        self.velocity = 1
        self.omega_max = 1.1
        self.collisionR = 0.1
        self.alpha_time = 1.0
        self.z_mean = 2.8
        self.z_var = 2.9
        self.G_1 = [-0.9, 0.0]
        self.G_2 = [ 0.9,-0.9]
        self.G_3 = [ 0.9, 0.9]
        super().__init__(
            loss_type='brt_aug_hjivi', set_mode=self.set_mode, state_dim=10, input_dim=11, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0,
                0, 0,
                0, 0, 0, self.z_mean,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor *
                math.pi, self.angle_alpha_factor*math.pi, self.z_var,
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
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],
            [self.z_mean - self.z_var, self.z_mean + self.z_var]
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
        dsdt[..., 9] = - self.l_x(state)
        return dsdt
    
    def l_x(self, state): # to be changed
        dist1 = state[..., 0:2]*1.0
        dist1[..., 0] = dist1[..., 0]-(self.G_1[0])
        dist1[..., 1] = dist1[..., 1]-(self.G_1[1])
        distance_1 = torch.norm(dist1[..., 0:2], dim=-1)

        dist2 = state[..., 2:4]*1.0
        dist2[..., 0] = dist2[..., 0]-(self.G_2[0])
        dist2[..., 1] = dist2[..., 1]-(self.G_2[1])
        distance_2 = torch.norm(dist2[..., 0:2], dim=-1)

        dist3 = state[..., 4:6]*1.0
        dist3[..., 0] = dist3[..., 0]-(self.G_3[0])
        dist3[..., 1] = dist3[..., 1]-(self.G_3[1])
        distance_3 = torch.norm(dist3[..., 0:2], dim=-1)

        return distance_1+ distance_2+ distance_3#torch.maximum(distance_1, torch.maximum(distance_2, distance_3))

    def avoid_fn(self, state):
        boundary_values = -torch.norm(
            state[..., 0:2] - state[..., 2:4], dim=-1) + self.collisionR
        for i in range(1, 2):
            boundary_values_current = -torch.norm(
                state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) + self.collisionR
            boundary_values = torch.max(
                boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = -torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] -
                                                     state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) + self.collisionR
                boundary_values = torch.max(
                    boundary_values, boundary_values_current)
        return boundary_values
    
    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,9]) #self.avoid_fn(state)#
        # computed using NN
        # device=state.device
        # lx=self.BCNN(state.cuda()).to(device)
        # return lx

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.avoid_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        dvds[..., 6:] = dvds[..., 6:] / self.angle_alpha_factor

        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(self.angle_alpha_factor*state[..., 6]) * dvds[..., 0] + torch.sin(
            self.angle_alpha_factor*state[..., 6]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 6])

        # Hamiltonian effect due to other vehicles
        for i in range(2):
            theta_index = 7+i
            xcostate_index = 2*(i+1)
            ycostate_index = 2*(i+1) + 1
            thetacostate_index = 7+i
            ham_local = self.velocity*(torch.cos(self.angle_alpha_factor*state[..., theta_index]) * dvds[..., xcostate_index] + torch.sin(
                self.angle_alpha_factor*state[..., theta_index]) * dvds[..., ycostate_index]) - self.omega_max * torch.abs(dvds[..., thetacostate_index])
            ham = ham + ham_local

        # Effect of time factor
        ham = ham * self.alpha_time - dvds[..., 9]*self.l_x(state)
        return ham

    def optimal_control(self, state, dvds):
        # dvds[..., 6] = -dvds[..., 6]
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                0, 0,
                -0.4, 0,
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4, self.z_mean
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$z$'
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 9,
        }
