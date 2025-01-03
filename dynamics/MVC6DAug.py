from .dynamics import Dynamics
import torch
import math

class MVC6DAug(Dynamics):
    def __init__(self):
        self.set_mode = 'reach'
        self.angle_alpha_factor = 1.2
        self.velocity = 1
        self.omega_max = 1.57
        self.collisionR = 0.25
        self.goalR = 0.1
        self.alpha_time = 1.0
        self.z_mean = 4.2
        self.z_var = 4.3
        self.G_1 = [-0.5,-0.5]
        self.G_2 = [ 0.5, 0.5]
        super().__init__(
            loss_type='brt_aug_hjivi', set_mode=self.set_mode, state_dim=7, input_dim=8, control_dim=2, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0,
                0, 0, self.z_mean,
            ],
            state_var=[
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor *
                math.pi, self.z_var,
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
            [-math.pi, math.pi], [-math.pi, math.pi],
            [self.z_mean - self.z_var, self.z_mean + self.z_var]
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 4] = (
            wrapped_state[..., 4] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 5] = (
            wrapped_state[..., 5] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 4])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 4])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 5])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 5])
        dsdt[..., 4] = control[..., 0]
        dsdt[..., 5] = control[..., 1]
        dsdt[..., 6] = - self.l_x(state)
        return dsdt
    
    def l_x(self, state): # to be changed
        dist1 = state[..., 0:2]*1.0
        dist1[..., 0] = dist1[..., 0]-(self.G_1[0])
        dist1[..., 1] = dist1[..., 1]-(self.G_1[1])
        distance_1 = torch.norm(dist1[..., 0:2], dim=-1) - self.goalR

        dist2 = state[..., 2:4]*1.0
        dist2[..., 0] = dist2[..., 0]-(self.G_2[0])
        dist2[..., 1] = dist2[..., 1]-(self.G_2[1])
        distance_2 = torch.norm(dist2[..., 0:2], dim=-1) - self.goalR

        return distance_1 + distance_2 #torch.maximum(distance_1, torch.maximum(distance_2, distance_3)) #

    def avoid_fn(self, state):
        boundary_values = -torch.norm(
            state[..., 0:2] - state[..., 2:4], dim=-1) + self.collisionR
        return boundary_values
    
    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,6]) #self.avoid_fn(state)#
        # computed using NN
        # device=state.device
        # lx=self.BCNN(state.cuda()).to(device)
        # return lx

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.avoid_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        dvds[..., 4:6] = dvds[..., 4:6] / self.angle_alpha_factor

        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(self.angle_alpha_factor*state[..., 4]) * dvds[..., 0] + torch.sin(
            self.angle_alpha_factor*state[..., 4]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 4]) + \
            self.velocity*(torch.cos(self.angle_alpha_factor*state[..., 5]) * dvds[..., 2] + torch.sin(
            self.angle_alpha_factor*state[..., 5]) * dvds[..., 3]) - self.omega_max * torch.abs(dvds[..., 5])

        # Effect of time factor
        ham = ham * self.alpha_time - dvds[..., 6]*self.l_x(state)
        return ham

    def optimal_control(self, state, dvds):
        # dvds[..., 6] = -dvds[..., 6]
        return -1.0*self.omega_max*torch.sign(dvds[..., [4, 5]])

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                0, 0,
                0, 0,
                math.pi/2, math.pi/2, self.z_mean
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$\theta_1$', r'$\theta_2$', r'$z$'
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }
