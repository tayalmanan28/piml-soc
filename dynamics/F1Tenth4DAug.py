from .dynamics import Dynamics
import torch
import math

class F1Tenth4DAug(Dynamics):
    def __init__(self):
        self.steer_max = 0.7
        self.a_max = 1
        self.L = 1
        self.avoid_fn_weight = -1
        self.set_mode = 'avoid'

        super().__init__(
            loss_type='brt_aug_hjivi', set_mode='avoid',
            state_dim=5, input_dim=6, control_dim=2, disturbance_dim=0,
            state_mean=[0, 0, 5, 0, 82],
            state_var=[30, 30, 5, math.pi, 82.1],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0
        )

    def control_range(self, state):
        return [[-self.a_max, self.a_max], [-self.steer_max, self.steer_max]]

    def state_test_range(self):
        return [
            [-30, 30],
            [-30, 30],
            [0, 10],
            [-math.pi, math.pi],
            [-0.1, 164.1]

        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 3] = (
            wrapped_state[..., 3] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :6] = input[..., :6]
        transformed_input[..., 4] = torch.sin(input[..., 4]*self.state_var[3])
        transformed_input[..., 6] = torch.cos(input[..., 4]*self.state_var[3])
        return transformed_input.cuda()
    # Dynamics of the F1Tenth4DAug Car:
    #         \dot{x}_0 = x_2 * cos(x_3)
    #         \dot{x}_1 = x_2 * sin(x_3)
    #         \dot{x}_2 = a
    #         \dot{x}_3 = x_2/L * sin(beta)
    #         \dot{z}   = - l(x)
    #         \beta = arctan(0.5tan(steer))
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        beta = torch.arctan(0.5*torch.tan(control[..., 1]))
        dsdt[..., 0] = state[..., 2] * \
            torch.cos(state[..., 3]) 
        dsdt[..., 1] = state[..., 2] * \
            torch.sin(state[..., 3]) 
        dsdt[..., 2] = control[..., 0] 
        dsdt[..., 3] = state[..., 2]/self.L * \
            torch.sin(beta) 
        dsdt[..., 4] = - self.l_x(state)
        return dsdt

    def reach_fn(self, state):
        return NotImplementedError

    def avoid_fn(self, state):
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(-5)
        dp1[..., 1] = dp1[..., 1]-(5)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) -4 

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-10)
        dp2[..., 1] = 0.2*(dp2[..., 1]-(-15))
        dist2 = torch.norm(dp2[..., 0:2], dim=-1) - 2

        dist = self.avoid_fn_weight * (torch.minimum(dist1, dist2))
        return dist#torch.where((dist >= 0), dist*5, dist)

    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,4])

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def l_x(self, state):
        dist = state[..., 0:2]*1.0
        dist[..., 0] = dist[..., 0]-(15)
        dist[..., 1] = dist[..., 1]-(0)
        return torch.norm(dist[..., 0:2], dim=-1)

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            return (state[..., 2] * torch.cos(state[..., 3]) * dvds[..., 0]) \
                    + (state[..., 2] * torch.sin(state[..., 3]) * dvds[..., 1]) \
                    - (self.a_max) * torch.abs(dvds[..., 2])\
                    - torch.abs(state[..., 2] / self.L * dvds[..., 3])\
                    - (self.l_x(state) * dvds[..., 4])

        elif self.set_mode == 'avoid':
            return (state[..., 2] * torch.cos(state[..., 3]) * dvds[..., 0]) \
                    + (state[..., 2] * torch.sin(state[..., 3]) * dvds[..., 1]) \
                    + (self.a_max) * torch.abs(dvds[..., 2])\
                    + torch.abs(state[..., 2] / self.L * dvds[..., 3])\
                    - (self.l_x(state) * dvds[..., 4])

        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        # if self.set_mode == 'reach':
        #     control1 = (-self.a_max*torch.sign(dvds[..., 2]))[..., None]
        #     control2 = (-self.w_max*torch.sign(dvds[..., 4]))[..., None]
        #     return torch.cat((control1, control2), dim=-1)

        # elif self.set_mode == 'avoid':
        #     control1 = (self.a_max*torch.sign(dvds[..., 2]))[..., None]
        #     control2 = (self.w_max*torch.sign(dvds[..., 4]))[..., None]
        #     return torch.cat((control1, control2), dim=-1)
        raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        return torch.zeros_like(state)

    def plot_config(self):
        return {
            'state_slices': [0, 0, 4, 0, 0],
            'state_labels': ['px', 'py', 'v', r'$\beta$', 'z'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }