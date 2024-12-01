from .dynamics import Dynamics
import torch
import math

class Boat2DAug(Dynamics):
    def __init__(self):
        self.goalR = 0.0
        self.avoid_fn_weight = -1
        self.v_max = 1
        super().__init__(
            loss_type='brt_aug_hjivi', set_mode="avoid",
            state_dim=3, input_dim=4, control_dim=2, disturbance_dim=0,
            state_mean=[-0.5, 0, 7.5],
            state_var=[2.5, 2, 7.6],
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-3, 2],
            [-2, 2],
            [-0.1, 15.1],
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
        dist1 = torch.norm(dp1[..., 0:2], float('inf'), dim=-1) -0.4 

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-1)
        dp2[..., 1] = 0.2*(dp2[..., 1]-(-1.5))
        dist2 = torch.norm(dp2[..., 0:2], float('inf'), dim=-1) - 0.2

        dist = self.avoid_fn_weight * (torch.minimum(dist1, dist2))
        return dist#torch.where((dist >= 0), dist*5, dist)

    def boundary_fn(self, state):
        return torch.maximum(self.avoid_fn(state), self.l_x(state) - state[...,2])

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def l_x(self, state):
        dist = state[..., 0:2]*1.0
        dist[..., 0] = dist[..., 0]-(1.5)
        dist[..., 1] = dist[..., 1]-(0)
        return torch.norm(dist[..., 0:2], dim=-1) -self.goalR

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return - torch.norm(dvds[..., 0:2], dim = -1) - dvds[..., 2]*self.l_x(state) + (2 - 0.5*state[..., 1]*state[..., 1])*dvds[..., 0]

    def optimal_control(self, state, dvds):
        # opt_v =  torch.where((- self.v_max*(torch.norm(dvds[..., 0:2], dim = -1) - dvds[..., 2]) >= 0), 0, self.v_max)
        # opt_angle = torch.atan2(dvds[..., 1], dvds[..., 0]) + math.pi
        return NotImplementedError#torch.cat((opt_v, opt_angle), dim=-1)

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

class Boat2DAug_RAF(Dynamics):
    def __init__(self):
        self.goalR = 0.25
        self.avoid_fn_weight = 1
        self.v_max = 1
        super().__init__(
            loss_type='brat_hjivi', set_mode="reach",
            state_dim=3, input_dim=4, control_dim=2, disturbance_dim=0,
            state_mean=[-0.5, 0, 5],
            state_var=[2.5, 2, 5],
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-3, 2],
            [-2, 2],
            [0, 10],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # wrapped_state[..., 2] = (
        #     wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    # Boat2D dynamics
    # \dot x    = v \cos \theta  + 2 - 0.5*y^2
    # \dot y    = v \sin \theta
    # \dot \theta = - v
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = control[..., 0] * torch.cos(control[..., 1]) + 2 - 0.5*state[..., 1]*state[..., 1]
        dsdt[..., 1] = control[..., 0] * torch.sin(control[..., 1])
        dsdt[..., 2] = - control[..., 0]
        return dsdt

    def reach_fn(self, state):
        # goal_pose (-2.5,-0.3) (0.4,2.8)  l1(2.5,-2.5)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(1.5)
        dp1[..., 1] = dp1[..., 1]-(0)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) - self.goalR

        # dist_x = torch.abs(state[..., 0]-(-0.9))-0.2
        # dist_y = torch.abs(state[..., 1]-(-0.3))-0.2
        # dist = torch.max(dist_x, dist_y)
        # return torch.where((dist >= 0), dist, dist*10.0)

        # # First compute the l(x) as you normally would but then normalize it later.
        # dp1 = state[..., 0:2]*1.0
        # dp1[..., 0] = dp1[..., 0]-(-0.9)
        # dp1[..., 1] = dp1[..., 1]-(-0.3)
        # dist1 = torch.norm(dp1[..., 0:2], dim=-1) - 0.2
        # # dist1[dist1 < 0] *= 10.0
        return dist1

    def avoid_fn(self, state):
        # distance between the vehicle and obstacles
        # obstacles (x,y,r): (-2,-1,0.6), (-0.1,0.2,0.5), (-1,1.5,0.9), (1,2.3,0.4), (1.5,0.1,0.9), (0.7,-1.6,0.6), (1,3,0.4)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(-0.5)
        dp1[..., 1] = dp1[..., 1]-(0.5)
        dist1 = torch.norm(dp1[..., 0:2], float('inf'), dim=-1) -0.4 

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-1)
        dp2[..., 1] = 0.2*(dp2[..., 1]-(-1.5))
        dist2 = torch.norm(dp2[..., 0:2], float('inf'), dim=-1) - 0.2

        dist = self.avoid_fn_weight * (torch.minimum(dist1, dist2))
        return torch.where((dist >= 0), dist*5, dist)

    def boundary_fn(self, state):
        return torch.maximum(torch.maximum(self.reach_fn(state), -self.avoid_fn(state)), - state[...,2])

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return torch.minimum(torch.zeros(state.shape[0], 1).to(state.device), - self.v_max*(torch.norm(dvds[..., 0:2], dim = -1) - dvds[..., 2])) + ( 2 - 0.5*state[..., 1]*state[..., 1])*dvds[..., 0]

    def optimal_control(self, state, dvds):
        opt_v =  torch.where((- self.v_max*(torch.norm(dvds[..., 0:2], dim = -1) - dvds[..., 2]) >= 0), 0, self.v_max)
        opt_angle = torch.atan2(dvds[..., 1], dvds[..., 0]) + math.pi
        return torch.cat((opt_v, opt_angle), dim=-1)

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