from abc import ABC, abstractmethod
from utils import diff_operators, quaternion
from utils.modules import BCNetwork
import math
import torch
import scipy
# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parametrized models


class Dynamics(ABC):
    def __init__(self,
                 loss_type: str, set_mode: str,
                 state_dim: int, input_dim: int,
                 control_dim: int, disturbance_dim: int,
                 state_mean: list, state_var: list,
                 value_mean: float, value_var: float, value_normto: float,
                 deepReach_model: bool, exact_factor: float):
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean)
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepReach_model = deepReach_model
        self.exact_factor = exact_factor

        assert self.loss_type in [
            'brt_hjivi', 'brt_aug_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in [
            'reach', 'avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + \
                str(len(state_descriptor)) + ' != ' + str(self.state_dim)

    # ALL METHODS ARE BATCH COMPATIBLE

    # MODEL-UNIT CONVERSIONS (TODO: refactor into separate model-unit conversion class?)

    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)
                          ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)
                          ) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == 'diff':
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact':

            # return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])

            # Crude Fix to handle NaN in Rocket Landing. V(x,t) = l(x) + (k*t + (1-k))*NN(x,t).
            # k=1 gives us V(x,t) = l(x) + t*NN(x,t) which is the correct exact_BC variant.
            # Setting k=1 gives NaN for the Rocket Landing example as the PDE loss but works if we keep k = 1 - ε where ε <<<< 1.
            # Hence, we keep ε = 1e-7. I feel the performance won't vary much as k ≈ 1.
            # Finally, k=1 for every variant except Rocket Landing where k = 0.9999999.
            k = self.exact_factor
            exact_BC_factor = k * input[..., 0] + (1-k)
            return (output * exact_BC_factor * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact_sin':
            # just for testing purpose, tMax is hardcoded to be 2
            return (output * torch.sin(input[..., 0]/2) * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact_exp':
            return (output * (-torch.exp(-5*input[..., 0])+1.0) * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact_diff':
            # V(x,t)= l(x) + NN(x,t) - NN(x,0)
            # print(input.shape, output.shape)
            # print(output[0])
            # print(output[1])
            output0 = output[0].squeeze(dim=-1)
            output1 = output[1].squeeze(dim=-1)
            return (output0 - output1) * self.value_var / self.value_normto + self.boundary_fn(self.input_to_coord(input[0].detach())[..., 1:])
        elif self.deepReach_model == 'reg':
            return (output * self.value_var / self.value_normto) + self.value_mean
        else:
            raise NotImplementedError

    # convert model io to real dv
    def io_to_dv(self, input, output):
        if self.deepReach_model == 'exact_diff':

            dodi1 = diff_operators.jacobian(
                output[0], input[0])[0].squeeze(dim=-2)
            dodi2 = diff_operators.jacobian(
                output[1], input[1])[0].squeeze(dim=-2)

            dvdt = (self.value_var / self.value_normto) * dodi1[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi1.device)) * (dodi1[..., 1:]-dodi2[..., 1:])

            state = self.input_to_coord(input[0])[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
            return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

        dodi = diff_operators.jacobian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        elif self.deepReach_model == 'exact':

            # Shrewd Fix to handle NaN in Rocket Landing. V(x,t) = l(x) + (k*t + (1-k))*NN(x,t).
            # k=1 gives us V(x,t) = l(x) + t*NN(x,t) which is the original variant.
            # Setting k=1 gives NaN as the PDE loss but works if we keep k = 1 - ε where ε <<<< 1.
            # Hence, we keep ε = 1e-7. I feel the performance won't vary much as k ≈ 1.
            # To summarize, k=1 for every variant except Rocket Landing where k = 0.9999999.
            k = self.exact_factor
            exact_BC_factor = k * input[..., 0] + (1-k)
            exact_factor_der = k
            dvdt = (self.value_var / self.value_normto) * \
                (exact_BC_factor*dodi[..., 0] + exact_factor_der*output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * exact_BC_factor.unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            # boundary_values = self.boundary_fn(state).unsqueeze(dim=-1)
            # gradient = torch.ones_like(boundary_values)
            # state.retain_grad()
            # boundary_values.backward(gradient=gradient)
            # dvds_term2 = state.grad
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
               state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepReach_model == 'exact_sin':

            dvdt = (self.value_var / self.value_normto) * \
                (torch.sin(input[..., 0]/2)*dodi[..., 0] +
                 0.5*torch.cos(input[..., 0]/2)*output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * torch.sin(input[..., 0]/2).unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepReach_model == 'exact_exp':

            dvdt = (self.value_var / self.value_normto) * \
                ((-torch.exp(-5*input[..., 0])+1)*dodi[...,
                 0] + 5*torch.exp(-5*input[..., 0])*output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * (-torch.exp(-5*input[..., 0])+1).unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        elif self.deepReach_model == 'reg':
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto /
                    self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        else:
            raise NotImplementedError
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # convert model io to real dv
    # TODO: need implementation for exact BC model and exact diff BC model
    def io_to_2nd_derivative(self, input, output):
        hes = diff_operators.batchHessian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            vis_term1 = (self.value_var / self.value_normto /
                         self.state_var.to(device=hes.device))**2 * hes[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            vis_term2 = diff_operators.batchHessian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            hes = vis_term1 + vis_term2

        else:
            hes = (self.value_var / self.value_normto /
                   self.state_var.to(device=hes.device))**2 * hes[..., 1:]

        return hes

    def set_model(self, deepreach_model):
        self.deepReach_model = deepreach_model
    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError


# class DroneDelivery(Dynamics):
#     def __init__(self):
#         self.goalR = 0.5
#         self.velocity = 4.0
#         self.omega_max = 2.0
#         self.angle_alpha_factor = 1.2
#         self.avoid_fn_weight = 1
#         self.d_bar = 0.6
#         super().__init__(
#             loss_type='brat_hjivi', set_mode="reach",
#             state_dim=4, input_dim=5, control_dim=1, disturbance_dim=2,
#             state_mean=[0, 0, 0, self.d_bar/2],
#             state_var=[3, 3, self.angle_alpha_factor * math.pi, self.d_bar/2],
#             value_mean=0.5,
#             value_var=1,
#             value_normto=0.02,
#             deepReach_model='exact',
#             exact_factor=1.0,
#         )

#     def state_test_range(self):
#         return [
#             [-3, 3],
#             [-3, 3],
#             [-math.pi, math.pi],
#             [0, self.d_bar],
#         ]

#     def equivalent_wrapped_state(self, state):
#         wrapped_state = torch.clone(state)
#         wrapped_state[..., 2] = (
#             wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
#         return wrapped_state

#     # Dubins3D dynamics
#     # \dot x    = v \cos \theta
#     # \dot y    = v \sin \theta
#     # \dot \theta = u
#     def dsdt(self, state, control, disturbance):
#         dsdt = torch.zeros_like(state)
#         dsdt[..., 0] = self.velocity * \
#             torch.cos(state[..., 2])+disturbance[..., 0]
#         dsdt[..., 1] = self.velocity * \
#             torch.sin(state[..., 2])+disturbance[..., 1]
#         dsdt[..., 2] = control[..., 0]
#         dsdt[..., 3] = 0.0
#         return dsdt

#     def reach_fn(self, state):
#         # goal_pose (-2.5,-0.3) (0.4,2.8)  l1(2.5,-2.5)
#         dp1 = state[..., 0:2]*1.0
#         dp1[..., 0] = dp1[..., 0]-(0)
#         dp1[..., 1] = dp1[..., 1]-(2.3)
#         dist1 = torch.norm(dp1[..., 0:2], dim=-1) - self.goalR

#         return dist1

#     def avoid_fn(self, state):
#         # distance between the vehicle and obstacles
#         # obstacles (x,y,r): (-2,-1,0.6)  (-0.8,-0.2,0.6) (-1,1.5,1) (1,2.5,0.5) (1.5,0.1,0.9)
#         dp1 = state[..., 0:2]*1.0
#         dist1 = torch.maximum(
#             torch.abs(dp1[..., 0]-(-1.9))-1.1, torch.abs(dp1[..., 1])-1.6)

#         dp2 = state[..., 0:2]*1.0
#         dist2 = torch.maximum(
#             torch.abs(dp2[..., 0]-(0.3))-0.5, torch.abs(dp2[..., 1])-1.6)

#         dist = self.avoid_fn_weight * torch.minimum(dist1, dist2)
#         return dist

#     def boundary_fn(self, state):
#         return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

#     def sample_target_state(self, num_samples):
#         raise NotImplementedError

#     def cost_fn(self, state_traj):
#         return torch.min(self.boundary_fn(state_traj), dim=-1).values

#     def hamiltonian(self, state, dvds):
#         ham = self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(
#             state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2])
#         ham += torch.abs(dvds[..., 0])*state[..., 3] + \
#             torch.abs(dvds[..., 1])*state[..., 3]
#         return ham

#     def optimal_control(self, state, dvds):
#         return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]

#     def optimal_disturbance(self, state, dvds):
#         d1 = -state[..., 3]*torch.sign(dvds[..., 0])
#         d2 = -state[..., 3]*torch.sign(dvds[..., 1])

#         return torch.cat((d1[..., None], d2[..., None]), dim=-1)

#     def plot_config(self):
#         return {
#             'state_slices': [0, 0, 0, 0.5],
#             'state_labels': ['x', 'y', r'$\theta$', 'd'],
#             'x_axis_idx': 0,
#             'y_axis_idx': 1,
#             'z_axis_idx': 3,
#         }


class DroneDelivery(Dynamics):
    def __init__(self):
        self.goalR = 4.0
        self.velocity = 4.0
        self.omega_max = 2.0
        self.angle_alpha_factor = 1.2
        self.avoid_fn_weight = 1
        self.d_bar = 1.5
        super().__init__(
            loss_type='brat_hjivi', set_mode="reach",
            state_dim=4, input_dim=5, control_dim=1, disturbance_dim=2,
            state_mean=[0, 0, 0, self.d_bar/2],
            state_var=[20, 20, self.angle_alpha_factor *
                       math.pi, self.d_bar/2],
            value_mean=10,
            value_var=12,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-20, 20],
            [-20, 20],
            [-math.pi, math.pi],
            [0, self.d_bar],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * \
            torch.cos(state[..., 2])+disturbance[..., 0]
        dsdt[..., 1] = self.velocity * \
            torch.sin(state[..., 2])+disturbance[..., 1]
        dsdt[..., 2] = control[..., 0]
        dsdt[..., 3] = 0.0
        return dsdt

    def reach_fn(self, state):
        # goal_pose (-2.5,-0.3) (0.4,2.8)  l1(2.5,-2.5)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(0)
        dp1[..., 1] = dp1[..., 1]-(5)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) - self.goalR

        return dist1

    def avoid_fn(self, state):
        # distance between the vehicle and obstacles
        # obstacles (x,y,r): (-2,-1,0.6)  (-0.8,-0.2,0.6) (-1,1.5,1) (1,2.5,0.5) (1.5,0.1,0.9)
        dp1 = state[..., 0:2]*1.0
        dist1 = torch.maximum(
            torch.abs(dp1[..., 0]-(-11.5))-8.5, torch.abs(dp1[..., 1]-(-3))-3)

        dp2 = state[..., 0:2]*1.0
        dist2 = torch.maximum(
            torch.abs(dp2[..., 0]-(2))-3, torch.abs(dp2[..., 1]-(-3))-3)

        dist = self.avoid_fn_weight * torch.minimum(dist1, dist2)
        return dist

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        ham = self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(
            state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2])
        ham += torch.abs(dvds[..., 0])*state[..., 3] + \
            torch.abs(dvds[..., 1])*state[..., 3]
        return ham

    def optimal_control(self, state, dvds):
        return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        d1 = -state[..., 3]*torch.sign(dvds[..., 0])
        d2 = -state[..., 3]*torch.sign(dvds[..., 1])

        return torch.cat((d1[..., None], d2[..., None]), dim=-1)

    def plot_config(self):
        return {
            'state_slices': [0, 0, 1.57, 0.5],
            'state_labels': ['x', 'y', r'$\theta$', 'd'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 3,
        }

class Boat2DAug(Dynamics):
    def __init__(self):
        self.goalR = 0.0
        self.avoid_fn_weight = -1
        self.v_max = 1
        super().__init__(
            loss_type='brt_aug_hjivi', set_mode="avoid",
            state_dim=3, input_dim=4, control_dim=2, disturbance_dim=0,
            state_mean=[-0.5, 0, 3],
            state_var=[2.5, 2, 3.1],
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
            [-0.1, 6.1],
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
        return torch.maximum(self.avoid_fn(state), - state[...,2])

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
    
class Boat2D(Dynamics):
    def __init__(self):
        self.goalR = 0.25
        self.avoid_fn_weight = 1
        self.v_max = 1
        super().__init__(
            loss_type='brat_hjivi', set_mode="reach",
            state_dim=2, input_dim=3, control_dim=2, disturbance_dim=0,
            state_mean=[0, 0],
            state_var=[2, 2],
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-2, 2],
            [-2, 2],
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
        return dist#torch.where((dist >= 0), dist*5, dist)

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return torch.minimum(torch.zeros(state.shape[0], 1).to(state.device), - self.v_max*(torch.norm(dvds[..., 0:2], dim = -1))) + ( 2 - 0.5*state[..., 1]*state[..., 1])*dvds[..., 0]

    def optimal_control(self, state, dvds):
        return NotImplementedError#(-self.omega_max * torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0],
            'state_labels': ['x', 'y'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': -1,
        }

class Dubins3DReachAvoid(Dynamics):
    def __init__(self):
        self.goalR = 0.2
        self.velocity = 1.5
        self.omega_max = 1.2
        self.angle_alpha_factor = 1.2
        self.avoid_fn_weight = 1
        super().__init__(
            loss_type='brat_hjivi', set_mode="reach",
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0, 0],
            state_var=[3, 3, self.angle_alpha_factor * math.pi],
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt

    def reach_fn(self, state):
        # goal_pose (-2.5,-0.3) (0.4,2.8)  l1(2.5,-2.5)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(0)
        dp1[..., 1] = dp1[..., 1]-(-0.5)
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
        dp1[..., 0] = dp1[..., 0]-(-2.0)
        dp1[..., 1] = dp1[..., 1]-(-1.0)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) - 0.6

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-0.1)
        dp2[..., 1] = dp2[..., 1]-(0.2)
        dist2 = torch.norm(dp2[..., 0:2], dim=-1) - 0.5

        dp3 = state[..., 0:2]*1.0
        dp3[..., 0] = dp3[..., 0]-(-1.0)
        dp3[..., 1] = dp3[..., 1]-(1.5)
        dist3 = torch.norm(dp3[..., 0:2], dim=-1) - 0.9

        dp4 = state[..., 0:2]*1.0
        dp4[..., 0] = dp4[..., 0]-(1)
        dp4[..., 1] = dp4[..., 1]-(2.3)
        dist4 = torch.norm(dp4[..., 0:2], dim=-1) - 0.4

        dp5 = state[..., 0:2]*1.0
        dp5[..., 0] = dp5[..., 0]-(1.5)
        dp5[..., 1] = dp5[..., 1]-(0.1)
        dist5 = torch.norm(dp5[..., 0:2], dim=-1) - 0.9

        dp6 = state[..., 0:2]*1.0
        dp6[..., 0] = dp6[..., 0]-(0.7)
        dp6[..., 1] = dp6[..., 1]-(-1.6)
        dist6 = torch.norm(dp6[..., 0:2], dim=-1) - 0.6

        dp7 = state[..., 0:2]*1.0
        dp7[..., 0] = dp7[..., 0]-(1)
        dp7[..., 1] = dp7[..., 1]-(3)
        dist7 = torch.norm(dp7[..., 0:2], dim=-1) - 0.4

        dist = self.avoid_fn_weight * torch.minimum(torch.minimum(torch.minimum(torch.minimum(
            torch.minimum(torch.minimum(dist1, dist2), dist3), dist4), dist5), dist6), dist7)
        return torch.where((dist >= 0), dist*5, dist)

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class Dubins3DReachAvoid2(Dynamics):
    def __init__(self):
        self.goalR = 0.2
        self.velocity = 1.5
        self.omega_max = 1.2
        self.angle_alpha_factor = 1.2
        self.avoid_fn_weight = 1
        super().__init__(
            loss_type='brat_hjivi', set_mode="reach",
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0, 0],
            state_var=[3, 3, self.angle_alpha_factor * math.pi],
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt

    def reach_fn(self, state):
        # goal_pose (-2.5,-0.3) (0.4,2.8)  l1(2.5,-2.5)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(-2.5)
        dp1[..., 1] = dp1[..., 1]-(-0.3)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) - self.goalR

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(0.4)
        dp2[..., 1] = dp2[..., 1]-(2.8)
        dist2 = torch.norm(dp2[..., 0:2], dim=-1) - self.goalR

        dp3 = state[..., 0:2]*1.0
        dp3[..., 0] = dp3[..., 0]-(2.5)
        dp3[..., 1] = dp3[..., 1]-(-2.5)
        dist3 = torch.norm(dp3[..., 0:2], dim=-1) - self.goalR

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
        dist = torch.minimum(torch.minimum(dist1, dist2), dist3)
        dist[dist < 0] *= 10.0
        return dist

    def avoid_fn(self, state):
        # distance between the vehicle and obstacles
        # obstacles (x,y,r): (-2,-1,0.6)  (-0.8,-0.2,0.6) (-1,1.5,1) (1,2.5,0.5) (1.5,0.1,0.9)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(-2.0)
        dp1[..., 1] = dp1[..., 1]-(-1.0)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) - 0.6

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-0.1)
        dp2[..., 1] = dp2[..., 1]-(0.2)
        dist2 = torch.norm(dp2[..., 0:2], dim=-1) - 0.5

        dp3 = state[..., 0:2]*1.0
        dp3[..., 0] = dp3[..., 0]-(-1.0)
        dp3[..., 1] = dp3[..., 1]-(1.5)
        dist3 = torch.norm(dp3[..., 0:2], dim=-1) - 0.9

        dp4 = state[..., 0:2]*1.0
        dp4[..., 0] = dp4[..., 0]-(1)
        dp4[..., 1] = dp4[..., 1]-(2.3)
        dist4 = torch.norm(dp4[..., 0:2], dim=-1) - 0.4

        dp5 = state[..., 0:2]*1.0
        dp5[..., 0] = dp5[..., 0]-(1.5)
        dp5[..., 1] = dp5[..., 1]-(0.1)
        dist5 = torch.norm(dp5[..., 0:2], dim=-1) - 0.9

        dp6 = state[..., 0:2]*1.0
        dp6[..., 0] = dp6[..., 0]-(0.7)
        dp6[..., 1] = dp6[..., 1]-(-1.6)
        dist6 = torch.norm(dp6[..., 0:2], dim=-1) - 0.6

        dp7 = state[..., 0:2]*1.0
        dp7[..., 0] = dp7[..., 0]-(1)
        dp7[..., 1] = dp7[..., 1]-(3)
        dist7 = torch.norm(dp7[..., 0:2], dim=-1) - 0.4

        dist = self.avoid_fn_weight * torch.minimum(torch.minimum(torch.minimum(torch.minimum(
            torch.minimum(torch.minimum(dist1, dist2), dist3), dist4), dist5), dist6), dist7)
        return torch.where((dist >= 0), dist*5, dist)

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class Dubins3DAvoid(Dynamics):
    def __init__(self):
        self.goalR = 0.2
        self.velocity = 1.0
        self.omega_max = 1.2
        self.angle_alpha_factor = 1.2
        self.avoid_fn_weight = 1
        super().__init__(
            loss_type='brt_hjivi', set_mode="avoid",
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0, 0],
            state_var=[3, 3, self.angle_alpha_factor * math.pi],
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        # distance between the vehicle and obstacles
        # obstacles (x,y,r): (-2,-1,0.6)  (-0.8,-0.2,0.6) (-1,1.5,1) (1,2.5,0.5) (1.5,0.1,0.9)
        dp1 = state[..., 0:3]*1.0
        dp1[..., 0] = dp1[..., 0]-(0.0)
        dp1[..., 1] = dp1[..., 1]-(0.0)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) - 0.8

        #dp2 = state[..., 0:2]*1.0
        #dp2[..., 0] = dp2[..., 0]-(-0.1)
        #dp2[..., 1] = dp2[..., 1]-(0.2)
        #dist2 = torch.norm(dp2[..., 0:2], dim=-1) - 0.5

        #dp3 = state[..., 0:2]*1.0
        #dp3[..., 0] = dp3[..., 0]-(-1.0)
        #dp3[..., 1] = dp3[..., 1]-(1.5)
        #dist3 = torch.norm(dp3[..., 0:2], dim=-1) - 0.9

        #dp4 = state[..., 0:2]*1.0
        #dp4[..., 0] = dp4[..., 0]-(1)
        #dp4[..., 1] = dp4[..., 1]-(2.3)
        #dist4 = torch.norm(dp4[..., 0:2], dim=-1) - 0.4

        #dp5 = state[..., 0:2]*1.0
        #dp5[..., 0] = dp5[..., 0]-(1.5)
        #dp5[..., 1] = dp5[..., 1]-(0.1)
        #dist5 = torch.norm(dp5[..., 0:2], dim=-1) - 0.9

        #dp6 = state[..., 0:2]*1.0
        #dp6[..., 0] = dp6[..., 0]-(0.7)
        #dp6[..., 1] = dp6[..., 1]-(-1.6)
        #dist6 = torch.norm(dp6[..., 0:2], dim=-1) - 0.6

        #dp7 = state[..., 0:2]*1.0
        #dp7[..., 0] = dp7[..., 0]-(1)
        #dp7[..., 1] = dp7[..., 1]-(3)
        #dist7 = torch.norm(dp7[..., 0:2], dim=-1) - 0.4

        #dist = self.avoid_fn_weight * torch.minimum(torch.minimum(torch.minimum(torch.minimum(
        #    torch.minimum(torch.minimum(dist1, dist2), dist3), dist4), dist5), dist6), dist7)
        dist = (dp1[..., 0]*torch.cos(dp1[..., 2]) + dp1[..., 0]*torch.sin(dp1[..., 2]))/(torch.norm(dp1[..., 0:2], dim=-1)) + 10*dist1
        #dist = dist1
        return dist

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        return (self.omega_max * torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class ReachAvoidDubins(Dynamics):
    def __init__(self, avoid_fn_weight: float, avoid_only: bool):
        self.L = 2.0
        self.tMax = 4.0

        # # Target positions
        self.goalX = [4.0, -4.0]
        self.goalY = [-1.4, 1.4]

        # State bounds
        self.vMin = 0.001
        self.vMax = 6.50
        self.phiMin = -0.3 * math.pi + 0.001
        self.phiMax = 0.3 * math.pi - 0.001

        # Control bounds
        self.aMin = -4.0
        self.aMax = 2.0
        self.psiMin = -3.0 * math.pi
        self.psiMax = 3.0 * math.pi

        # Lower and upper curb positions (in the y direction)
        self.curb_positions = [-4.5, 4.5]

        # Stranded car position
        self.stranded_car_pos = [0.0, 0.0]

        self.avoid_fn_weight = avoid_fn_weight

        self.avoid_only = False

        super().__init__(
            loss_type='brt_hjivi' if self.avoid_only else 'brat_hjivi', set_mode='avoid' if self.avoid_only else 'reach',
            state_dim=10, input_dim=11, control_dim=4, disturbance_dim=0,
            # state = [x1, y1, th1, v1, phi1, x2, y2, th2, v2, phi2]
            state_mean=[
                0, 0, 0, 3, 0,
                0, 0, 0, 3, 0
            ],
            state_var=[
                5.0, 5.0, 1.2 * math.pi, 4.0, 1.2 * 0.3 * math.pi,
                5.0, 5.0, 1.2 * math.pi, 4.0, 1.2 * 0.3 * math.pi,
            ],
            value_mean=0.25 * 5.0,
            value_var=0.5 * 5.0,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-5, 5],
            [-5, 5],
            [-math.pi, math.pi],
            [0, 6.5],
            [-3.0, -3.0],
            [-5, 5],
            [-5, 5],
            [-math.pi, -math.pi],
            [0, 6.5],
            [-3.0, -3.0],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 7] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    # NarrowPassage dynamics
    # \dot x   = v * cos(th)
    # \dot y   = v * sin(th)
    # \dot th  = v * tan(phi) / L
    # \dot v   = u1
    # \dot phi = u2
    # \dot x   = ...
    # \dot y   = ...
    # \dot th  = ...
    # \dot v   = ...
    # \dot phi = ...
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3] * torch.cos(state[..., 2])
        dsdt[..., 1] = state[..., 3] * torch.sin(state[..., 2])
        dsdt[..., 2] = state[..., 3] * torch.tan(state[..., 4]) / self.L
        dsdt[..., 3] = control[..., 0]
        dsdt[..., 4] = control[..., 1]
        dsdt[..., 5] = state[..., 8] * torch.cos(state[..., 7])
        dsdt[..., 6] = state[..., 8] * torch.sin(state[..., 7])
        dsdt[..., 7] = state[..., 8] * torch.tan(state[..., 9]) / self.L
        dsdt[..., 8] = control[..., 2]
        dsdt[..., 9] = control[..., 3]
        return dsdt

    def reach_fn(self, state):
        if self.avoid_only:
            raise RuntimeError
        # vehicle 1
        goal_tensor_R1 = torch.tensor(
            [self.goalX[0], self.goalY[0]], device=state.device)
        dist_R1 = torch.norm(state[..., 0:2] - goal_tensor_R1, dim=-1) - self.L
        # vehicle 2
        goal_tensor_R2 = torch.tensor(
            [self.goalX[1], self.goalY[1]], device=state.device)
        dist_R2 = torch.norm(state[..., 5:7] - goal_tensor_R2, dim=-1) - self.L
        return torch.maximum(dist_R1, dist_R2)

    def avoid_fn(self, state):
        # distance from lower curb
        dist_lc_R1 = state[..., 1] - self.curb_positions[0] - 0.5 * self.L
        dist_lc_R2 = state[..., 6] - self.curb_positions[0] - 0.5 * self.L
        dist_lc = torch.minimum(dist_lc_R1, dist_lc_R2)

        # distance from upper curb
        dist_uc_R1 = self.curb_positions[1] - state[..., 1] - 0.5 * self.L
        dist_uc_R2 = self.curb_positions[1] - state[..., 6] - 0.5 * self.L
        dist_uc = torch.minimum(dist_uc_R1, dist_uc_R2)

        # # distance from the stranded car
        # stranded_car_pos = torch.tensor(
        #     self.stranded_car_pos, device=state.device)
        # dist_stranded_R1 = torch.norm(
        #     state[..., 0:2] - stranded_car_pos, dim=-1) - self.L
        # dist_stranded_R2 = torch.norm(
        #     state[..., 5:7] - stranded_car_pos, dim=-1) - self.L
        # dist_stranded = torch.minimum(dist_stranded_R1, dist_stranded_R2)

        # distance between the vehicles themselves
        dist_R1R2 = torch.norm(
            state[..., 0:2] - state[..., 5:7], dim=-1) - self.L

        # return self.avoid_fn_weight * torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2)
        return self.avoid_fn_weight * torch.min(torch.min(dist_lc, dist_uc), dist_R1R2)

    def boundary_fn(self, state):
        if self.avoid_only:
            return self.avoid_fn(state)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        if self.avoid_only:
            return torch.min(self.avoid_fn(state_traj), dim=-1).values
        else:
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, dvds):
        optimal_control = self.optimal_control(state, dvds)
        return state[..., 3] * torch.cos(state[..., 2]) * dvds[..., 0] + \
            state[..., 3] * torch.sin(state[..., 2]) * dvds[..., 1] + \
            state[..., 3] * torch.tan(state[..., 4]) * dvds[..., 2] / self.L + \
            optimal_control[..., 0] * dvds[..., 3] + \
            optimal_control[..., 1] * dvds[..., 4] + \
            state[..., 8] * torch.cos(state[..., 7]) * dvds[..., 5] + \
            state[..., 8] * torch.sin(state[..., 7]) * dvds[..., 6] + \
            state[..., 8] * torch.tan(state[..., 9]) * dvds[..., 7] / self.L + \
            optimal_control[..., 2] * dvds[..., 8] + \
            optimal_control[..., 3] * dvds[..., 9]

    def optimal_control(self, state, dvds):
        a1_min = self.aMin * (state[..., 3] > self.vMin)
        a1_max = self.aMax * (state[..., 3] < self.vMax)
        psi1_min = self.psiMin * (state[..., 4] > self.phiMin)
        psi1_max = self.psiMax * (state[..., 4] < self.phiMax)
        a2_min = self.aMin * (state[..., 8] > self.vMin)
        a2_max = self.aMax * (state[..., 8] < self.vMax)
        psi2_min = self.psiMin * (state[..., 9] > self.phiMin)
        psi2_max = self.psiMax * (state[..., 9] < self.phiMax)

        if self.avoid_only:
            a1 = torch.where(dvds[..., 3] < 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] < 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] < 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] < 0, psi2_min, psi2_max)

        else:
            a1 = torch.where(dvds[..., 3] > 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] > 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] > 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] > 0, psi2_min, psi2_max)

        return torch.cat((a1[..., None], psi1[..., None], a2[..., None], psi2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                -5, -5, 0.0, 6.5, 0.0,
                3, 2, -math.pi, 0.0, 0.0
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$', r'$\theta_1$', r'$v_1$', r'$\phi_1$',
                r'$x_2$', r'$y_2$', r'$\theta_2$', r'$v_2$', r'$\phi_2$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class InvertedPendulum(Dynamics):
    def __init__(self):
        self.g = 9.81
        self.m = 1.0
        self.l = 1.0
        self.b = 0.0
        self.u_max = 2.0
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=2, input_dim=3, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0],  # theta, omega
            state_var=[math.pi/2, 5],    # theta, omega
            value_mean=0,
            value_var=math.pi/2,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-math.pi/2, math.pi/2],                        # theta
            [-5, 5],                    # omega
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # dynamics:
    #     theta_dot=omega;
    #     omega_dot=(-b*omega + m*g*l*sin(theta)/2 ) / (m*l^2/3)-u/ (m*l^2/3);

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 1]
        dsdt[..., 1] = (-self.b*state[..., 1] + self.m*self.g*self.l*torch.sin(state[..., 0])/2) \
            / (self.m*self.l ** 2/3)-control[..., 0] / (self.m*self.l ** 2/3)
        return dsdt

    def boundary_fn(self, state):
        return torch.minimum(math.pi/4-torch.abs(state[..., 0]), 4-torch.abs(state[..., 1]))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return state[..., 1]*dvds[..., 0]+dvds[..., 1]*(-self.b*state[..., 1] + self.m*self.g*self.l*torch.sin(state[..., 0])/2) \
            / (self.m*self.l ** 2/3) + self.u_max*torch.abs(dvds[..., 1]) / (self.m*self.l ** 2/3)

    def optimal_control(self, state, dvds):
        return -self.u_max * torch.sign(dvds[..., 1])[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0],
            'state_labels': ['theta', 'omega'],
            'x_axis_idx': 1,
            'y_axis_idx': 0,
            'z_axis_idx': -1,
        }


class VertDrone2D(Dynamics):
    def __init__(self, gravity: float, input_magnitude_max: float, K: float):
        self.gravity = gravity                             # g
        self.input_magnitude_max = input_magnitude_max     # u_max
        self.K = K
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=2, input_dim=3, control_dim=1, disturbance_dim=0,
            state_mean=[0, 1.5],  # v, z
            state_var=[4, 2],    # v, z
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-4, 4],                        # v
            [-0.5, 3.5],                    # z
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.K*control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return torch.minimum(state[..., 1], 3.0-state[..., 1])

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return self.K*torch.abs(dvds[..., 0]*self.input_magnitude_max) \
            - dvds[..., 0]*self.gravity \
            + dvds[..., 1]*state[..., 0]

    def optimal_control(self, state, dvds):
        return self.input_magnitude_max * torch.sign(dvds[..., [0]])

    def optimal_disturbance(self, state, dvds):
        return torch.zeros_like(dvds[..., [0]])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5],
            'state_labels': ['v', 'z'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': -1,
        }


'''VertDrone2D with switched states'''


class VertDrone(Dynamics):
    def __init__(self, gravity: float, input_magnitude_max: float, K: float):
        self.gravity = gravity                             # g
        self.input_magnitude_max = input_magnitude_max     # u_max
        self.K = K
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=2, input_dim=3, control_dim=1, disturbance_dim=0,
            state_mean=[1.5, 0],  # z, v
            state_var=[2, 4],    # z, v
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='reg',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-0.5, 3.5],                        # z
            [-4., 4.0],                    # v
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 1] = self.K*control[..., 0] - self.gravity
        dsdt[..., 0] = state[..., 1]
        return dsdt

    def boundary_fn(self, state):
        # return -torch.abs(state[..., 0] - 1.5) + 1.5
        return torch.minimum(state[..., 0], 3.0-state[..., 0])

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return self.K*torch.abs(dvds[..., 1]*self.input_magnitude_max) \
            - dvds[..., 1]*self.gravity \
            + dvds[..., 0]*state[..., 1]

    def optimal_control(self, state, dvds):
        return self.input_magnitude_max * torch.sign(dvds[..., 0])

    def optimal_disturbance(self, state, dvds):
        return torch.zeros_like(dvds[..., 1])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5],
            'state_labels': ['z', 'v'],
            'x_axis_idx': 1,
            'y_axis_idx': 0,
            'z_axis_idx': -1,
        }


class QuadrotorReachAvoid(Dynamics):
    def __init__(self, collisionR: float, collective_thrust_max: float):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4

        self.collisionR = collisionR

        super().__init__(
            loss_type='brat_hjivi', set_mode='reach',
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=4,
            state_mean=[0 for i in range(13)],
            state_var=[3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            value_mean=(math.sqrt(3**2 + 3**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3**2 + 3**2),
            value_normto=0.02,
            deepReach_model='reg',
            exact_factor=1,
        )

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]+disturbance[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]+disturbance[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]+disturbance[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]+disturbance[..., 3]) * 1.0

        return dsdt

    def reach_fn(self, state):
        states_ = state*1.0
        states_[..., 3] -= 1.0
        states_[..., 3:7] *= 5
        states_[..., 7:] *= 0.5
        return torch.norm(states_, dim=-1) - 0.5

    def avoid_fn(self, state):
        '''for cylinder with full body collision'''
        # create normal vector
        v = torch.zeros_like(state[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder

        px = state[..., 0]-1.5
        py = state[..., 1]*1.0

        # get full body distance
        dist = torch.norm(
            torch.cat((px[..., None], py[..., None]), dim=-1), dim=-1)
        # return dist- self.collisionR
        dist -= torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))

        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):

        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0

        c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
        c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
        c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
              torch.pow(qy, 2)) * self.CT / self.m

        # Compute the hamiltonian for the quadrotor
        ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
        ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
        ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
        ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
        ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
        ham += dvds[..., 9] * self.Gz
        ham += -dvds[..., 10] * 5 * wy * wz / \
            9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

        ham -= (torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                          c2 + dvds[..., 9] * c3) * (self.collective_thrust_max))

        ham -= (torch.abs(dvds[..., 10]) * (self.dwx_max) + torch.abs(
            dvds[..., 11]) * (self.dwy_max) + torch.abs(dvds[..., 12]) * (self.dwz_max))

        return ham

    def optimal_control(self, state, dvds):

        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0

        c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
        c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
        c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
              torch.pow(qy, 2)) * self.CT / self.m

        u1 = -self.collective_thrust_max * \
            torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                       c2 + dvds[..., 9] * c3)
        u2 = -self.dwx_max * torch.sign(dvds[..., 10])
        u3 = -self.dwy_max * torch.sign(dvds[..., 11])
        u4 = -self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0.0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }


'''Simplified quadrotor dynamics with a cylinder obstacle'''


class Quadrotor(Dynamics):
    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR

        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=4,
            state_mean=[0 for i in range(13)],
            state_var=[3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            value_mean=(math.sqrt(3**2 + 3**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3**2 + 3**2),
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=0.1,
        )

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]+disturbance[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]+disturbance[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]+disturbance[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]+disturbance[..., 3]) * 1.0

        return dsdt

    def boundary_fn(self, state):
        '''for cylinder with full body collision'''
        # create normal vector
        v = torch.zeros_like(state[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state[..., 0]
        py = state[..., 1]

        # get full body distance
        dist = torch.norm(state[..., :2], dim=-1)
        # return dist- self.collisionR
        dist -= torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))

        angular_v_bound = torch.minimum(torch.minimum(4-torch.abs(state[..., 10]), 4-torch.abs(state[..., 11])),
                                        4-torch.abs(state[..., 12]))

        # dist_ground_ceiling = 2-torch.abs(state[..., 2])
        # return torch.minimum(torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR, dist_ground_ceiling)
        return torch.minimum(torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR, angular_v_bound)

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * (self.collective_thrust_max-self.dist_f)

            ham += torch.abs(dvds[..., 10]) * (self.dwx_max-self.dist_dwx_max) + torch.abs(
                dvds[..., 11]) * (self.dwy_max-self.dist_dwy_max) + torch.abs(dvds[..., 12]) * (self.dwz_max-self.dist_dwz_max)

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = -self.dist_f * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = -self.dist_dwx_max * torch.sign(dvds[..., 10])
            u3 = -self.dist_dwy_max * torch.sign(dvds[..., 11])
            u4 = -self.dist_dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }


'''Simplified quadrotor dynamics using Euler-angle representation and ego frame with a cylinder obstacle'''


class QuadrotorEuler(Dynamics):
    def __init__(self, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 20
        self.dwy_max = 20
        self.dwz_max = 8

        # self.collisionR = collisionR
        self.cylinder1_info = [0, 0, 0.4]  # x y r
        self.cylinder2_info = [0, 0.5, 0.4]  # x y r
        self.sphere1_info = [0, -1, 0, 0.5]  # x y z r
        self.ground = -2
        self.ceiling = 2

        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=12, input_dim=13, control_dim=4, disturbance_dim=0,
            state_mean=[0 for i in range(12)],
            state_var=[3, 3, 3, math.pi, math.pi, math.pi, 5, 5, 5, 15, 15, 5],
            value_mean=(math.sqrt(3**2 + 3**2)) / 2,
            value_var=math.sqrt(3**2 + 3**2),
            value_normto=0.02,
            deepReach_model='reg'
        )

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [-3.2, 3.2],
            [-3.2, 3.2],
            [-3.2, 3.2],
            [-1, 1],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-15, 15],
            [-15, 15],
            [-5, 5],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 3] = (
            wrapped_state[..., 3] + math.pi) % (2 * math.pi) - math.pi

        wrapped_state[..., 4] = (
            wrapped_state[..., 4] + math.pi) % (2 * math.pi) - math.pi

        wrapped_state[..., 5] = (
            wrapped_state[..., 5] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        phi = state[..., 3] * 1.0  # x
        theta = state[..., 4] * 1.0  # y
        psi = state[..., 5] * 1.0  # z
        vx = state[..., 6] * 1.0
        vy = state[..., 7] * 1.0
        vz = state[..., 8] * 1.0
        wx = state[..., 9] * 1.0
        wy = state[..., 10] * 1.0
        wz = state[..., 11] * 1.0
        f = control[..., 0] * 1.0
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = torch.cos(theta)*torch.cos(psi)*vx + (torch.sin(theta)*torch.sin(phi)*torch.cos(psi) - torch.cos(
            phi)*torch.sin(psi))*vy + (torch.sin(theta)*torch.cos(phi)*torch.cos(psi) + torch.sin(phi)*torch.sin(psi))*vz
        dsdt[..., 1] = -(torch.cos(theta)*torch.sin(psi)*vx + (torch.sin(theta)*torch.sin(phi)*torch.sin(psi) + torch.cos(
            phi)*torch.cos(psi))*vy + (torch.sin(theta)*torch.cos(phi)*torch.sin(psi) - torch.sin(phi)*torch.cos(psi))*vz)
        dsdt[..., 2] = torch.sin(theta)*vx - torch.cos(theta) * \
            torch.sin(phi)*vy - torch.cos(theta)*torch.cos(phi)*vz
        dsdt[..., 3] = wx+torch.sin(phi)*torch.tan(theta) * \
            wy+torch.cos(phi)*torch.tan(theta)*wz
        dsdt[..., 4] = torch.cos(phi)*wy-torch.sin(phi)*wz
        dsdt[..., 5] = (torch.sin(phi)*wy+torch.cos(phi)*wz)/torch.cos(theta)
        dsdt[..., 6] = wz*vy-wy*vz
        dsdt[..., 7] = wx*vz-wz*vx
        dsdt[..., 8] = wy*vx-wx*vy + f/self.m
        dsdt[..., 9] = control[..., 1] * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 10] = control[..., 2] * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 11] = control[..., 3] * 1.0

        # here we add the components of G.. (inverse of ZYX euler angles)
        dsdt[..., 6] += torch.sin(-theta)*self.Gz
        dsdt[..., 7] += -torch.cos(-theta)*torch.sin(-phi)*self.Gz
        dsdt[..., 8] += torch.cos(-phi)*torch.cos(-theta)*self.Gz
        return dsdt

    def boundary_fn(self, state):
        dist_cylinder_1 = self.dist_from_cylinder(state, self.cylinder1_info)
        # dist_cylinder_2 = self.dist_from_cylinder(state, self.cylinder2_info)

        # p_sphere1 = state[..., :3]
        # for i in range(3):
        #     p_sphere1[..., i] -= self.sphere1_info[i]
        # dist_sphere1 = torch.norm(p_sphere1, dim=-1) - self.sphere1_info[3]

        # dist_ground = state[..., 2]-self.ground
        # dist_ceiling = -state[..., 2]+self.ceiling
        # dist_ground_ceiling = 2-torch.abs(state[..., 2])
        # return torch.minimum(torch.minimum(torch.minimum(dist_cylinder_1, dist_cylinder_2), dist_sphere1), dist_ground_ceiling)
        # return torch.minimum(torch.minimum(torch.minimum(dist_cylinder_1, dist_cylinder_2), dist_sphere1), dist_ground)

        return dist_cylinder_1

    def dist_from_cylinder(self, state, ceilinder_info):
        '''for cylinder with full body collision'''
        phi = state[..., 3] * 1.0  # x
        theta = state[..., 4] * 1.0  # y
        psi = state[..., 5] * 1.0  # z

        vx = (torch.sin(theta)*torch.cos(phi) *
              torch.cos(psi) + torch.sin(phi)*torch.sin(psi))*-1
        vy = (torch.sin(theta)*torch.cos(phi) *
              torch.sin(psi) - torch.sin(phi)*torch.cos(psi))*-1
        vz = - torch.cos(theta)*torch.cos(psi)

        # compute vector from center of quadrotor to the center of cylinder
        p = state[..., :2]*1.0
        p[..., 0] -= ceilinder_info[0]
        p[..., 1] -= ceilinder_info[1]
        px = p[..., 0]
        py = p[..., 1]
        # get full body distance
        dist = torch.norm(p[..., :2], dim=-1)
        # return dist- collisionR
        dist -= torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        return torch.maximum(dist, torch.zeros_like(dist)) - ceilinder_info[2]

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            control_ = self.optimal_control(state, dvds)
            disturbance_ = self.optimal_disturbance(state, dvds)
            dsdt_ = self.dsdt(state, control_, disturbance_)
            ham = 0.0
            for i in range(self.state_dim):
                ham += dsdt_[..., i]*dvds[..., i]

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            u1 = self.collective_thrust_max * torch.sign(dvds[..., 8])
            u2 = self.dwx_max * torch.sign(dvds[..., 9])
            u3 = self.dwy_max * torch.sign(dvds[..., 10])
            u4 = self.dwz_max * torch.sign(dvds[..., 11])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }


'''Parameter (disturbance) conditioned quadrotor dynamics'''


class QuadrotorDisturbanceConditioned(Dynamics):
    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 12
        self.dwy_max = 12
        self.dwz_max = 5
        # self.dist_dwx_max = 2
        # self.dist_dwy_max = 2
        # self.dist_dwz_max = 1
        # self.dist_f = 3

        self.collisionR = collisionR

        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=14, input_dim=15, control_dim=4, disturbance_dim=4,
            state_mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3],
            state_var=[3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 15, 15, 5, 0.3],
            value_mean=(math.sqrt(3**2 + 3**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3**2 + 3**2),
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-15, 15],
            [-15, 15],
            [-5, 5],
            [0, 0.6],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]+disturbance[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]+disturbance[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]+disturbance[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]+disturbance[..., 3]) * 1.0

        return dsdt

    def boundary_fn(self, state):
        '''for cylinder with full body collision'''
        # create normal vector
        v = torch.zeros_like(state[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state[..., 0]
        py = state[..., 1]

        # get full body distance
        dist = torch.norm(state[..., :2], dim=-1)
        # return dist- self.collisionR
        dist -= torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))

        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max * (1-state[..., 13])

            ham += torch.abs(dvds[..., 10]) * self.dwx_max*(1-state[..., 13]) + torch.abs(
                dvds[..., 11]) * self.dwy_max*(1-state[..., 13]) + torch.abs(dvds[..., 12]) * self.dwz_max*(1-state[..., 13])

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = -self.collective_thrust_max * state[..., 13] * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = -self.dwx_max*state[..., 13] * torch.sign(dvds[..., 10])
            u3 = -self.dwy_max*state[..., 13] * torch.sign(dvds[..., 11])
            u4 = -self.dwz_max*state[..., 13] * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'disturbance'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }

# class RocketLanding(Dynamics):
#     def __init__(self):
#         self.set_mode = 'reach'
#         super().__init__(
#             loss_type='brt_hjivi', set_mode=self.set_mode, state_dim=6, input_dim=7, control_dim=2, disturbance_dim=0,
#             state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
#             state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
#             value_mean=0.0,
#             value_var=1.0,
#             value_normto=0.02,
#             deepReach_model='reg',
#             exact_factor=0.9999999,
#         )

#     def state_test_range(self):
#         return [
#             [-150, 150],
#             [10, 150],
#             [-math.pi, math.pi],
#             [-200, 200],
#             [-200, 200],
#             [-10, 10],
#         ]

#     def equivalent_wrapped_state(self, state):
#         wrapped_state = torch.clone(state)
#         wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
#         return wrapped_state

#     # \dot x = v_x
#     # \dot y = v_y
#     # \dot th = w
#     # \dot v_x = u1 * cos(th) - u2 sin(th)
#     # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
#     # \dot w = 0.3 * u1
#     def dsdt(self, state, control, disturbance):
#         dsdt = torch.zeros_like(state)
#         dsdt[..., 0] = state[..., 3]
#         dsdt[..., 1] = state[..., 4]
#         dsdt[..., 2] = state[..., 5]
#         dsdt[..., 3] = control[..., 0]*torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
#         dsdt[..., 4] = control[..., 0]*torch.sin(state[..., 2]) + control[..., 1]*torch.cos(state[..., 2]) - 9.81
#         dsdt[..., 5] = 0.3*control[..., 0]
#         return dsdt

#     def boundary_fn(self, state):
#         # Only target set in the yz direction
#         # Target set position in y direction
#         dist_y = torch.abs(state[..., 0]) - 20.0 #[-20, 150] range

#         # Target set position in z direction
#         dist_z = state[..., 1] - 20.0  #[-10, 130] range

#         # First compute the l(x) as you normally would but then normalize it later.
#         lx = torch.max(dist_y, dist_z)
#         return torch.where((lx >= 0), lx/150.0, lx/10.0)

#     def sample_target_state(self, num_samples):
#         raise NotImplementedError

#     def cost_fn(self, state_traj):
#         return torch.min(self.boundary_fn(state_traj), dim=-1).values

#     def hamiltonian(self, state, dvds):
#         ## Compute the Hamiltonian
#         # Control Hamiltonian
#         u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
#         u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
#         ham_ctrl = -250 * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)

#         # Constant Hamiltonian
#         ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
#                       dvds[..., 2] * state[..., 5]  - dvds[..., 4] * 9.81

#         # Compute the Hamiltonian
#         ham_vehicle = ham_ctrl + ham_constant

#         return ham_vehicle

#     def optimal_control(self, state, dvds):
#         u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
#         u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
#         opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
#         return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

#     def optimal_disturbance(self, state, dvds):
#         return 0

#     def plot_config(self):
#         return {
#             'state_slices': [-100, 120, 0, 150, -5, 0.0],
#             'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
#             'x_axis_idx': 0,
#             'y_axis_idx': 1,
#             'z_axis_idx': 4,
#         }


class RocketLanding(Dynamics):
    def __init__(self):
        super().__init__(
            loss_type='brt_hjivi', set_mode='reach',
            state_dim=6, input_dim=8, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            deepReach_model='diff',
            exact_factor=1,
        )

    # convert model input to real coord
    def input_to_coord(self, input):
        input = input[..., :-1]
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)
                          ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)
                          ) / self.state_var.to(device=coord.device)
        input = torch.cat(
            (input, torch.zeros((*input.shape[:-1], 1), device=input.device)), dim=-1)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == "diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(
            dim=-1), input)[0].squeeze(dim=-2)[..., :-1]

        if self.deepReach_model == "diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto /
                    self.state_var.to(device=dodi.device)) * dodi[..., 1:]

        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    def state_test_range(self):
        return [
            [-150, 150],
            [10, 150],
            [-math.pi, math.pi],
            [-200, 200],
            [-200, 200],
            [-10, 10],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # \dot x = v_x
    # \dot y = v_y
    # \dot th = w
    # \dot v_x = u1 * cos(th) - u2 sin(th)
    # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
    # \dot w = 0.3 * u1
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]
        dsdt[..., 1] = state[..., 4]
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = control[..., 0] * \
            torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
        dsdt[..., 4] = control[..., 0] * \
            torch.sin(state[..., 2]) + control[..., 1] * \
            torch.cos(state[..., 2]) - 9.81
        dsdt[..., 5] = 0.3*control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        # Only target set in the yz direction
        # Target set position in y direction
        # [-20, 150] boundary_fn range
        dist_y = torch.abs(state[..., 0]) - 20.0

        # Target set position in z direction
        dist_z = state[..., 1] - 20.0  # [-10, 130] boundary_fn range

        # First compute the l(x) as you normally would but then normalize it later.
        lx = torch.max(dist_y, dist_z)
        return torch.where((lx >= 0), lx/150.0, lx/10.0)

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-20, 20]  # y in [-20, 20]
        target_state_range[1] = [10, 20]  # z in [10, 20]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        # Control Hamiltonian
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(
            state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = - \
            dvds[..., 3] * torch.sin(state[..., 2]) + \
            dvds[..., 4] * torch.cos(state[..., 2])
        ham_ctrl = -250.0 * \
            torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)
        # Constant Hamiltonian
        ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
            dvds[..., 2] * state[..., 5] - dvds[..., 4] * 9.81
        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant
        return ham_vehicle

    def optimal_control(self, state, dvds):
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(
            state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = - \
            dvds[..., 3] * torch.sin(state[..., 2]) + \
            dvds[..., 4] * torch.cos(state[..., 2])
        opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
        return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [-100, 120, 0, 150, -5, 0.0],
            'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }


class MultiVehicleCollision(Dynamics):
    def __init__(self):
        self.set_mode = 'avoid'
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        self.alpha_time = 1.0
        super().__init__(
            loss_type='brt_hjivi', set_mode=self.set_mode, state_dim=9, input_dim=10, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0,
                0, 0,
                0, 0, 0,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor *
                math.pi, self.angle_alpha_factor*math.pi,
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
        return dsdt

    def boundary_fn(self, state):
        boundary_values = torch.norm(
            state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(
                state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(
                boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] -
                                                     state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(
                    boundary_values, boundary_values_current)
        return boundary_values

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

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
                self.angle_alpha_factor*state[..., theta_index]) * dvds[..., ycostate_index]) + self.omega_max * torch.abs(dvds[..., thetacostate_index])
            ham = ham + ham_local

        # Effect of time factor
        ham = ham * self.alpha_time
        return ham

    def optimal_control(self, state, dvds):
        dvds[..., 6] = -dvds[..., 6]
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                0, 0,
                -0.4, 0,
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }


class RimlessWheel(Dynamics):
    def __init__(self):
        self.E=1.132
        #self.alpha=math.pi/8
        self.alpha=0.4
        self.gamma=0.2
        self.eps=0.03
        super().__init__(
            loss_type='brt_hjivi', set_mode='reach',
            state_dim=2, input_dim=3, control_dim=0, disturbance_dim=0,
            state_mean=[0.2, 0.35],  # theta, omega
            state_var=[0.4, 0.95],    # theta, omega
            value_mean=0,
            value_var=1.2,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-0.2, 0.6],                        # theta
            [-0.6, 1.3],                    # omega
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # dynamics:
    #     theta_dot=omega;
    #     omega_dot=(-b*omega + m*g*l*sin(theta)/2 ) / (m*l^2/3)-u/ (m*l^2/3);

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 1]
        dsdt[..., 1] = torch.sin(state[..., 0])
        return dsdt

    def boundary_fn(self, state, verbose=False): # everything pre-switch state is converted to post-switch first
        # idx=(torch.abs(state[..., 0]-0.6)<1e-5).flatten() 
        # if verbose:
        #     print("state in boundary_fn:",state,idx)
        # state[...,idx, 0]=-0.2
        # state[...,idx, 1]=state[...,idx, 1]*math.cos(self.alpha*2)
        # if verbose:
        #     print("post switch state in boundary_fn:",state)
        lx=torch.abs(torch.sqrt(2*(self.E-torch.cos(state[..., 0])))-state[..., 1])-self.eps
        # lx[lx<0]*=20.0
        return lx

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    # def sample_switch_state(self, num_samples):
    #     switch_state_range = torch.tensor(self.state_test_range())
    #     switch_samples=torch.rand(num_samples, self.state_dim).repeat(2,1,1)*(switch_state_range[:, 1] - switch_state_range[:, 0])+ switch_state_range[:, 0]
    #     switch_samples[0,...,0]=switch_state_range[0][1]
    #     switch_samples[1,...,0]=switch_state_range[0][0]
    #     switch_samples[1,...,1]= switch_samples[0,...,1]*math.cos(self.alpha*2)
    #     return switch_samples
    
    # def sample_switch_state(self, num_samples):
    #     switch_state_range = torch.tensor(self.state_test_range())
    #     # switch_samples=torch.rand(num_samples, self.state_dim).repeat(2,1,1)*(switch_state_range[:, 1] - switch_state_range[:, 0])+ switch_state_range[:, 0]

    #     switch_samples=torch.zeros(2,num_samples, self.state_dim)
    #     num_forward=int(num_samples/2.0) 
    #     # samples with positive qdot
    #     switch_samples[0,:num_forward,0]=switch_state_range[0][1]
    #     switch_samples[1,:num_forward,0]=switch_state_range[0][0]
    #     switch_samples[0,:num_forward,1]=torch.rand(num_forward)*switch_state_range[1][1]
    #     switch_samples[1,:num_forward,1]= switch_samples[0,:num_forward,1]*math.cos(self.alpha*2)

    #     # samples with negative qdot
    #     switch_samples[0,num_forward:,0]=switch_state_range[0][0] # -0.2
    #     switch_samples[1,num_forward:,0]=switch_state_range[0][1] # 0.6 
    #     switch_samples[0,num_forward:,1]=torch.rand(num_samples-num_forward)*switch_state_range[1][0]
    #     switch_samples[1,num_forward:,1]= switch_samples[0,num_forward:,1]*math.cos(self.alpha*2)
    #     return switch_samples

    def sample_switch_state(self, num_samples):
        switch_state_range = torch.tensor(self.state_test_range())
        # switch_samples=torch.rand(num_samples, self.state_dim).repeat(2,1,1)*(switch_state_range[:, 1] - switch_state_range[:, 0])+ switch_state_range[:, 0]

        switch_samples=torch.zeros(2,num_samples, self.state_dim)
        # samples with positive qdot
        switch_samples[0,:,0]=switch_state_range[0][1]
        switch_samples[1,:,0]=switch_state_range[0][0]
        switch_samples[0,:,1]=torch.rand(num_samples)*switch_state_range[1][1]
        switch_samples[1,:,1]= switch_samples[0,:,1]*math.cos(self.alpha*2)


        return switch_samples
        
    def find_switching_surface_mask(self, states):
        states = self.input_to_coord(states)
        switching_angle = self.gamma + self.alpha
        switching_surface_mask = (states[..., 1] == switching_angle) & \
                                 (states[..., 2] > 0.0)
        return switching_surface_mask


    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return state[..., 1]*dvds[..., 0]+dvds[..., 1]*torch.sin(state[..., 0])

    def optimal_control(self, state, dvds):
        return 0

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0],
            'state_labels': ['theta', 'omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': -1,
        }

class CompassWalker(Dynamics):
    def __init__(self):
        mat = scipy.io.loadmat('gait_from_clf_qp_ral.mat')
        self.gait=mat['xRec'][:, [2, 3, 6, 7]]

        self.scale=torch.tensor([1,1,0.1,0.1])
        self.gait_scaled=torch.tensor(self.gait)*self.scale

        self.distance_threshold=0.05

        self.lL = 1.0
        self.mL = 1.0
        self.g = 9.81
        self.mH = 1.0
        self.umax=4.0

        self.BCNN = BCNetwork()
        self.BCNN.load_state_dict(
                        torch.load('BC_estimator.pth', map_location='cpu'))
        self.BCNN.eval()
        self.BCNN.cuda()
        for param in self.BCNN.parameters():
            param.requires_grad = False

        super().__init__(
            loss_type='brt_hjivi', set_mode='reach',
            state_dim=4, input_dim=5, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 0.0, 0.0, 0.0],  # q1, q2, omega1, omega2
            state_var=[0.52, 1.04, 4.0, 8.0],  # q1, q2, omega1, omega2
            value_mean=0.0,
            value_var=3.0,
            value_normto=0.02,
            deepReach_model='exact',
            exact_factor=1.0,
        )

    def state_test_range(self):
        return [
            [-0.52, 0.52],                    # q1
            [-1.04, 1.04],                    # q2
            [-4.0, 4.0],                      # q3
            [-8.0, 8.0],                      # q4
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # dynamics:
    #     theta_dot=omega;
    #     omega_dot=(-b*omega + m*g*l*sin(theta)/2 ) / (m*l^2/3)-u/ (m*l^2/3);
    def get_fvec(self,state_condensed):
        q1 = state_condensed[...,0]*1.0
        q2 = state_condensed[...,1]*1.0
        dq1 = state_condensed[...,2]*1.0
        dq2 = state_condensed[...,3]*1.0

        t2 = torch.cos(q1)
        t3 = torch.sin(q1)
        t4 = torch.sin(q2)
        t5 = dq1**2
        t6 = dq2**2
        t7 = self.mH*4.0
        t8 = self.mL*3.0
        t9 = q2*2.0
        t12 = 1.0/self.lL
        t10 = torch.cos(t9)
        t11 = torch.sin(t9)
        t13 = self.g*t3*t7
        t14 = self.g*self.mL*t3*4.0
        t17 = dq1*dq2*self.lL*self.mL*t4*4.0
        t18 = self.lL*self.mL*t4*t6*2.0
        t15 = self.mL*t10*2.0
        t16 = -t15
        t19 = t7+t8+t16
        t20 = 1.0/t19
        fvec=torch.zeros_like(state_condensed)
        fvec[...,0]=dq1
        fvec[...,1]=dq2
        fvec[...,2]=-t12*t20*(-t14+t17+t18-self.g*self.mH*t3*4.0+self.g*self.mL*torch.sin(q1+t9)*2.0+self.lL*self.mL*t4*t5*2.0-self.lL*self.mL*t5*t11*2.0)
        fvec[...,3]=-t12*t20*(t13+t14-t17-t18+self.g*self.mH*t2*t4*8.0+self.g*self.mL*t2*t4*10.0-self.g*self.mL*t2*t11*2.0-self.g*self.mL*t3*t10*2.0-self.lL*self.mH*t4*t5*8.0\
                              -self.lL*self.mL*t4*t5*12.0+self.lL*self.mL*t5*t11*4.0+self.lL*self.mL*t6*t11*2.0-self.g*self.mL*t3*torch.cos(q2)*2.0+dq1*dq2*self.lL*self.mL*t11*4.0)
        return fvec.to(state_condensed)

    def get_gvec(self,state_condensed):

        q2 = state_condensed[...,1]*1.0

        t2 = torch.cos(q2)
        t3 = self.mH*4.0
        t4 = self.mL*5.0
        t6 = 1.0/self.lL**2
        t5 = t2**2
        t7 = self.mL*t5*4.0
        t8 = -t7
        t9 = t3+t4+t8
        t10 = 1.0/t9
        gvec = torch.zeros(1, state_condensed.shape[1],4)
        gvec[...,2] = t6*t10*(t2*8.0-4.0)
        gvec[...,3] = (t6*t10*(self.mH*16.0+self.mL*24.0-self.mL*t2*16.0))/self.mL
        return gvec.to(state_condensed)
    
    def dsdt(self, state, control, disturbance):
        dx = self.get_fvec(state.clone()) + self.get_gvec(state.clone()) * torch.cat([torch.zeros_like(control),control],dim=-1)  
        return dx

    def boundary_fn(self, state):
        # computed using NN
        device=state.device
        lx=self.BCNN(state.cuda()).to(device)
        return lx
    
    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def reset_map_condensed(self, xs_pre):
        xs_post=torch.zeros_like(xs_pre).to(xs_pre)
        xs_post[...,0]=xs_pre[...,0]+xs_pre[...,1]
        xs_post[...,1]=-xs_pre[...,1]
        dq_post_impact=self.dq_post_impact_condensed(xs_pre.clone())
        xs_post[...,2] = dq_post_impact[...,0]+dq_post_impact[...,1]
        xs_post[...,3]= -dq_post_impact[...,1]
        return xs_post 

    def dq_post_impact_condensed(self, xs):
        q1 = xs[...,0]
        q2 = xs[...,1]
        dq1 = xs[...,2]
        dq2 = xs[...,3]

        dx = -torch.cos(q1) * dq1
        dy = -torch.sin(q1) * dq1

        t2 = torch.cos(q1)
        t3 = torch.cos(q2)
        t4 = torch.sin(q1)
        t5 = q1+q2
        t7 = q2*2.0
        t14 = -q2
        t9 = torch.cos(t7)
        t12 = torch.cos(t5)
        t13 = torch.sin(t5)
        t15 = dq1*t3*2.0
        t16 = dq2*t3*2.0
        t17 = q2+t5
        t21 = dx*t2*8.0
        t22 = q1+t14
        t23 = dy*t4*8.0
        t18 = torch.cos(t17)
        t19 = torch.sin(t17)
        t20 = t9*2.0
        t25 = -t21
        t26 = -t23
        t27 = dq1*t20
        t30 = t20-7.0
        t31 = dx*t18*10.0
        t32 = dy*t19*10.0
        t33 = 1.0/t30
        dq_post_impact = torch.cat(((t33*(dq1*-7.0+t15+t16+t25+t26+t27+t31+t32))[...,None],
            (-t33*(dq1*-8.0-dq2+t15+t16+t25+t26+t27+t31+t32-dx*t12*8.0-dy*t13*8.0+dx*torch.cos(t22)*2.0+dy*torch.sin(t22)*2.0))[...,None]),dim=-1)
        return dq_post_impact
    
    def find_switching_surface_mask(self, states):
        # Given the time and (normalized) state coordinates, find the
        # coordinates that are on the switching surface.
        # check if x \in S = {x | 2*q1 + q2 == 0, q1>q1_th, 2*dq1 + dq2 > 0}
        states = self.input_to_coord(states)
        switching_surface_mask = (0.05 <= states[..., 1]) & \
                                 (torch.abs(2 * states[..., 1] + states[..., 2]) < 1e-3) & \
                                 (states[..., 4] >= -2 * states[..., 3])
        return switching_surface_mask

    def sample_switch_state(self, num_samples):
        switch_state_range = torch.tensor(self.state_test_range())
        xs_pre_switch=torch.rand(1,num_samples,4)
        xs_pre_switch[...,0]=xs_pre_switch[...,0]*switch_state_range[0,0]
        xs_pre_switch[...,1]=-2*xs_pre_switch[...,0]
        xs_pre_switch[...,2]=xs_pre_switch[...,2]*switch_state_range[2,1]*2+switch_state_range[2,0]
        # randomly sample dq2 from [-8, -2dq1]
        xs_pre_switch[...,3]= switch_state_range[3,0] + (-2*xs_pre_switch[...,2]+switch_state_range[3,1])*xs_pre_switch[...,3]
        xs_post_switch=self.reset_map_condensed(xs_pre_switch)



        return torch.cat([xs_pre_switch,xs_post_switch],dim=0)


    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        gvec=self.get_gvec(state.clone())
        ham= torch.sum(self.get_fvec(state.clone())*dvds,dim=-1) -  torch.abs(gvec[...,2]*dvds[...,2]*self.umax) -  torch.abs(gvec[...,3]*dvds[...,3]*self.umax)
        return ham

    def optimal_control(self, state, dvds):
        return torch.sign(self.get_gvec(state)[...,2:])*self.umax

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0],
            'state_labels': ['q1', 'q2', 'dq1', 'dq2'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': -1,
        }

class Dubins5D(Dynamics):
    def __init__(self):
        self.w_max = 2
        self.a_max = 2
        self.L = 1
        self.set_mode = 'avoid'

        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=5, input_dim=6, control_dim=2, disturbance_dim=5,
            state_mean=[0, 0, 2.5, 0, 0],
            state_var=[10, 10, 2.5, math.pi, math.pi / 6],
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
            [0, 5],
            [-math.pi, math.pi],
            [-math.pi/6, math.pi/6]
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
    # Dynamics of the Dubins5D Car:
    #         \dot{x}_0 = x_2 * cos(x_3) + d0
    #         \dot{x}_1 = x_2 * sin(x_3) + d1
    #         \dot{x}_2 = a + d2
    #         \dot{x}_3 = x_2/L * tan(x_4) + d3
    #         \dot{x}_4 = w + d4

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] * \
            torch.cos(state[..., 3]) + disturbance[..., 0]
        dsdt[..., 1] = state[..., 2] * \
            torch.sin(state[..., 3]) + disturbance[..., 1]
        dsdt[..., 2] = control[..., 0] + disturbance[..., 2]
        dsdt[..., 3] = state[..., 2]/self.L * \
            torch.tan(state[..., 4]) + disturbance[..., 3]
        dsdt[..., 4] = control[..., 1] + disturbance[..., 4]
        return dsdt

    def boundary_fn(self, state):
        # Defining a Rectangular Target
        # return torch.max(torch.abs(state[..., 0]) - 2, torch.abs(state[..., 1]) - 3)

        # Defining a Circular Target
        # obstacles (x,y,r): (-2,-1,0.6), (-0.1,0.2,0.5), (-1,1.5,0.9), (1,2.3,0.4), (1.5,0.1,0.9) * 3 #, (0.7,-1.6,0.6), (1,3,0.4)
        dp1 = state[..., 0:2]*1.0
        dp1[..., 0] = dp1[..., 0]-(-6.0)
        dp1[..., 1] = dp1[..., 1]-(-3.0)
        dist1 = torch.norm(dp1[..., 0:2], dim=-1) - 1.8

        dp2 = state[..., 0:2]*1.0
        dp2[..., 0] = dp2[..., 0]-(-0.3)
        dp2[..., 1] = dp2[..., 1]-(0.6)
        dist2 = torch.norm(dp2[..., 0:2], dim=-1) - 1.5

        dp3 = state[..., 0:2]*1.0
        dp3[..., 0] = dp3[..., 0]-(-3.0)
        dp3[..., 1] = dp3[..., 1]-(4.5)
        dist3 = torch.norm(dp3[..., 0:2], dim=-1) - 2.7

        dp4 = state[..., 0:2]*1.0
        dp4[..., 0] = dp4[..., 0]-(3)
        dp4[..., 1] = dp4[..., 1]-(6.9)
        dist4 = torch.norm(dp4[..., 0:2], dim=-1) - 1.2

        dp5 = state[..., 0:2]*1.0
        dp5[..., 0] = dp5[..., 0]-(4.5)
        dp5[..., 1] = dp5[..., 1]-(0.3)
        dist5 = torch.norm(dp5[..., 0:2], dim=-1) - 2.7

        # dp6 = state[..., 0:2]*1.0
        # dp6[..., 0] = dp6[..., 0]-(0.7)
        # dp6[..., 1] = dp6[..., 1]-(-1.6)
        # dist6 = torch.norm(dp6[..., 0:2], dim=-1) - 0.6

        # dp7 = state[..., 0:2]*1.0
        # dp7[..., 0] = dp7[..., 0]-(1)
        # dp7[..., 1] = dp7[..., 1]-(3)
        # dist7 = torch.norm(dp7[..., 0:2], dim=-1) - 0.4

        lx = (torch.minimum(torch.minimum(torch.minimum(torch.minimum(dist1, dist2), dist3), dist4), dist5))
        return lx


    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            return (state[..., 2] * torch.cos(state[..., 3]) * dvds[..., 0]) \
                    + (state[..., 2] * torch.sin(state[..., 3]) * dvds[..., 1]) \
                    - (self.a_max) * torch.abs(dvds[..., 2])\
                    + (state[..., 2] / self.L * torch.tan(state[..., 4]) * dvds[..., 3])\
                    - (self.w_max) * torch.abs(dvds[..., 4])

        elif self.set_mode == 'avoid':
            return (state[..., 2] * torch.cos(state[..., 3]) * dvds[..., 0]) \
                    + (state[..., 2] * torch.sin(state[..., 3]) * dvds[..., 1]) \
                    + (self.a_max) * torch.abs(dvds[..., 2])\
                    + (state[..., 2] / self.L * torch.tan(state[..., 4]) * dvds[..., 3])\
                    + (self.w_max) * torch.abs(dvds[..., 4])


        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            control1 = (-self.a_max*torch.sign(dvds[..., 2]))[..., None]
            control2 = (-self.w_max*torch.sign(dvds[..., 4]))[..., None]
            return torch.cat((control1, control2), dim=-1)

        elif self.set_mode == 'avoid':
            control1 = (self.a_max*torch.sign(dvds[..., 2]))[..., None]
            control2 = (self.w_max*torch.sign(dvds[..., 4]))[..., None]
            return torch.cat((control1, control2), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros_like(state)

    def plot_config(self):
        return {
            'state_slices': [0, 0, 4, 0, 0],
            'state_labels': ['px', 'py', 'v', r'$\psi$', r'$\delta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 3,
        }
