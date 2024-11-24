import torch
from torch.utils.data import Dataset
import numpy as np


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(
            np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, num_switch_samples):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin
        self.tMax = tMax
        self.counter = counter_start
        self.counter_end = counter_end
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples
        self.num_switch_samples = num_switch_samples
    def __len__(self):
        return 1

    def normalize_q(self, x):
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x

    def __getitem__(self, idx):
        # uniformly sample domain and include coordinates where source is non-zero
        model_states = torch.zeros(
            self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)

        # TODO: currently hard-coded for 13D quadrotor dynamics. Handling it in the dynamics makes more sense.
        # if self.dynamics.state_dim == 13:
        #     model_states = self.normalize_q(model_states)  # normalize q
        
        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(
                self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(
                self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        switch_samples = None
        if self.num_switch_samples > 0:
            switch_times = self.tMin + torch.zeros(self.num_switch_samples, 1).uniform_(
                0, (self.tMax-self.tMin) * min(self.counter / self.counter_end, 1.0))
            switch_samples = self.dynamics.sample_switch_state(self.num_switch_samples)
            # need to convert them to input, very important!
            switch_samples = (switch_samples - self.dynamics.state_mean.to(device=switch_samples.device)
                          ) / self.dynamics.state_var.to(device=switch_samples.device)
            
            near_switch_samples = switch_samples[[0],...]*1.0
            near_switch_samples = torch.clamp(near_switch_samples + torch.zeros_like(near_switch_samples).uniform_(-0.02,0.02), max=1.0, min=-1.0)
            switch_samples= torch.cat((switch_samples,near_switch_samples), dim=0)
            switch_samples = torch.cat((switch_times.repeat(3,1,1), switch_samples), dim=-1)

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), self.tMin)
        else:
            # slowly grow time values from start time
            times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(
                0, (self.tMax-self.tMin) * min(self.counter / self.counter_end, 1.0))
            # make sure we always have training samples at the initial time
            if self.dynamics.deepReach_model in ["reg","diff"]:
                times[-self.num_src_samples:, 0] = self.tMin
            # else:
            #     near_switch_samples = self.dynamics.sample_switch_state(self.num_src_samples)[0,...]
            #     # need to convert them to input, very important!
            #     near_switch_samples = (near_switch_samples - self.dynamics.state_mean.to(device=near_switch_samples.device)
            #                 ) / self.dynamics.state_var.to(device=near_switch_samples.device)
            #     near_switch_samples = torch.clamp(near_switch_samples + torch.zeros_like(near_switch_samples).uniform_(-0.05,0.05),max=1.0, min=-1.0)
            #     model_states[-self.num_src_samples:, :] = near_switch_samples

        model_coords = torch.cat((times, model_states), dim=1)

        # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
        if self.dynamics.input_dim > self.dynamics.state_dim + 1:
            model_coords = torch.cat((model_coords, torch.zeros(
                self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)

        boundary_values = self.dynamics.boundary_fn(
            self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(
                self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(
                self.dynamics.input_to_coord(model_coords)[..., 1:])

        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        else:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.num_switch_samples > 0:
            switch_mask = self.dynamics.find_switching_surface_mask(model_coords)

        
        if self.dynamics.loss_type == 'brt_hjivi' and self.num_switch_samples > 0:
            return {'model_coords': model_coords, 'switch_samples': switch_samples}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks, 'switch_masks': switch_mask}
        elif self.dynamics.loss_type == 'brt_hjivi' and self.num_switch_samples == 0:
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError
