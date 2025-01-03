import wandb
import torch
import os
import shutil
import time
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm
from utils import diff_operators
from utils.error_evaluators import performance_scenario_optimization, scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
import seaborn as sns
from scipy.stats import beta as beta__dist


class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(
                self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path)['model'])
        else:
            model_path = os.path.join(
                self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    def plotCompassWalker(self,time_resolution):
        
        xs = torch.linspace(-0.52, 0.52, 30)
        ys = torch.linspace(-1.04, 1.04, 30)
        dq1s = torch.linspace(-4.0, 4.0, 30)
        dq2s = torch.linspace(-8.0, 8.0, 30)
        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        fig = plt.figure(figsize=(6, 5*len(times)))
        for i in range(len(times)):
            coords = torch.zeros(
                    30**4, self.dataset.dynamics.state_dim + 1)
            coords[...,0] = times[i]
            coords[...,1:] = torch.cartesian_prod(xs, ys, dq1s, dq2s)
            with torch.no_grad():
                model_results_=self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(model_results_['model_in'].detach(
                        ), model_results_['model_out'].squeeze(dim=-1).detach())
                
            ax = fig.add_subplot(len(times), 1, 1 + i)
            ax.set_title('t = %0.2f' % (times[i]))
            brt_counts=torch.sum(torch.sum(values.reshape(30,30,30,30)<=0,axis=-1),axis=-1).T.float()

            BRT_img = brt_counts.detach().cpu().numpy()
            imshow_kwargs = {
                'vmax': max(10,float(torch.max(brt_counts.flatten()))),
                'vmin': 0,
                'cmap': 'YlGnBu',
                'extent': (-0.52, 0.52, -1.04, 1.04),
                'origin': 'lower',
            }
            plt.plot([0,-0.52],[0,1.04],'r')
            plt.plot(self.dataset.dynamics.gait_scaled[...,0],self.dataset.dynamics.gait_scaled[...,1],'b')
            s1 = ax.imshow(BRT_img, **imshow_kwargs)
            fig.colorbar(s1)

        
        return fig, fig
    
    def plotSingleFig(self, state_test_range, plot_config, x_resolution, y_resolution, time_resolution):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys)
        fig = plt.figure(figsize=(6, 5*len(times)))
        fig2 = plt.figure(figsize=(6, 5*len(times)))

        for i in range(len(times)):
            coords = torch.zeros(
                x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i]
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            with torch.no_grad():
                model_results = self.model(
                    {'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                if self.dataset.dynamics.deepReach_model == 'exact_diff':
                    t0_input = model_results['model_in'][..., 1:].detach(
                    ).clone().requires_grad_()
                    t0_input = torch.cat(
                        (torch.zeros(t0_input.shape[0], 1).to(t0_input.device), t0_input), dim=-1)
                    model_results2 = self.model(
                        {'coords': t0_input})
                    model_results['model_in'] = (
                        model_results['model_in'], model_results2['model_in'])
                    model_results['model_out'] = (
                        model_results['model_out'], model_results2['model_out'])

                    values = self.dataset.dynamics.io_to_value(
                        model_results['model_in'], model_results['model_out'])
                else:
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                    ), model_results['model_out'].squeeze(dim=-1).detach())

            ax = fig.add_subplot(len(times), 1, 1 + i)
            ax.set_title('t = %0.2f' % (times[i]))
            BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
            max_value = np.amax(BRT_img)
            min_value = np.amin(BRT_img)
            # We'll also create a grey background into which the pixels will fade
            greys = np.full((*BRT_img.shape, 3), 70, dtype=np.uint8)
            imshow_kwargs = {
                'vmax': max_value,
                'vmin': min_value,
                'cmap': 'RdYlBu',
                'extent': (x_min, x_max, y_min, y_max),
                'origin': 'lower',
                'aspect': 'auto',
            }
            ax.imshow(greys)
            ax.imshow(BRT_img, **imshow_kwargs)

            ax2 = fig2.add_subplot(len(times), 1, 1 + i)
            ax2.set_title('t = %0.2f' % (times[i]))
            ax2.imshow(1*(BRT_img <= 0), cmap='bwr',
                       origin='lower', extent=(x_min, x_max, y_min, y_max), aspect='auto')

        return fig, fig2

    def plotMultipleFigs(self, state_test_range, plot_config, x_resolution, y_resolution, z_resolution, time_resolution):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]
        # z_min, z_max = (torch.Tensor([0]), torch.Tensor([6]))

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min+0.1, z_max-0.1, z_resolution)
        xys = torch.cartesian_prod(xs, ys)

        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        fig2 = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(
                    x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model(
                        {'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    if self.dataset.dynamics.deepReach_model == 'exact_diff':
                        t0_input = model_results['model_in'][..., 1:].detach(
                        ).clone().requires_grad_()
                        # print(t0_input.shape)
                        t0_input = torch.cat(
                            (torch.zeros(t0_input.shape[0], 1).to(t0_input.device), t0_input), dim=-1)

                        model_results2 = self.model(
                            {'coords': t0_input})
                        model_results['model_in'] = (
                            model_results['model_in'], model_results2['model_in'])
                        model_results['model_out'] = (
                            model_results['model_out'], model_results2['model_out'])

                        values = self.dataset.dynamics.io_to_value(
                            model_results['model_in'], model_results['model_out'])
                    else:
                        values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                        ), model_results['model_out'].squeeze(dim=-1).detach())

                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (
                    times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))

                BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T

                max_value = np.amax(BRT_img)
                min_value = np.amin(BRT_img)
                # We'll also create a grey background into which the pixels will fade
                greys = np.full((*BRT_img.shape, 3), 70, dtype=np.uint8)
                imshow_kwargs = {
                    'vmax': max_value,
                    'vmin': min_value,
                    'cmap': 'RdYlBu',
                    'extent': (x_min, x_max, y_min, y_max),
                    'origin': 'lower',
                }
                ax.imshow(greys)
                s1 = ax.imshow(BRT_img, **imshow_kwargs)
                fig.colorbar(s1)

                ax2 = fig2.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax2.set_title('t = %0.2f, %s = %0.2f' % (
                    times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                ax2.imshow(greys)
                ax2.imshow(1*(BRT_img <= 0), cmap='bwr',
                           origin='lower', extent=(x_min, x_max, y_min, y_max))
        return fig, fig2

    def validate(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        if self.dataset.dynamics.state_dim==4: # if it is CompassWalker TODO: add name member variable
            #fig, fig2 = self.plotCompassWalker(time_resolution)
            fig, fig2 = self.plotSingleFig(
                state_test_range, plot_config, x_resolution, y_resolution, time_resolution)
        elif plot_config['z_axis_idx'] == -1:
            fig, fig2 = self.plotSingleFig(
                state_test_range, plot_config, x_resolution, y_resolution, time_resolution)
            # fig.savefig(save_path)
        else:
            fig, fig2 = self.plotMultipleFigs(
                state_test_range, plot_config, x_resolution, y_resolution, z_resolution, time_resolution)

        wandb.log({
            'step': epoch,
            'val_plot': wandb.Image(fig),
            'val_plot2': wandb.Image(fig2),
        })
        plt.close()
        plt.close()
        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def train(
        self, batch_size, epochs, lr, csl_lr,
        steps_til_summary, epochs_til_checkpoint,
        loss_fn, clip_grad, use_lbfgs, adjust_relative_grads,
        val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
        use_CSL, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size, viscosity_coef
    ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(
            self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=1)  # by default running without scheduler
        # copy settings from Raissi et al. (2019) and here
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                      history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')

        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0

        if adjust_relative_grads:
            new_weight = 1

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):

                if self.dataset.pretrain:  # skip CSL
                    last_CSL_epoch = epoch
                # if current epochs exceed the counter end, then we train with t \in [tMin,tMax]
                time_interval_length = min(1.0,
                                           self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)

                # self-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()

                    model_input = {key: value.cuda()
                                   for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    model_results = self.model(
                        {'coords': model_input['model_coords']})

                    states = self.dataset.dynamics.input_to_coord(
                        model_results['model_in'].detach())[..., 1:]
                    
                    if self.dataset.dynamics.deepReach_model == 'exact_diff':
                        t0_input = model_results['model_in'][..., 1:].detach(
                        ).clone().requires_grad_()
                        t0_input = torch.cat(
                            (torch.zeros(1, self.dataset.numpoints, 1).to(t0_input.device), t0_input), dim=-1)

                        model_results2 = self.model(
                            {'coords': t0_input})
                        model_results['model_in'] = (
                            model_results['model_in'], model_results2['model_in'])
                        model_results['model_out'] = (
                            model_results['model_out'], model_results2['model_out'])

                        values = self.dataset.dynamics.io_to_value(
                            model_results['model_in'], model_results['model_out'])

                        dvs = self.dataset.dynamics.io_to_dv(
                            model_results['model_in'], model_results['model_out'])

                    else:
                        values = self.dataset.dynamics.io_to_value(
                            model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))

                        dvs = self.dataset.dynamics.io_to_dv(
                            model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                        
                    # switch_value1=torch.tensor([0.0]).cuda()
                    # switch_value2=torch.tensor([0.0]).cuda()
                    # switch_value3=torch.tensor([0.0]).cuda()
                    #if model_input["switch_samples"] is not None:
                         # print(model_input['model_coords'].shape, model_input['switch_samples'].shape)

                    #     switch_results1 = self.model(
                    #        {'coords': model_input['switch_samples'][:,0,...]})
                         # switch_value1=switch_results1['model_out'].detach()
                    #     switch_value1= self.dataset.dynamics.io_to_value(
                    #         switch_results1['model_in'].detach(), switch_results1['model_out'].squeeze(dim=-1))
                        
                    #     switch_results2 = self.model(
                    #         {'coords': model_input['switch_samples'][:,1,...]})
                         # switch_value2=switch_results2['model_out'].detach()
                    #     switch_value2 = self.dataset.dynamics.io_to_value(
                    #         switch_results2['model_in'].detach(), switch_results2['model_out'].squeeze(dim=-1))
                        
                    #     switch_results3 = self.model(
                    #         {'coords': model_input['switch_samples'][:,2,...]})
                    #     switch_value3 = self.dataset.dynamics.io_to_value(
                    #         switch_results3['model_in'].detach(), switch_results3['model_out'].squeeze(dim=-1))
                         #print(model_input['switch_samples'][...,0])
                         #print(model_input['switch_samples'][0,:,:5,:])
                    #    switch_states1=self.dataset.dynamics.input_to_coord(model_input['switch_samples'][0,0,:5,:])[..., 1:]
                    #     switch_states2=self.dataset.dynamics.input_to_coord(model_input['switch_samples'][0,1,:5,:])[..., 1:]
                        # print(switch_states1,switch_states2)
                        # print(self.dataset.dynamics.boundary_fn(switch_states1,verbose=True))
                        # print(self.dataset.dynamics.boundary_fn(switch_states2))
                        # print(switch_value2-switch_value1)

                    boundary_values = gt['boundary_values']
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']
                    #switch_mask = gt['switch_masks']

                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(
                            states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out']) #switch_results1, switch_results2, switch_results3)
                    elif self.dataset.dynamics.loss_type == 'brt_aug_hjivi':
                        losses = loss_fn(
                            states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(
                            states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    else:
                        raise NotImplementedError

                    # Adjust the relative magnitude of the losses if required
                    if self.dataset.dynamics.deepReach_model in ['reg', 'diff'] and adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(
                                retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            # Set the new weight according to the paper
                            # num = torch.max(torch.abs(grads_PDE))
                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight * \
                                losses['dirichlet']
                        writer.add_scalar('weight_scaling',
                                          new_weight, total_steps)

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_name == 'dirichlet':
                            writer.add_scalar(
                                loss_name, single_loss/new_weight, total_steps)
                        else:
                            writer.add_scalar(
                                loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss",
                                      train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(self.model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=clip_grad)

                        optim.step()
                        scheduler.step()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (
                            epoch, train_loss, time.time() - start_time))
                        wandb.log({
                            'train_loss': train_loss,
                            'pde_loss': losses['diff_constraint_hom'],
                            # 'switch_loss': losses['switch_loss'],
                            # 'boundary_loss': losses['dirichlet'],
                        })

                    total_steps += 1

                if not (epoch+1) % epochs_til_checkpoint:
                    # Saving the optimizer state is important to produce consistent results
                    checkpoint = {
                        'epoch': epoch+1,
                        'model': self.model.state_dict(),
                        'optimizer': optim.state_dict()}
                    torch.save(checkpoint,
                               os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                               np.array(train_losses))
                    self.validate(
                        epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        x_resolution=val_x_resolution, y_resolution=val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

        torch.save(checkpoint, os.path.join(
            checkpoints_dir, 'model_final.pth'))

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        if data_step in ["plot_basic_recovery", 'run_basic_recovery']:
            testing_dir = self.experiment_dir
        else:
            testing_dir = os.path.join(
                self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
            if os.path.exists(testing_dir):
                overwrite = input(
                    "The testing directory %s already exists. Overwrite? (y/n)" % testing_dir)
                if not (overwrite == 'y'):
                    print('Exiting.')
                    quit()
                shutil.rmtree(testing_dir)
            os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            # checkpoint x simulation_time square matrices
            sidelen = 10
            assert (last_checkpoint /
                    checkpoint_dt) % sidelen == 0, 'checkpoints cannot be even divided by sidelen'
            BRT_volumes_matrix = np.zeros((sidelen, sidelen))
            BRT_errors_matrix = np.zeros((sidelen, sidelen))
            BRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            BRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            exBRT_volumes_matrix = np.zeros((sidelen, sidelen))
            exBRT_errors_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            checkpoints = np.linspace(0, last_checkpoint, num=sidelen+1)[1:]
            checkpoints[-1] = -1
            times = np.linspace(self.dataset.tMin,
                                self.dataset.tMax, num=sidelen+1)[1:]
            print('constructing matrices for')
            print('checkpoints:', checkpoints)
            print('times:', times)
            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                for j in tqdm(range(sidelen), desc='Simulation Time', leave=False):
                    # get BRT volume, error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        violation_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    BRT_volumes_matrix[i, j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        BRT_errors_matrix[i,
                                          j] = results['max_violation_error']
                        BRT_error_rates_matrix[i,
                                               j] = results['violation_rate']
                        BRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=float('-inf'), v_max=0.0),
                            target_validator=ValueThresholdValidator(
                                v_min=-results['max_violation_error'], v_max=0.0),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        BRT_errors_matrix[i, j] = np.NaN
                        BRT_error_rates_matrix[i, j] = np.NaN
                        BRT_error_region_fracs_matrix[i, j] = np.NaN

                    # get exBRT error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    exBRT_volumes_matrix[i,
                                         j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        exBRT_errors_matrix[i,
                                            j] = results['max_violation_error']
                        exBRT_error_rates_matrix[i,
                                                 j] = results['violation_rate']
                        exBRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=float('inf')),
                            target_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=results['max_violation_error']),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        exBRT_errors_matrix[i, j] = np.NaN
                        exBRT_error_rates_matrix[i, j] = np.NaN
                        exBRT_error_region_fracs_matrix[i, j] = np.NaN

            # save the matrices
            matrices = {
                'BRT_volumes_matrix': BRT_volumes_matrix,
                'BRT_errors_matrix': BRT_errors_matrix,
                'BRT_error_rates_matrix': BRT_error_rates_matrix,
                'BRT_error_region_fracs_matrix': BRT_error_region_fracs_matrix,
                'exBRT_volumes_matrix': exBRT_volumes_matrix,
                'exBRT_errors_matrix': exBRT_errors_matrix,
                'exBRT_error_rates_matrix': exBRT_error_rates_matrix,
                'exBRT_error_region_fracs_matrix': exBRT_error_region_fracs_matrix,
            }
            for name, arr in matrices.items():
                with open(os.path.join(testing_dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, arr)

            # plot the matrices
            matrices = {
                'BRT_volumes_matrix': [
                    BRT_volumes_matrix, 'BRT Fractions of Test State Space'
                ],
                'BRT_errors_matrix': [
                    BRT_errors_matrix, 'BRT Errors'
                ],
                'BRT_error_rates_matrix': [
                    BRT_error_rates_matrix, 'BRT Error Rates'
                ],
                'BRT_error_region_fracs_matrix': [
                    BRT_error_region_fracs_matrix, 'BRT Error Region Fractions'
                ],
                'exBRT_volumes_matrix': [
                    exBRT_volumes_matrix, 'exBRT Fractions of Test State Space'
                ],
                'exBRT_errors_matrix': [
                    exBRT_errors_matrix, 'exBRT Errors'
                ],
                'exBRT_error_rates_matrix': [
                    exBRT_error_rates_matrix, 'exBRT Error Rates'
                ],
                'exBRT_error_region_fracs_matrix': [
                    exBRT_error_region_fracs_matrix, 'exBRT Error Region Fractions'
                ],
            }
            for name, data in matrices.items():
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                ax.imshow(data[0], cmap=cmap)
                plt.title(data[1])
                for (y, x), label in np.ndenumerate(data[0]):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(testing_dir, name + '.png'), dpi=600)
                plt.clf()
                # log version
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                new_matrix = np.log(data[0])
                ax.imshow(new_matrix, cmap=cmap)
                plt.title('(Log) ' + data[1])
                for (y, x), label in np.ndenumerate(new_matrix):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(
                    testing_dir, name + '_log' + '.png'), dpi=600)
                plt.clf()

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics

            if data_step == 'plot_violations':
                # plot violations on slice
                plot_config = dynamics.plot_config()
                slices = plot_config['state_slices']
                slices[plot_config['x_axis_idx']] = None
                slices[plot_config['y_axis_idx']] = None
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=100000, sample_batch_size=100000,
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=slices),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                plt.title('violations for slice = %s' %
                          plot_config['state_slices'], fontsize=8)
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][~results['violations']], results['states']
                            [...,  plot_config['y_axis_idx']][~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][results['violations']], results['states']
                            [...,  plot_config['y_axis_idx']][results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                x_min, x_max = dynamics.state_test_range()[
                    plot_config['x_axis_idx']]
                y_min, y_max = dynamics.state_test_range()[
                    plot_config['y_axis_idx']]
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.savefig(os.path.join(
                    testing_dir, f'violations.png'), dpi=800)
                plt.clf()

                # plot distribution of violations over state variables
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=100000, sample_batch_size=100000,
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                for i in range(dynamics.state_dim):
                    plt.title('violations over %s' %
                              plot_config['state_labels'][i])
                    plt.scatter(results['states'][..., i][~results['violations']], results['values']
                                [~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                    plt.scatter(results['states'][..., i][results['violations']], results['values']
                                [results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                    plt.savefig(os.path.join(
                        testing_dir, f'violations_over_state_dim_{i}.png'), dpi=800)
                    plt.clf()
                    
            if data_step == 'run_robust_recovery':
                logs = {}

                # rollout samples all over the state space
                beta_ = 1e-10
                N = 300000

                logs['beta_'] = beta_
                logs['N'] = N

                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level,
                        v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                values_ = results['values'].cpu().numpy()
                # print(results)
                unsafe_cost_safe_value_indeces = np.argwhere(
                    np.logical_and(costs_ > 0, values_ <= 0))

                print("k max: ", unsafe_cost_safe_value_indeces.shape[0])
                # determine delta_level_max
                #delta_level_max = np.max(
                #    values_[unsafe_cost_safe_value_indeces])
                #print("delta_level_max: ", delta_level_max)
                delta_level_min = np.min(
                    values_[unsafe_cost_safe_value_indeces])
                print("delta_level_min: ", delta_level_min)
                # print(costs_[unsafe_cost_safe_value_indeces], values_[unsafe_cost_safe_value_indeces])
                # for each delta level, determine (1) the corresponding volume;
                # (2) k and and corresponding epsilon
                ks = []
                epsilons = []
                volumes = []

                for delta_level_ in np.arange(delta_level_min, 0, abs(delta_level_min/100)):
                    k = int(np.argwhere(np.logical_and(
                        costs_ > 0, values_ <= delta_level_)).shape[0])
                    eps = beta__dist.ppf(beta_,  N-k, k+1)
                    volume = values_[values_ <= delta_level_].shape[0]/N
                    ks.append(k)
                    epsilons.append(eps)
                    print("epsilon:",eps, "Delta Level:", delta_level_, "No. of Outliers:", k)
                    volumes.append(volume)
                # plot epsilon volume graph
                fig1, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('volumes')
                ax1.set_ylabel('epsilons', color=color)
                ax1.plot(volumes, epsilons, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()

                color = 'tab:blue'
                ax2.set_ylabel('number of outliers', color=color)
                ax2.plot(volumes, ks, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                # fig1.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.title("beta_=1e-10, N =3e6")
                fig1.savefig(os.path.join(
                    testing_dir, f'robust_verification_results.png'), dpi=800)
                plt.close(fig1)

            if data_step == 'run_perf_recovery':
                logs = {}

                # rollout samples all over the state space
                beta_ = 1e-10
                N = 300000

                logs['beta_'] = beta_
                logs['N'] = N

                delta = -0.03

                results = performance_scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta,
                        v_max=float('inf')),
                    delta = delta,
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                # print(costs_)
                values_ = results['values'].cpu().numpy()
                psi_level_max = np.max(costs_)
                print("psi_level_max: ", psi_level_max)

                ks = []
                epsilons = []
                volumes = []

                unsafe_k = int(np.argwhere(costs_ == 0).shape[0])
                print("Unsafe Volume:", unsafe_k/N)

                for psi_level_ in np.arange(0, psi_level_max, abs(psi_level_max/100)):
                    k = int(np.argwhere(costs_ >= psi_level_).shape[0])
                    eps = beta__dist.ppf(beta_,  N-k, k+1)
                    volume = 0# costs_[costs_ <= delta_level_].shape[0]/N
                    ks.append(k)
                    epsilons.append(eps)
                    print("epsilon:",eps, "Psi Level:", psi_level_, "No. of Outliers:", k)
                    volumes.append(volume)
                # plot epsilon volume graph
                fig1, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('volumes')
                ax1.set_ylabel('epsilons', color=color)
                ax1.plot(volumes, epsilons, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()

                color = 'tab:blue'
                ax2.set_ylabel('number of outliers', color=color)
                ax2.plot(volumes, ks, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                # fig1.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.title("beta_=1e-10, N =3e6")
                fig1.savefig(os.path.join(
                    testing_dir, f'performance_verification_results.png'), dpi=800)
                plt.close(fig1)

            if data_step == 'run_basic_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for tMax
                # record state/learned_value/violation for each while loop iteration
                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')
                algorithm_iters = []
                for i in range(M):
                    print('algorithm iter', str(i))
                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=N, max_samples=1000*min(N, 10000))
                    if not results['maxed_scenarios']:
                        delta_level = float(
                            '-inf') if dynamics.set_mode == 'reach' else float('inf')
                        break
                    algorithm_iters.append(
                        {
                            'states': results['states'],
                            'values': results['values'],
                            'violations': results['violations']
                        }
                    )
                    if results['violation_rate'] == 0:
                        break
                    violation_levels = results['values'][results['violations']]
                    delta_level_arg = np.argmin(
                        violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                    delta_level = violation_levels[delta_level_arg].item()

                    print('violation_rate:', str(results['violation_rate']))
                    print('delta_level:', str(delta_level))
                    print('valid_sample_fraction:', str(
                        results['valid_sample_fraction'].item()))
                    sns.set_style('whitegrid')
                    # density_plot=sns.kdeplot(results['costs'].cpu().numpy(), bw=0.5)
                    # density_plot=sns.displot(results['costs'].cpu().numpy(), x="cost function")
                    # fig1 = density_plot.get_figure()
                    fig1 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy(), bins=200)
                    fig1.savefig(os.path.join(
                        testing_dir, f'cost distribution.png'), dpi=800)
                    plt.close(fig1)
                    fig2 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy() -
                             results['values'].cpu().numpy(), bins=200)
                    fig2.savefig(os.path.join(
                        testing_dir, f'diff distribution.png'), dpi=800)
                    plt.close(fig1)

                logs['algorithm_iters'] = algorithm_iters
                logs['delta_level'] = delta_level

                # 2. record solution volume, recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['recovered_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000)
                ).item()

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0

                print('learned_volume', str(logs['learned_volume']))
                print('recovered_volume', str(logs['recovered_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['recovered_violation_rate'] = results['violation_rate']
                else:
                    logs['recovered_violation_rate'] = 0
                print('recovered_violation_rate', str(
                    logs['recovered_violation_rate']))

                with open(os.path.join(testing_dir, 'basic_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_basic_recovery':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                print('S:', str(logs['S']))
                print('delta level', str(logs['delta_level']))
                delta_level = logs['delta_level']
                print('learned volume', str(logs['learned_volume']))
                print('recovered volume', str(logs['recovered_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('recovered violation rate', str(
                    logs['recovered_violation_rate']))

                fig = self.plot_recovery_fig(
                    dataset, dynamics, model, delta_level)

                plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'basic_BRTs.png'), dpi=800)

            if data_step == 'collect_samples':
                logs = {}

                # 1. record 10M state, learned value, violation
                P = int(1e7)
                logs['P'] = P
                print('collecting training samples')
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(P, 100000), sample_batch_size=10*min(P, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=P, max_samples=1000*min(P, 10000))
                logs['training_samples'] = {
                    'states': results['states'],
                    'values': results['values'],
                    'violations': results['violations'],
                }
                with open(os.path.join(testing_dir, 'sample_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'train_binner':
                with open(os.path.join(self.experiment_dir, 'sample_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 1. train MLP predictor
                # plot validation of MLP predictor
                def validate_predictor(predictor, epoch):
                    print('validating predictor at epoch', str(epoch))
                    predictor.eval()

                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=100000, sample_batch_size=100000,
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=100000, max_samples=1000000)

                    inputs = torch.cat(
                        (results['states'], results['values'][..., None]), dim=-1)
                    preds = torch.sigmoid(
                        predictor(inputs.cuda())).detach().cpu().numpy()

                    plt.title(f'Predictor Validation at Epoch {epoch}')
                    plt.ylabel('Value')
                    plt.xlabel('Prediction')
                    plt.scatter(preds[~results['violations']], results['values']
                                [~results['violations']], color='blue', label='nonviolations', alpha=0.1)
                    plt.scatter(preds[results['violations']], results['values']
                                [results['violations']], color='red', label='violations', alpha=0.1)
                    plt.legend()
                    plt.savefig(os.path.join(
                        testing_dir, f'predictor_validation_at_epoch_{epoch}.png'), dpi=800)
                    plt.clf()

                    predictor.train()

                print('training predictor')
                violation_scale = 5
                violation_weight = 1.5
                states = logs['training_samples']['states']
                values = logs['training_samples']['values']
                violations = logs['training_samples']['violations']
                violation_strengths = torch.where(violations, (torch.max(
                    values[violations]) - values) if dynamics.set_mode == 'reach' else (values - torch.min(values[violations])), torch.tensor([0.0])).cuda()
                violation_scales = torch.exp(
                    violation_scale * violation_strengths / torch.max(violation_strengths))

                plt.title(f'Violation Scales')
                plt.ylabel('Frequency')
                plt.xlabel('Scale')
                plt.hist(violation_scales.cpu().numpy(), range=(0, 10))
                plt.savefig(os.path.join(
                    testing_dir, f'violation_scales.png'), dpi=800)
                plt.clf()

                inputs = torch.cat((states, values[..., None]), dim=-1).cuda()
                outputs = 1.0*violations.cuda()
                # outputs = violation_strengths / torch.max(violation_strengths)

                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.cuda()
                predictor.train()

                lr = 0.00005
                lr_decay = 0.2
                decay_patience = 20
                decay_threshold = 1e-12
                opt = torch.optim.Adam(predictor.parameters(), lr=lr)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=lr_decay, patience=decay_patience, threshold=decay_threshold)

                pos_weight = violation_weight * \
                    ((outputs <= 0).sum() / (outputs > 0).sum())

                n_epochs = 1000
                batch_size = 100000
                for epoch in range(n_epochs):
                    idxs = torch.randperm(len(outputs))
                    for batch in range(math.ceil(len(outputs) / batch_size)):
                        batch_idxs = idxs[batch *
                                          batch_size: (batch+1)*batch_size]

                        BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(
                            weight=violation_scales[batch_idxs], pos_weight=pos_weight)
                        loss = BCEWithLogitsLoss(
                            predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        # MSELoss = torch.nn.MSELoss()
                        # loss = MSELoss(predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        loss.backward()
                        opt.step()
                    print(f'Epoch {epoch}: loss: {loss.item()}')
                    sched.step(loss.item())

                    if (epoch+1) % 100 == 0:
                        torch.save(predictor.state_dict(), os.path.join(
                            testing_dir, f'predictor_at_epoch_{epoch}.pth'))
                        validate_predictor(predictor, epoch)
                logs['violation_scale'] = violation_scale
                logs['violation_weight'] = violation_weight
                logs['n_epochs'] = n_epochs
                logs['lr'] = lr
                logs['lr_decay'] = lr_decay
                logs['decay_patience'] = decay_patience
                logs['decay_threshold'] = decay_threshold

                with open(os.path.join(testing_dir, 'train_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'run_binned_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for each bin of MLP predictor
                epoch = 699
                logs['epoch'] = epoch
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(
                    self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.train()

                bins = [0, 0.8, 0.85, 0.9, 0.95, 1]
                logs['bins'] = bins

                binned_delta_levels = []
                for i in range(len(bins)-1):
                    print('bin', str(i))
                    binned_delta_level = float(
                        'inf') if dynamics.set_mode == 'reach' else float('-inf')
                    for j in range(M):
                        print('algorithm iter', str(j))
                        results = scenario_optimization(
                            model=model, dynamics=dynamics,
                            tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                            set_type=set_type, control_type=control_type,
                            scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                            sample_generator=SliceSampleGenerator(
                                dynamics=dynamics, slices=[None]*dynamics.state_dim),
                            sample_validator=MultiValidator([
                                MLPValidator(
                                    mlp=predictor, o_min=bins[i], o_max=bins[i+1], model=model, dynamics=dynamics),
                                ValueThresholdValidator(v_min=float(
                                    '-inf'), v_max=binned_delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=binned_delta_level, v_max=float('inf')),
                            ]),
                            violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                                'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                            max_scenarios=N, max_samples=100000*min(N, 10000))
                        if not results['maxed_scenarios']:
                            binned_delta_level = float(
                                '-inf') if dynamics.set_mode == 'reach' else float('inf')
                            break
                        if results['violation_rate'] == 0:
                            break
                        violation_levels = results['values'][results['violations']]
                        binned_delta_level_arg = np.argmin(
                            violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                        binned_delta_level = violation_levels[binned_delta_level_arg].item(
                        )
                        print('violation_rate:', str(
                            results['violation_rate']))
                        print('binned_delta_level:', str(binned_delta_level))
                        print('valid_sample_fraction:', str(
                            results['valid_sample_fraction'].item()))
                    binned_delta_levels.append(binned_delta_level)
                logs['binned_delta_levels'] = binned_delta_levels

                # 2. record solution volume, auto-binned recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['binned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [
                            binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0
                print('learned_volume', str(logs['learned_volume']))
                print('binned_volume', str(logs['binned_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [
                            binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['binned_violation_rate'] = results['violation_rate']
                else:
                    logs['binned_violation_rate'] = 0
                print('binned_violation_rate', str(
                    logs['binned_violation_rate']))

                with open(os.path.join(testing_dir, 'binned_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_binned_recovery':
                with open(os.path.join(self.experiment_dir, 'binned_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                # print('P:', str(logs['P']))
                print('S:', str(logs['S']))
                print('bins', str(logs['bins']))
                bins = logs['bins']
                print('binned delta levels', str(logs['binned_delta_levels']))
                binned_delta_levels = logs['binned_delta_levels']
                print('learned volume', str(logs['learned_volume']))
                print('binned volume', str(logs['binned_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('binned violation rate', str(
                    logs['binned_violation_rate']))

                epoch = logs['epoch']
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(
                    self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.eval()

                # 1. for ground truth slices (if available), record (higher-res) grid of learned values and MLP predictions
                # plot (with ground truth) learned BRTs, auto-binned recovered BRTs
                # plot MLP predictor bins
                z_res = 5
                plot_config = dataset.dynamics.plot_config()
                if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
                    ground_truth = spio.loadmat(os.path.join(
                        self.experiment_dir, 'ground_truth.mat'))
                    if 'gmat' in ground_truth:
                        ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                        ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                        ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                        ground_truth_values = ground_truth['data']
                        ground_truth_ts = np.linspace(
                            0, 1, ground_truth_values.shape[3])
                    elif 'g' in ground_truth:
                        ground_truth_xs = ground_truth['g']['vs'][0,
                                                                  0][0][0][:, 0]
                        ground_truth_ys = ground_truth['g']['vs'][0,
                                                                  0][1][0][:, 0]
                        ground_truth_zs = ground_truth['g']['vs'][0,
                                                                  0][2][0][:, 0]
                        ground_truth_ts = ground_truth['tau'][0]
                        ground_truth_values = ground_truth['data']

                    # idxs to plot
                    x_idxs = np.linspace(
                        0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
                    y_idxs = np.linspace(
                        0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
                    z_idxs = np.linspace(
                        0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
                    t_idxs = np.array(
                        [len(ground_truth_ts)-1]).astype(dtype=int)

                    # indexed ground truth to plot
                    ground_truth_xs = ground_truth_xs[x_idxs]
                    ground_truth_ys = ground_truth_ys[y_idxs]
                    ground_truth_zs = ground_truth_zs[z_idxs]
                    ground_truth_ts = ground_truth_ts[t_idxs]
                    ground_truth_values = ground_truth_values[
                        x_idxs[:, None, None, None],
                        y_idxs[None, :, None, None],
                        z_idxs[None, None, :, None],
                        t_idxs[None, None, None, :]
                    ]
                    ground_truth_grids = ground_truth_values
                    xs = ground_truth_xs
                    ys = ground_truth_ys
                    zs = ground_truth_zs

                else:
                    ground_truth_grids = None
                    resolution = 512
                    xs = np.linspace(*dynamics.state_test_range()
                                     [plot_config['x_axis_idx']], resolution)
                    ys = np.linspace(*dynamics.state_test_range()
                                     [plot_config['y_axis_idx']], resolution)
                    zs = np.linspace(*dynamics.state_test_range()
                                     [plot_config['z_axis_idx']], z_res)

                xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
                value_grids = np.zeros((len(zs), len(xs), len(ys)))
                prediction_grids = np.zeros((len(zs), len(xs), len(ys)))
                for i in range(len(zs)):
                    coords = torch.zeros(
                        xys.shape[0], dataset.dynamics.state_dim + 1)
                    coords[:, 0] = dataset.tMax
                    coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                    coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

                    model_results = model(
                        {'coords': dataset.dynamics.coord_to_input(coords.cuda())})
                    values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
                    ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
                    value_grids[i] = values.reshape(len(xs), len(ys))

                    inputs = torch.cat(
                        (coords[..., 1:], values[:, None]), dim=-1)
                    outputs = torch.sigmoid(
                        predictor(inputs.cuda()).cpu().squeeze(dim=-1))
                    prediction_grids[i] = outputs.reshape(
                        len(xs), len(ys)).detach().cpu()

                def overlay_ground_truth(image, z_idx):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
                    ground_truth_BRTs = ground_truth_grid < 0
                    for x in range(ground_truth_BRTs.shape[0]):
                        for y in range(ground_truth_BRTs.shape[1]):
                            if not ground_truth_BRTs[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_BRTs.shape[0] and neighbor[1] < ground_truth_BRTs.shape[1]:
                                    if not ground_truth_BRTs[neighbor]:
                                        image[x-thickness:x+1+thickness, y-thickness:y +
                                              1+thickness] = np.array([50, 50, 50])
                                        break

                def overlay_border(image, set, color):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    for x in range(set.shape[0]):
                        for y in range(set.shape[1]):
                            if not set[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                                    if not set[neighbor]:
                                        image[x-thickness:x+1, y -
                                              thickness:y+1+thickness] = color
                                        break

                fig = plt.figure()
                fig.suptitle(plot_config['state_slices'], fontsize=8)
                for i in range(len(zs)):
                    values = value_grids[i]
                    predictions = prediction_grids[i]

                    # learned BRT and bin-recovered BRT
                    ax = fig.add_subplot(1, len(zs), (i+1))
                    ax.set_title('%s = %0.2f' % (
                        plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

                    image = np.full((*values.shape, 3), 255, dtype=int)
                    BRT = values < 0
                    bin_recovered_BRT = np.full(values.shape, 0, dtype=bool)
                    # per bin, set accordingly
                    for j in range(len(bins)-1):
                        mask = (
                            (predictions >= bins[j])*(predictions < bins[j+1]))
                        binned_delta_level = binned_delta_levels[j]
                        bin_recovered_BRT[mask *
                                          (values < binned_delta_level)] = True

                    if dynamics.set_mode == 'reach':
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        overlay_border(image, bin_recovered_BRT,
                                       np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)
                    else:
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        # overlay recovered border over learned BRT
                        overlay_border(image, bin_recovered_BRT,
                                       np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)

                    ax.imshow(image.transpose(1, 0, 2),
                              origin='lower', extent=(-1., 1., -1., 1.))
                    ax.set_xlabel(
                        plot_config['state_labels'][plot_config['x_axis_idx']])
                    ax.set_ylabel(
                        plot_config['state_labels'][plot_config['y_axis_idx']])
                    ax.set_xticks([-1, 1])
                    ax.set_yticks([-1, 1])
                    ax.tick_params(labelsize=6)
                    if i != 0:
                        ax.set_yticks([])
                plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'binned_BRTs.png'), dpi=800)

            if data_step == 'plot_cost_function':
                if os.path.exists(os.path.join(self.experiment_dir, 'cost_logs.pickle')):
                    with open(os.path.join(self.experiment_dir, 'cost_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                else:
                    with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                    S = logs['S']
                    delta_level = logs['delta_level']
                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['learned_costs'] = results['costs']
                    else:
                        logs['learned_costs'] = None

                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['recovered_costs'] = results['costs']
                    else:
                        logs['recovered_costs'] = None

                if logs['learned_costs'] is not None and logs['recovered_costs'] is not None:
                    plt.title(f'Trajectory Costs')
                    plt.ylabel('Frequency')
                    plt.xlabel('Cost')
                    plt.hist(logs['learned_costs'], color=(
                        247/255, 187/255, 8/255), alpha=0.5)
                    plt.hist(logs['recovered_costs'], color=(
                        14/255, 222/255, 241/255), alpha=0.5)
                    plt.axvline(x=0, linestyle='--', color='black')
                    plt.savefig(os.path.join(
                        testing_dir, f'cost_function.png'), dpi=800)

                with open(os.path.join(testing_dir, 'cost_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def plot_recovery_fig(self, dataset, dynamics, model, delta_level):
        # 1. for ground truth slices (if available), record (higher-res) grid of learned values
        # plot (with ground truth) learned BRTs, recovered BRTs
        z_res = 5
        plot_config = dataset.dynamics.plot_config()
        if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
            ground_truth = spio.loadmat(os.path.join(
                self.experiment_dir, 'ground_truth.mat'))
            if 'gmat' in ground_truth:
                ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                ground_truth_values = ground_truth['data']
                ground_truth_ts = np.linspace(
                    0, 1, ground_truth_values.shape[3])

            elif 'g' in ground_truth:
                ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                ground_truth_ts = ground_truth['tau'][0]
                ground_truth_values = ground_truth['data']

            # idxs to plot
            x_idxs = np.linspace(0, len(ground_truth_xs)-1,
                                 len(ground_truth_xs)).astype(dtype=int)
            y_idxs = np.linspace(0, len(ground_truth_ys)-1,
                                 len(ground_truth_ys)).astype(dtype=int)
            z_idxs = np.linspace(0, len(ground_truth_zs) -
                                 1, z_res).astype(dtype=int)
            t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # indexed ground truth to plot
            ground_truth_xs = ground_truth_xs[x_idxs]
            ground_truth_ys = ground_truth_ys[y_idxs]
            ground_truth_zs = ground_truth_zs[z_idxs]
            ground_truth_ts = ground_truth_ts[t_idxs]
            ground_truth_values = ground_truth_values[
                x_idxs[:, None, None, None],
                y_idxs[None, :, None, None],
                z_idxs[None, None, :, None],
                t_idxs[None, None, None, :]
            ]
            ground_truth_grids = ground_truth_values

            xs = ground_truth_xs
            ys = ground_truth_ys
            zs = ground_truth_zs
        else:
            ground_truth_grids = None
            resolution = 512
            xs = np.linspace(*dynamics.state_test_range()
                             [plot_config['x_axis_idx']], resolution)
            ys = np.linspace(*dynamics.state_test_range()
                             [plot_config['y_axis_idx']], resolution)
            zs = np.linspace(*dynamics.state_test_range()
                             [plot_config['z_axis_idx']], z_res)

        xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
        value_grids = np.zeros((len(zs), len(xs), len(ys)))
        for i in range(len(zs)):
            coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
            coords[:, 0] = dataset.tMax
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            if dataset.dynamics.state_dim > 2:
                coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

            model_results = model(
                {'coords': dataset.dynamics.coord_to_input(coords.cuda())})
            values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
            ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
            value_grids[i] = values.reshape(len(xs), len(ys))

        fig = plt.figure()
        fig.suptitle(plot_config['state_slices'], fontsize=8)
        x_min, x_max = dataset.dynamics.state_test_range()[
            plot_config['x_axis_idx']]
        y_min, y_max = dataset.dynamics.state_test_range()[
            plot_config['y_axis_idx']]

        for i in range(len(zs)):
            values = value_grids[i]

            # learned BRT and recovered BRT
            ax = fig.add_subplot(1, len(zs), (i+1))
            ax.set_title('%s = %0.2f' % (
                plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

            image = np.full((*values.shape, 3), 255, dtype=int)
            BRT = values < 0
            recovered_BRT = values < delta_level

            if dynamics.set_mode == 'reach':
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                image[recovered_BRT] = np.array([155, 241, 249])
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)
            else:
                image[recovered_BRT] = np.array([155, 241, 249])
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                # overlay recovered border over learned BRT
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)

            ax.imshow(image.transpose(1, 0, 2), origin='lower',
                      extent=(x_min, x_max, y_min, y_max))

            ax.set_xlabel(plot_config['state_labels']
                          [plot_config['x_axis_idx']])
            ax.set_ylabel(plot_config['state_labels']
                          [plot_config['y_axis_idx']])
            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.tick_params(labelsize=6)
            if i != 0:
                ax.set_yticks([])
        return fig

    def overlay_ground_truth(self, image, z_idx, ground_truth_grids):
        thickness = max(0, image.shape[0] // 120 - 1)
        ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
        ground_truth_brts = ground_truth_grid < 0
        for x in range(ground_truth_brts.shape[0]):
            for y in range(ground_truth_brts.shape[1]):
                if not ground_truth_brts[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_brts.shape[0] and neighbor[1] < ground_truth_brts.shape[1]:
                        if not ground_truth_brts[neighbor]:
                            image[x-thickness:x+1, y-thickness:y +
                                  1+thickness] = np.array([50, 50, 50])
                            break

    def overlay_border(self, image, set, color):
        thickness = max(0, image.shape[0] // 120 - 1)
        for x in range(set.shape[0]):
            for y in range(set.shape[1]):
                if not set[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                        if not set[neighbor]:
                            image[x-thickness:x+1, y -
                                  thickness:y+1+thickness] = color
                            break

    # def post_training_CSL(self,loss_fn, clip_grad, use_lbfgs,
    #     val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
    #     use_CSL, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size):

    #             last_CSL_epoch = epoch

    #             # generate CSL datasets
    #             self.model.eval()

    #             CSL_dataset = scenario_optimization(
    #                 model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_samples, 100000), sample_batch_size=min(10*num_CSL_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 max_scenarios=num_CSL_samples, tStart_generator=lambda n: torch.zeros(
    #                     n).uniform_(self.dataset.tMin, CSL_tMax)
    #             )
    #             CSL_coords = torch.cat(
    #                 (CSL_dataset['times'].unsqueeze(-1), CSL_dataset['states']), dim=-1)
    #             CSL_costs = CSL_dataset['costs']

    #             num_CSL_val_samples = int(0.1*num_CSL_samples)
    #             CSL_val_dataset = scenario_optimization(
    #                 model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 max_scenarios=num_CSL_val_samples, tStart_generator=lambda n: torch.zeros(
    #                     n).uniform_(self.dataset.tMin, CSL_tMax)
    #             )
    #             CSL_val_coords = torch.cat(
    #                 (CSL_val_dataset['times'].unsqueeze(-1), CSL_val_dataset['states']), dim=-1)
    #             CSL_val_costs = CSL_val_dataset['costs']

    #             CSL_val_tMax_dataset = scenario_optimization(
    #                 model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=self.dataset.tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 # no tStart_generator, since I want all tMax times
    #                 max_scenarios=num_CSL_val_samples
    #             )
    #             CSL_val_tMax_coords = torch.cat(
    #                 (CSL_val_tMax_dataset['times'].unsqueeze(-1), CSL_val_tMax_dataset['states']), dim=-1)
    #             CSL_val_tMax_costs = CSL_val_tMax_dataset['costs']

    #             self.model.train()

    #             # CSL optimizer
    #             CSL_optim = torch.optim.Adam(
    #                 lr=csl_lr, params=self.model.parameters())

    #             # initial CSL val loss
    #             CSL_val_results = self.model(
    #                 {'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.cuda())})
    #             CSL_val_preds = self.dataset.dynamics.io_to_value(
    #                 CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #             CSL_val_errors = CSL_val_preds - CSL_val_costs.cuda()
    #             CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
    #             CSL_initial_val_loss = CSL_val_loss
    #             wandb.log({
    #                 "step": epoch,
    #                 "CSL_val_loss": CSL_val_loss.item()
    #             })

    #             # initial self-supervised learning (SSL) val loss
    #             # right now, just took code from dataio.py and the SSL training loop above; TODO: refactor all this for cleaner modular code
    #             CSL_val_states = CSL_val_coords[..., 1:].cuda()
    #             CSL_val_dvs = self.dataset.dynamics.io_to_dv(
    #                 CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #             CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(
    #                 CSL_val_states)
    #             if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                 CSL_val_reach_values = self.dataset.dynamics.reach_fn(
    #                     CSL_val_states)
    #                 CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(
    #                     CSL_val_states)
    #             # assumes time unit in dataset (model) is same as real time units
    #             CSL_val_dirichlet_masks = CSL_val_coords[:, 0].cuda(
    #             ) == self.dataset.tMin
    #             if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                 SSL_val_losses = loss_fn(
    #                     CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
    #             elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                 SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:],
    #                                             CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
    #             else:
    #                 NotImplementedError
    #             # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
    #             SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean()
    #             wandb.log({
    #                 "step": epoch,
    #                 "SSL_val_loss": SSL_val_loss.item()
    #             })

    #             # CSL training loop
    #             for CSL_epoch in tqdm(range(max_CSL_epochs)):
    #                 CSL_idxs = torch.randperm(num_CSL_samples)
    #                 for CSL_batch in range(math.ceil(num_CSL_samples/CSL_batch_size)):
    #                     CSL_batch_idxs = CSL_idxs[CSL_batch *
    #                                                 CSL_batch_size:(CSL_batch+1)*CSL_batch_size]
    #                     CSL_batch_coords = CSL_coords[CSL_batch_idxs]

    #                     CSL_batch_results = self.model(
    #                         {'coords': self.dataset.dynamics.coord_to_input(CSL_batch_coords.cuda())})
    #                     CSL_batch_preds = self.dataset.dynamics.io_to_value(
    #                         CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
    #                     CSL_batch_costs = CSL_costs[CSL_batch_idxs].cuda()
    #                     CSL_batch_errors = CSL_batch_preds - CSL_batch_costs
    #                     CSL_batch_loss = CSL_loss_weight * \
    #                         torch.mean(torch.pow(CSL_batch_errors, 2))

    #                     CSL_batch_states = CSL_batch_coords[..., 1:].cuda()
    #                     CSL_batch_dvs = self.dataset.dynamics.io_to_dv(
    #                         CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
    #                     CSL_batch_boundary_values = self.dataset.dynamics.boundary_fn(
    #                         CSL_batch_states)
    #                     if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                         CSL_batch_reach_values = self.dataset.dynamics.reach_fn(
    #                             CSL_batch_states)
    #                         CSL_batch_avoid_values = self.dataset.dynamics.avoid_fn(
    #                             CSL_batch_states)
    #                     # assumes time unit in dataset (model) is same as real time units
    #                     CSL_batch_dirichlet_masks = CSL_batch_coords[:, 0].cuda(
    #                     ) == self.dataset.tMin
    #                     if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                         SSL_batch_losses = loss_fn(
    #                             CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_dirichlet_masks)
    #                     elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                         SSL_batch_losses = loss_fn(
    #                             CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_reach_values, CSL_batch_avoid_values, CSL_batch_dirichlet_masks)
    #                     else:
    #                         NotImplementedError
    #                     # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_batch_dirichlet_masks == False))
    #                     SSL_batch_loss = SSL_batch_losses['diff_constraint_hom'].mean(
    #                     )

    #                     CSL_optim.zero_grad()
    #                     SSL_batch_loss.backward(retain_graph=True)
    #                     # no adjust_relative_grads, because I assume even with adjustment, the diff_constraint_hom remains unaffected and the only other loss (dirichlet) is zero
    #                     if (not use_lbfgs) and clip_grad:
    #                         if isinstance(clip_grad, bool):
    #                             torch.nn.utils.clip_grad_norm_(
    #                                 self.model.parameters(), max_norm=1.)
    #                         else:
    #                             torch.nn.utils.clip_grad_norm_(
    #                                 self.model.parameters(), max_norm=clip_grad)
    #                     CSL_batch_loss.backward()
    #                     CSL_optim.step()

    #                 # evaluate on CSL_val_dataset
    #                 CSL_val_results = self.model(
    #                     {'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.cuda())})
    #                 CSL_val_preds = self.dataset.dynamics.io_to_value(
    #                     CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_errors = CSL_val_preds - CSL_val_costs.cuda()
    #                 CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))

    #                 CSL_val_states = CSL_val_coords[..., 1:].cuda()
    #                 CSL_val_dvs = self.dataset.dynamics.io_to_dv(
    #                     CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(
    #                     CSL_val_states)
    #                 if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                     CSL_val_reach_values = self.dataset.dynamics.reach_fn(
    #                         CSL_val_states)
    #                     CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(
    #                         CSL_val_states)
    #                 # assumes time unit in dataset (model) is same as real time units
    #                 CSL_val_dirichlet_masks = CSL_val_coords[:, 0].cuda(
    #                 ) == self.dataset.tMin
    #                 if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                     SSL_val_losses = loss_fn(
    #                         CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
    #                 elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                     SSL_val_losses = loss_fn(
    #                         CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
    #                 else:
    #                     raise NotImplementedError
    #                 # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
    #                 SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean(
    #                 )

    #                 CSL_val_tMax_results = self.model(
    #                     {'coords': self.dataset.dynamics.coord_to_input(CSL_val_tMax_coords.cuda())})
    #                 CSL_val_tMax_preds = self.dataset.dynamics.io_to_value(
    #                     CSL_val_tMax_results['model_in'], CSL_val_tMax_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_tMax_errors = CSL_val_tMax_preds - CSL_val_tMax_costs.cuda()
    #                 CSL_val_tMax_loss = torch.mean(
    #                     torch.pow(CSL_val_tMax_errors, 2))

    #                 # log CSL losses, recovered_safe_set_fracs
    #                 if self.dataset.dynamics.set_mode == 'reach':
    #                     CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_batch_costs.cuda() < 0) / len(CSL_batch_preds)
    #                     if len(CSL_batch_preds[CSL_batch_costs.cuda() > 0]) > 0:
    #                         CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds < torch.min(
    #                             CSL_batch_preds[CSL_batch_costs.cuda() > 0])) / len(CSL_batch_preds)
    #                     else:
    #                         CSL_train_batch_recovered_safe_set_frac = torch.FloatTensor(
    #                             1)
    #                     CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_costs.cuda() < 0) / len(CSL_val_preds)
    #                     CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds < torch.min(
    #                         CSL_val_preds[CSL_val_costs.cuda() > 0])) / len(CSL_val_preds)
    #                     CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_tMax_costs.cuda() < 0) / len(CSL_val_tMax_preds)
    #                     CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds < torch.min(
    #                         CSL_val_tMax_preds[CSL_val_tMax_costs.cuda() > 0])) / len(CSL_val_tMax_preds)
    #                 elif self.dataset.dynamics.set_mode == 'avoid':
    #                     CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_batch_costs.cuda() > 0) / len(CSL_batch_preds)
    #                     # if len(CSL_batch_preds[CSL_batch_costs.cuda() < 0])>0:
    #                     #     CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(CSL_batch_preds[CSL_batch_costs.cuda() < 0])) / len(CSL_batch_preds)
    #                     # else:
    #                     #     CSL_train_batch_recovered_safe_set_frac=torch.FloatTensor(1)
    #                     CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(
    #                         CSL_batch_preds[CSL_batch_costs.cuda() < 0])) / len(CSL_batch_preds)

    #                     CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_costs.cuda() > 0) / len(CSL_val_preds)
    #                     CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds > torch.max(
    #                         CSL_val_preds[CSL_val_costs.cuda() < 0])) / len(CSL_val_preds)
    #                     CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_tMax_costs.cuda() > 0) / len(CSL_val_tMax_preds)
    #                     CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds > torch.max(
    #                         CSL_val_tMax_preds[CSL_val_tMax_costs.cuda() < 0])) / len(CSL_val_tMax_preds)

    #                     wandb.log({
    #                         "step": epoch+(CSL_epoch+1)*int(0.5*epochs_til_CSL/max_CSL_epochs),
    #                         "CSL_train_batch_loss": CSL_batch_loss.item(),
    #                         "SSL_train_batch_loss": SSL_batch_loss.item(),
    #                         "CSL_val_loss": CSL_val_loss.item(),
    #                         "SSL_val_loss": SSL_val_loss.item(),
    #                         "CSL_val_tMax_loss": CSL_val_tMax_loss.item(),
    #                         "CSL_train_batch_theoretically_recoverable_safe_set_frac": CSL_train_batch_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_val_theoretically_recoverable_safe_set_frac": CSL_val_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_val_tMax_theoretically_recoverable_safe_set_frac": CSL_val_tMax_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_train_batch_recovered_safe_set_frac": CSL_train_batch_recovered_safe_set_frac.item(),
    #                         "CSL_val_recovered_safe_set_frac": CSL_val_recovered_safe_set_frac.item(),
    #                         "CSL_val_tMax_recovered_safe_set_frac": CSL_val_tMax_recovered_safe_set_frac.item(),
    #                     })

    #             if not (epoch+1) % epochs_til_checkpoint:
    #                 # Saving the optimizer state is important to produce consistent results
    #                 checkpoint = {
    #                     'epoch': epoch+1,
    #                     'model': self.model.state_dict(),
    #                     'optimizer': optim.state_dict()}
    #                 torch.save(checkpoint,
    #                            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
    #                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
    #                            np.array(train_losses))
    #                 self.validate(
    #                     epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
    #                     x_resolution=val_x_resolution, y_resolution=val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

    #     torch.save(checkpoint, os.path.join(
    #         checkpoints_dir, 'model_CSL.pth'))


class DeepReach(Experiment):
    def init_special(self):
        pass
