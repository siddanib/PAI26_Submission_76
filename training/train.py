import sys
import os
import numpy as np
import math
import torch
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
#######################################################
####### Local imports ################################
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_density
from random_walkers_pytorch import random_walk_just_evolve
from model_reflect import Flow_Transformer
from helpers_extended_domain import get_particle_data
#######################################################

def save_model (len_system, noise_std_fctr, n_layers, d_model,
                batch_size, cfg, learning_rate,
                history_length, flow, optimizer):
    torch.save({
                'len_system'          : len_system,
                'noise_stc_fctr'      : noise_std_fctr,
                'n_layers'            : n_layers,
                'd_model'             : d_model,
                'batch_size'          : batch_size,
                'act_func'            : cfg.model.act_func,
                'learning_rate'       : learning_rate,
                'history_length'      : history_length,
                'model_state_dict'    : flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(HydraConfig.get().runtime.output_dir,
                             "flow_model_checkpoint.tar")
                )
    torch.save({
                'model_state_dict'    : flow.state_dict(),
                },
                os.path.join(HydraConfig.get().runtime.output_dir,
                             "flow_model.tar")
                )

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def fhd_model_run (cfg):
    N_min  = min(cfg.n_range)
    N_max  = max(cfg.n_range)
    N_scale = cfg.n_scale
    dx = 1.0/100
    ncells = cfg.batch_size - 1
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    nmoves = cfg.nmoves # Number of steps of size dt
    cell_centers = torch.linspace(0.5*dx,(ncells-0.5)*dx,ncells)

    noise_std_fctr = 0
    half_window = cfg.model.half_window
    n_layers = cfg.model.n_layers
    d_model = cfg.model.d_model
    history_length = int(cfg.model.history_length)
    act_func     = instantiate(cfg.model.act_func)
    flow = Flow_Transformer(input_N_dim=2*half_window, input_F_dim=1,
                            d_model = d_model, nhead=cfg.model.n_head,
                            num_encoder_layers=n_layers,
                            num_decoder_layers=n_layers,
                            dropout = cfg.model.dropout,
                            max_len=50, d_embed=d_model,
                            n_layers=n_layers,act_func=act_func,
                            residual_con=cfg.model.residual_con)

    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    optimizer = torch.optim.AdamW(flow.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                    base_lr = learning_rate,
                                    max_lr=cfg.max_lr,
                                    step_size_up=2*cfg.n_iter_per_epoch)
    loss_fn = torch.nn.MSELoss()
    # Create StudentT distribution
    student_t = torch.distributions.StudentT(cfg.df, loc=0., scale=1.0)

    epoch_start = 0
    # Look if there a previous version to start
    if 'file_name' in cfg['model']:
        if cfg.model.file_name != "":
            chpt_fl = torch.load(cfg.model.file_name,weights_only=False)
            flow.load_state_dict(chpt_fl['model_state_dict'])
            epoch_start = cfg.model.epoch_start

    flow.compile()

    if cfg.model.train:
        n_iter_per_epoch = min(cfg.max_iter,cfg.n_iter_per_epoch)
        max_epoch = int(cfg.max_iter/n_iter_per_epoch)
        for epch in range(epoch_start, max_epoch):
            mean_loss = 0
            n_mn_vals = 0
            for _ in range(int(n_iter_per_epoch)+1):
                n_mn_vals += 1
                # Create the mini-batch
                par_per_cell = np.random.randint(N_min+1,N_max+1)
                # Create a uniform system and thermalize for few steps
                init_pos = get_uni_initial_pos(ncells, par_per_cell, len_system)
                init_pos = random_walk_just_evolve(ncells, 50, dt,
                                                   init_pos.clone(), left_boundary,
                                                   right_boundary, len_system)
                N_cell_batch = get_density(cell_centers, init_pos)

                # Use variable history_length
                batch_hist_len = np.random.randint(0,high=history_length+1)

                input_batch_N, input_batch_F, output_batch = get_particle_data(N_cell_batch,
                                                                        batch_hist_len,
                                                                        dx, dt, left_boundary,
                                                                        right_boundary,
                                                                        half_window, nmoves)
                
                x_1  = output_batch
                # Scale the output instead of input
                N_left_t = torch.narrow(input_batch_N, -2, -1, 1)
                N_left_t = torch.narrow(N_left_t, -1,-half_window-1,1)

                N_right_t = torch.narrow(input_batch_N, -2, -1, 1)
                N_right_t = torch.narrow(N_right_t, -1,-half_window, 1)
                # Shift the mean based on (N_left-N_right)
                x_1 -= 0.069*(N_left_t-N_right_t)
                # Change the standard deviation
                std_scale = torch.sqrt(torch.clamp(N_left_t, min=0.0)
                                     +torch.clamp(N_right_t,min=0.))
                std_scale = torch.clamp(std_scale,min=0.5)
                x_1 /= 0.2537*std_scale
                ##############################################################
                x_0 = student_t.sample(x_1.size())
                t    = torch.rand_like(x_1)
                x_t  = torch.cos(0.5*math.pi*t)*x_0 + torch.sin(0.5*math.pi*t)*x_1
                dx_t = torch.cos(0.5*math.pi*t)*x_1 - torch.sin(0.5*math.pi*t)*x_0
                dx_t *= (0.5*math.pi)
                optimizer.zero_grad()
                loss_batch = loss_fn(flow(x_t, input_batch_N/N_scale,
                                          input_batch_F/N_scale, t),
                                     dx_t)
                loss_batch.backward()
                optimizer.step()
                scheduler.step()
                mean_loss += loss_batch.item()
            mean_loss /= n_mn_vals
            print(f"Epoch: {epch}, loss: {mean_loss}")
            if epch % 10 == 0:
                save_model(len_system, noise_std_fctr, n_layers, d_model,
                           batch_size, cfg, learning_rate,
                           history_length, flow, optimizer)

        # Final model
        save_model(len_system, noise_std_fctr, n_layers, d_model,
                   batch_size, cfg, learning_rate,
                   history_length, flow, optimizer)
    else:
        chpt_fl = torch.load(cfg.model.file_name)
        flow.load_state_dict(chpt_fl['model_state_dict'])
    return None

if __name__ == "__main__":
    fhd_model_run()
