import sys
import os
import numpy as np
import math
import torch
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import get_particle_positions
from random_walkers_pytorch import boundary_asserts
#######################################################
"""
This function creates (input, output) pairs for faces
"""
@torch.no_grad()
def get_particle_data(N_cell_tnsr, n_hist_steps,
                      dx, dt, left_boundary,
                      right_boundary, half_window, nmoves):
    #### Check if it is a periodic_boundary
    periodic_boundary = boundary_asserts(left_boundary,
                                         right_boundary)
    ncells = N_cell_tnsr.size(0)
    len_system = ncells*dx
    # Number of unique faces
    nfaces = ncells
    if not periodic_boundary:
        nfaces += 1

    # Ensuring that atleast 1 particle exists in the system
    if torch.sum(N_cell_tnsr) == 0:
        N_cell_tnsr[0] = 1

    initial_pos = get_particle_positions(N_cell_tnsr,dx)
    # total data
    density_data =  torch.zeros((n_hist_steps+1,ncells))
    density_data[0,:] = N_cell_tnsr
    # total flux data
    if n_hist_steps > 0:
        flux_data = torch.zeros(n_hist_steps, nfaces)
    for jj in range(1,n_hist_steps+1):
        initial_pos, density, flux = random_walk_v2(ncells, nmoves, dt, initial_pos,
                                                 left_boundary, right_boundary,
                                                 len_system = len_system)
        density_data[jj,:] = density
        if n_hist_steps > 0:
            flux_data[jj-1,:] = flux

    # Do the final step
    _, _, flux = random_walk_v2(ncells, nmoves, dt, initial_pos,
                                left_boundary, right_boundary,
                                len_system = len_system)

    n_batch_size = ncells
    ### In non-periodic case the first and last cells are ghost cells
    ### so need to ignore the first and last faces
    if not periodic_boundary:
        n_batch_size -= 1

    if periodic_boundary:
        # Apply circular padding for density data
        pad_mdl = torch.nn.CircularPad1d(half_window)
        input_batch_N = pad_mdl(density_data)
    else:
        input_batch_N = density_data.clone()

    input_batch_N = input_batch_N.unfold(-1,2*half_window,1)

    if periodic_boundary:
        # Need to narrow dim=1 as duplicate exists at the end
        input_batch_N = torch.narrow(input_batch_N,1,0,n_batch_size)

    # Need to swap dims 0 and 1
    input_batch_N = torch.swapdims(input_batch_N,0,1)

    if n_hist_steps > 0:
        # Unsqueeze last dim for flux_data and swap 0 and -1
        input_batch_F = flux_data.unsqueeze(0)
        input_batch_F = torch.swapdims(input_batch_F, 0, -1)
        ## Ignore first and last faces for non-periodic
        if not periodic_boundary:
            input_batch_F = torch.narrow(input_batch_F, 0, 1,
                                         n_batch_size)

    else:
        input_batch_F = torch.empty((n_batch_size, 0, 1))

    output_batch = torch.zeros((n_batch_size, 1, 1))
    if not periodic_boundary:
        flux = torch.narrow(flux, 0, 1, n_batch_size)

    output_batch[:,0, 0] = flux[...]

    return input_batch_N, input_batch_F, output_batch

"""
This function converts density and flux data into model inputs.
Assumes density_data has a shape (n_time_steps, ncells)
Assumes flux_data has a shape    (n_time_steps-1, nfaces)
"""
@torch.no_grad()
def convert_system_data_to_model_inputs (density_data, flux_data,
                                         half_window, periodic_boundary):
    n_batch_size = density_data.size(-1)
    ### In non-periodic case the first and last cells are ghost cells
    ### so need to ignore the first and last faces
    if not periodic_boundary:
        n_batch_size -= 1

    if periodic_boundary:
        # Apply circular padding for density data
        pad_mdl = torch.nn.CircularPad1d(half_window)
        input_batch_N = pad_mdl(density_data)
    else:
        input_batch_N = density_data.clone()

    input_batch_N = input_batch_N.unfold(-1,2*half_window,1)

    if periodic_boundary:
        # Need to narrow dim=1 as duplicate exists at the end
        input_batch_N = torch.narrow(input_batch_N,1,0,n_batch_size)

    # Need to swap dims 0 and 1
    input_batch_N = torch.swapdims(input_batch_N,0,1)

    if density_data.size(-2) > 1:
        # Unsqueeze last dim for flux_data and swap 0 and -1
        input_batch_F = flux_data.unsqueeze(0)
        input_batch_F = torch.swapdims(input_batch_F, 0, -1)
        ## Ignore first and last faces for non-periodic
        if not periodic_boundary:
            input_batch_F = torch.narrow(input_batch_F, 0, 1,
                                         n_batch_size)
    else:
        input_batch_F = torch.empty((n_batch_size,0,1))

    return input_batch_N, input_batch_F

"""
This function converts density and flux data into model inputs.
Assumes density_data has a shape (batch_size, n_time_steps, ncells)
Assumes flux_data has a shape    (batch_size, n_time_steps-1, nfaces)
"""
@torch.no_grad()
def convert_batched_system_data_to_model_inputs (density_data, flux_data,
                                                 half_window, periodic_boundary):
    batch_size = density_data.size(0)
    n_cells = density_data.size(-1)
    ### In non-periodic case the first and last cells are ghost cells
    ### so need to ignore the first and last faces
    if not periodic_boundary:
        n_cells -= 1

    if periodic_boundary:
        # Apply circular padding for density data
        pad_mdl = torch.nn.CircularPad1d(half_window)
        input_batch_N = pad_mdl(density_data)
    else:
        input_batch_N = density_data.clone()

    input_batch_N = input_batch_N.unfold(-1,2*half_window,1)

    if periodic_boundary:
        # Need to narrow dim=-2 as duplicate exists at the end
        input_batch_N = torch.narrow(input_batch_N,-2,0,n_cells)

    # Need to swap dims -2, and -3
    input_batch_N = torch.swapdims(input_batch_N,-2,-3)
    ## Need to combine Batch and cell dimension
    N_shape = input_batch_N.shape
    input_batch_N = torch.reshape(input_batch_N,
                                  (-1,N_shape[-2],N_shape[-1]))

    if density_data.size(1) > 1:
        # Unsqueeze last dim for flux_data
        input_batch_F = flux_data.unsqueeze(-1)
        input_batch_F = torch.swapdims(input_batch_F, -2, -3)
        ## Ignore first and last faces for non-periodic
        if not periodic_boundary:
            input_batch_F = torch.narrow(input_batch_F, 1, 1,
                                         n_cells)
        ## Need to combine Batch and cell dimension
        F_shape = input_batch_F.shape
        input_batch_F = torch.reshape(input_batch_F,
                                      (-1,F_shape[-2],F_shape[-1]))
    else:
        input_batch_F = torch.empty((batch_size*n_cells, 0, 1))

    return input_batch_N, input_batch_F

"""
This function converts ML model's output to system data
THIS ONLY PROVIDES INCREMENT FOR EACH CELL
Expected model_ouputs shape:  (n_faces, 1)
For periodic; nfaces = ncells
For non-periodic; nfaces = ncells-1
"""
@torch.no_grad()
def convert_model_outputs_to_system_data (model_outputs,
                                          periodic_boundary):
    if periodic_boundary:
        flux_left = torch.zeros_like(model_outputs)
        flux_right = torch.zeros_like(flux_left)
        flux_left[...] = model_outputs[...]
        flux_right[:-1, 0] = model_outputs[1:,0]
        flux_right[-1,0] = model_outputs[0,0]
    else:
        flux_left = model_outputs.clone()
        flux_left = torch.nn.functional.pad(flux_left, (0,0,1,0),
                                            "constant",0)
        flux_right = model_outputs.clone()
        flux_right = torch.nn.functional.pad(flux_right, (0,0,0,1),
                                             "constant", 0)
    return (flux_left - flux_right).squeeze(-1)

"""
This function converts ML model's output to BATCHED system data
THIS ONLY PROVIDES INCREMENT FOR EACH CELL
Expected model_ouputs shape:  (Batch*n_faces, 1)
For periodic; nfaces = ncells
For non-periodic; nfaces = ncells-1
"""
@torch.no_grad()
def convert_model_outputs_to_batched_system_data (model_outputs, batch_size,
                                                  periodic_boundary):
    model_outputs = torch.reshape(model_outputs,(batch_size, -1, 1))

    if periodic_boundary:
        flux_left = torch.zeros_like(model_outputs)
        flux_right = torch.zeros_like(flux_left)
        flux_left[...] = model_outputs[...]
        flux_right[:, :-1, 0] = model_outputs[:, 1:, 0]
        flux_right[:, -1, 0] = model_outputs[:, 0, 0]
    else:
        flux_left = model_outputs.clone()
        flux_left = torch.nn.functional.pad(flux_left, (0,0,1,0),
                                            "constant",0)
        flux_right = model_outputs.clone()
        flux_right = torch.nn.functional.pad(flux_right, (0,0,0,1),
                                             "constant", 0)

    return (flux_left - flux_right).squeeze(-1)

"""
This function adds net particles crossing from Left to right
due to external potential.
input = old_N shape (..., ncells)
"""
def add_extrnl_pot_net_ptcls_crsng(old_N, dt, alpha, beta,
                            gamma, len_system, periodic_boundary):
    ncells = old_N.size(-1)
    dx = len_system/ncells
    face_pos = torch.linspace(0., len_system, ncells+1)
    ## The calculation is being done in terms of density first
    ## and then converted to Number of particles later
    dens_left = torch.zeros((*old_N.shape[:-1],ncells+1))
    dens_right = torch.zeros_like(dens_left)

    dens_left[...,1:] = old_N[...,:]
    dens_right[...,:-1] = old_N[...,:]
    if periodic_boundary:
        dens_left[...,0] = old_N[..., -1]
        dens_right[...,-1] = old_N[...,0]

    dens_left /= dx
    dens_right /= dx

    dist_1 = face_pos - alpha
    dist_2 = face_pos - beta
    if periodic_boundary:
        dist_1 = torch.where(dist_1 > 0.5*len_system,
                               dist_1-len_system, dist_1)
        dist_1 = torch.where(dist_1 < -0.5*len_system,
                               dist_1 + len_system, dist_1)

        dist_2 = torch.where(dist_2 > 0.5*len_system,
                               dist_2 - len_system, dist_2)
        dist_2 = torch.where(dist_2 < -0.5*len_system,
                               dist_2 + len_system, dist_2)

    flux_pot = 2.*dist_1*dist_2*dist_2
    flux_pot += 2.*dist_2*dist_1*dist_1

    if old_N.ndim == 2:
        flux_pot = flux_pot.unsqueeze(0)
        flux_pot = flux_pot.expand(old_N.size(0),-1).clone()

    flux_pot *= (dens_left+dens_right)/(2.*gamma)
    #### The change is sign is because particles moving
    #### left to right are considered positive
    flux_pot *= (-dt/dx)
    #### Converting from density to number of particles
    ### Additional operation but kept for clarity
    flux_pot *= dx
    #### While not truly needed; just making sure
    if periodic_boundary:
        flux_pot[...,-1] = flux_pot[...,0]
    #### Use the net particle crossing to get change in density
    flux_left = torch.narrow(flux_pot, -1, 0, ncells)
    flux_right = torch.narrow(flux_pot, -1, 1, ncells)
    ########################################################
    if periodic_boundary:
        flux_pot = torch.narrow(flux_pot, -1, 0, ncells)

    return (flux_left - flux_right), flux_pot

"""
This function applies corrections to boundary cells
when there are non-periodic boundaries
Assumes that N_new shape = (...,ncells)
"""
def apply_N_boundary_effects(N_new, left_boundary,
                             right_boundary):

    periodic_boundary = boundary_asserts(left_boundary,
                                         right_boundary)
    if not periodic_boundary:
        if left_boundary[1] > 0:
            N_new[...,0] = np.random.poisson(float(left_boundary[1]))
        else:
            N_new[...,0] = 0.

        if right_boundary[1] > 0:
            N_new[...,-1] = np.random.poisson(float(right_boundary[1]))
        else:
            N_new[...,-1] = 0.
    return N_new

def test_get_particle_data():
    N_min = 0
    N_max = 50
    dx = 1.0/100
    ncells = 10
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["put", 10]
    right_boundary = ["put", 0]
    half_window = 1
    nmoves = 1

    N_cell_batch = torch.randint(low=N_min, high=N_max+1,
                                 size=(ncells,),dtype=torch.float32)
    print(N_cell_batch)
    input_batch_N, input_batch_F, output_batch = get_particle_data(N_cell_batch, 2,
                                                    dx, dt, left_boundary,
                                                    right_boundary, half_window, nmoves)
    print(input_batch_N.shape)
    print(input_batch_F.shape)
    print(output_batch.shape)



if __name__ == "__main__":
    test_get_particle_data()
    #N_min = 0
    #N_max = 50
    #dx = 1.0/100
    #ncells = 100
    #len_system=ncells*dx
    #dt = 0.03*dx*dx
    #left_boundary  = ["periodic", 0]
    #right_boundary = ["periodic", 0]
    #half_window = 1
    #batch_size=ncells

    #N_cell_batch = torch.randint(low=N_min, high=N_max+1,
    #                             size=(10, batch_size,),dtype=torch.float32)

    ##### get_particle_data function testing ##########################
    #input_batch_N, input_batch_F, output_batch = get_particle_data(N_cell_batch, 2,
    #                                                dx, dt, left_boundary,
    #                                                right_boundary, half_window, 1)
    ##print(input_batch_N)
    ##print(input_batch_F)
    #print(output_batch.shape)

    #N_left_t  = torch.narrow(input_batch,-1,-half_window-1,1)
    #N_right_t = torch.narrow(input_batch,-1,-half_window,1)
    #print(N_left_t)
    #print(N_right_t)
    ###################################################################
    ####### convert_system_data_to_model_inputs test ##################
    #density_data = torch.randint(low=N_min, high=N_max+1,
    #                             size=(2, 3, batch_size,),dtype=torch.float32)
    #print(density_data)
    #input_batch_N, input_batch_F = convert_batched_system_data_to_model_inputs(
    #                                              density_data, density_data,
    #                                                          half_window)
    ##print(input_batch_N.shape)
    ##print(input_batch_F.shape)
    #print(input_batch_N)
    #print(input_batch_F)
    ####################################################################
    ####### convert_model_outputs_to_system_data test ###################
    #N_cell_batch = torch.randint(low=N_min, high=N_max+1,
    #                             size=(batch_size,),dtype=torch.float32)
    #initial_pos = get_particle_positions(N_cell_batch,dx)

    #_, new_density, flux = random_walk_v2(ncells, 40, dt, initial_pos,
    #                                  left_boundary, right_boundary,
    #                                  len_system = len_system)

    #increment_density = convert_model_outputs_to_batched_system_data(
    #                                  flux.clone().unsqueeze(-1), ncells)

    #print(new_density)
    #print(N_cell_batch+increment_density)
    #####################################################################
    #N_cell_batch = torch.randint(low=N_min, high=N_max+1,
    #                             size=(1, ncells,),dtype=torch.float32)
    #pot_change = add_extrnl_pot_net_ptcls_crsng(N_cell_batch, dt, 0.3, 0.7,
    #                        5.e-4, len_system, True)

    #print(pot_change)
