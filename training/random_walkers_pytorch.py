import numpy as np
import torch

# get_density has nothing to do with boundary type
#def get_density(cell_centers, pos):
#    # Evaluating density of particles
#    # pos should be periodic boundary shifted
#    ncells = torch.numel(cell_centers)
#    dx = cell_centers[1]-cell_centers[0]
#    density, _ = torch.histogram(pos, bins=ncells,
#                                 range=(0.,cell_centers[-1]+0.5*dx),
#                                 density=False)
#    return density

def get_density(cell_centers, pos):
    # Evaluating density of particles
    # pos should be periodic boundary shifted
    ncells = torch.numel(cell_centers)
    dx = cell_centers[1]-cell_centers[0]
    density = torch.histc(pos, bins=ncells,
                          min=0.,max=cell_centers[-1]+0.5*dx)
    return density

# If flux at a face is positive, then net particles are moving from left to right
# For the periodic boundary:
# 1)the first element in flux corresponds to the periodic face
# 2)the flux at that face is a sum of the two NON-UNIQUE boundary faces
# The size of flux is ncells+1 if non-periodic and ncells if periodic
def get_flux(face_centers, pos_0, jump, periodic_boundary):
    # Evaluating flux at faces
    num_par = torch.numel(pos_0)
    nfaces = torch.numel(face_centers)
    if periodic_boundary:
        flux = torch.zeros(nfaces-1)
    else:
        flux = torch.zeros(nfaces)
    face_centers = face_centers.unsqueeze(0)
    face_centers = face_centers.expand((num_par,-1))

    # Do NOT apply periodic shift
    pos_1 = pos_0 + jump
    # Get position at step i
    pos_0 = pos_0.unsqueeze(-1)
    pos_0 = pos_0.expand((-1,nfaces))

    pos_1 = pos_1.unsqueeze(-1)
    pos_1 = pos_1.expand((-1,nfaces))

    # particles going left to right
    cond_1_1 = (face_centers-pos_0) > 0.
    cond_1_2 = (face_centers-pos_1) <= 0.
    # Logical AND
    cond_1 = torch.logical_and(cond_1_1,cond_1_2)
    flux_Left_Right = torch.sum(cond_1,dim=0)

    # particles going right to left
    cond_2_1 = (face_centers-pos_0) <= 0.
    # EQUAL TO SIGN SHOULD NOT APPEAR
    cond_2_2 = (face_centers-pos_1) > 0.
    # Logical AND
    cond_2 = torch.logical_and(cond_2_1,cond_2_2)
    flux_Right_Left = torch.sum(cond_2,dim=0)

    if periodic_boundary:
        flux[1:] = flux_Left_Right[1:-1] - flux_Right_Left[1:-1]
        # For the periodic face, sum both ends
        flux[0] = (flux_Left_Right[0] - flux_Right_Left[0]
                  + flux_Left_Right[-1] - flux_Right_Left[-1])
    else:
        flux = flux_Left_Right - flux_Right_Left

    return flux

def boundary_variable_asserts(boundary_var):
    assert isinstance(boundary_var,list)
    assert len(boundary_var) == 2
    assert isinstance(boundary_var[0],str)
    assert isinstance(boundary_var[1],int)
    assert boundary_var[0] in ["periodic","put","ignore"]

def boundary_asserts(left_boundary,right_boundary):
    boundary_variable_asserts(left_boundary)
    boundary_variable_asserts(right_boundary)
    periodic_boundary = False
    if left_boundary[0] == "periodic":
        assert right_boundary[0] == "periodic"
        periodic_boundary = True
    if right_boundary[0] == "periodic":
        assert left_boundary[0] == "periodic"
        periodic_boundary = True
    return periodic_boundary

# This function takes the updated particles position
# and applies the boundary condition
def apply_boundary_effects(new_pos,len_system,ncells,
                           left_boundary,right_boundary):
    periodic_boundary = boundary_asserts(left_boundary,right_boundary)
    if periodic_boundary:
        # Apply periodic effect
        # lower bound
        new_pos = torch.where(new_pos < 0., new_pos + len_system, new_pos)
        # upper bound
        new_pos = torch.where(new_pos >= len_system,
                              new_pos - len_system, new_pos)
        return new_pos
    # Coming here means non-periodic boundaries
    # Let us consider left boundary first
    dx = len_system/ncells
    if left_boundary[0] == "put":
        # Remove existing particles in the first cell
        new_pos = new_pos[new_pos > dx]
        n_poisson = int(np.random.poisson(lam=left_boundary[1], size=1))
        if n_poisson > 0:
            # Create new_particles to put in first cell
            put_ptcls = torch.rand(n_poisson)*dx
            put_ptcls = torch.clamp(put_ptcls,1.0e-3*dx,0.999*dx)
            new_pos = torch.cat([new_pos,put_ptcls],dim=0)
    elif left_boundary[0] == "ignore":
        # Remove particles that are to the left of left boundary
        new_pos = new_pos[new_pos > 0.]

    # Now considering right boundary
    if right_boundary[0] == "put":
        # Remove existing particles in the last cell
        new_pos = new_pos[new_pos < len_system - dx]
        # Create new_particles to put in last cell
        n_poisson = int(np.random.poisson(lam=right_boundary[1], size=1))
        if n_poisson > 0:
            put_ptcls = torch.rand(n_poisson)*dx
            put_ptcls = torch.clamp(put_ptcls,1.0e-3*dx,0.999*dx)
            put_ptcls += (len_system-dx)
            new_pos = torch.cat([new_pos,put_ptcls],dim=0)
    elif right_boundary[0] == "ignore":
        # Remove particles that are to the right of right boundary
        new_pos = new_pos[new_pos < len_system]
    return new_pos

# Currently only coded potential of the form V(x) = (x-alpha)^2 (x-beta)^2
def add_external_potential_effect(pos, dt, alpha, beta, gamma,
                                  len_system, periodic_boundaries):
    dist_1 = pos-alpha # Distance from first well
    dist_2 = pos-beta  # Distance from second well

    if periodic_boundaries:
        dist_1 = torch.where(dist_1 > 0.5*len_system,
                             dist_1-len_system, dist_1)
        dist_1 = torch.where(dist_1 < -0.5*len_system,
                             dist_1+len_system, dist_1)

        dist_2 = torch.where(dist_2 > 0.5*len_system,
                             dist_2-len_system, dist_2)
        dist_2 = torch.where(dist_2 < -0.5*len_system,
                             dist_2+len_system, dist_2)


    pot_effect = 2.*dist_1*dist_2*dist_2
    pot_effect += 2.*dist_2*dist_1*dist_1
    pot_effect *= (-dt/gamma)

    return pot_effect

def random_walk_just_evolve(ncells, nmoves, dt, initial_pos,
                            left_boundary, right_boundary,
                            len_system = 1.0, add_potential=False,
                            alpha=0.3, beta=0.7, gamma=5.0e-4):
    # Some asserts related to boundaries
    ##########################################
    periodic_boundary = boundary_asserts(left_boundary,right_boundary)
    ##########################################
    # total number of particles initially
    num_par = torch.numel(initial_pos)
    # Initial position of particles
    # ASSUMING SYSTEM IS OF UNIT LENGTH
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,(ncells-0.5)*dx,ncells)
    face_centers = torch.linspace(0,len_system,ncells+1)

    # std of jump
    jump_std = torch.tensor(dt)
    jump_std = torch.sqrt(jump_std)
    # Particles cannot jump more than dx
    jump_lim = torch.tensor(dx)
    # Temporally evolving particle positions
    pos = initial_pos
    for i in range(1,nmoves+1):
        num_par = torch.numel(pos)
        jump = torch.zeros_like(pos)
        ### External potential part ############
        if add_potential:
            jump += add_external_potential_effect(pos, dt, alpha,
                                                  beta, gamma,
                                                  len_system,
                                                  periodic_boundary) 
        ### Gaussian Noise part ##########
        jump += jump_std*torch.randn(num_par)
        # Clamp the jump to cell size
        jump = torch.clamp(jump,min=(-1+1e-6)*jump_lim,max=(1-1e-6)*jump_lim)
        # Update particle position with Gaussian noise part
        pos += jump
        ##################################################################
        # Apply boundary effects
        pos = apply_boundary_effects(pos,len_system,ncells,
                                         left_boundary,right_boundary)
    return pos

# This function returns density and flux for all the nmoves
# For periodic_boundary it additionally also returns
# the positions of particles for all nmoves
def random_walk_v1(ncells, nmoves, dt, initial_pos,
                left_boundary, right_boundary,
                len_system = 1.0, add_potential=False,
                alpha=0.3, beta=0.7, gamma=5.0e-4):
    # Some asserts related to boundaries
    ##########################################
    periodic_boundary = boundary_asserts(left_boundary,right_boundary)
    assert periodic_boundary
    # Apply boundary effects
    initial_pos = apply_boundary_effects(initial_pos,len_system,ncells,
                                     left_boundary,right_boundary)
    ##########################################
    # total number of particles initially
    num_par = torch.numel(initial_pos)
    # Tensor to keep track of particle positions
    # This can be used only when total particles is constant
    if periodic_boundary:
        particles = torch.zeros((nmoves+1,num_par))
    # Tensor to keep track of density of particles
    density = torch.zeros((nmoves+1,ncells))
    if periodic_boundary:
        # Tensor to keep track of flux;
        # Periodic case one NON-UNIQUE boundary
        flux = torch.zeros((nmoves,ncells))
    else:
        # All faces are unique in non-periodic case
        flux = torch.zeros((nmoves,ncells+1))
    # Initial position of particles
    # ASSUMING SYSTEM IS OF UNIT LENGTH
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,(ncells-0.5)*dx,ncells)
    face_centers = torch.linspace(0,len_system,ncells+1)
    if periodic_boundary:
        particles[0,:] = initial_pos
    density[0,:] = get_density(cell_centers,particles[0,:])

    # std of jump
    jump_std = torch.tensor(dt)
    jump_std = torch.sqrt(jump_std)
    # Particles cannot jump more than dx
    jump_lim = torch.tensor(dx)
    # Temporally evolving particle positions
    pos = initial_pos
    for i in range(1,nmoves+1):
        num_par = torch.numel(pos)
        jump = torch.zeros_like(pos)
        ### External potential part ############
        if add_potential:
            jump += add_external_potential_effect(pos, dt, alpha,
                                                  beta, gamma,
                                                  len_system,
                                                  periodic_boundary) 
        ##### Gaussian Noise part ################
        jump += jump_std*torch.randn(num_par)
        # Clamp the jump to cell size
        jump = torch.clamp(jump,min=(-1+1e-6)*jump_lim,max=(1-1e-6)*jump_lim)
        # Leverage pos and jump to get flux
        flux[i-1,:] = get_flux(face_centers,pos,jump,periodic_boundary)
        # Update particle position
        new_pos = pos + jump
        # Apply boundary effects
        new_pos = apply_boundary_effects(new_pos,len_system,ncells,
                                         left_boundary,right_boundary)
        if periodic_boundary:
            particles[i,:] = new_pos
        # Get density
        density[i,:] = get_density(cell_centers,new_pos)
        pos = new_pos

    if periodic_boundary:
        return particles, density, flux
    else:
        return density, flux

# This function only returns the
# final particles position, density, and flux
def random_walk_v2(ncells, nmoves, dt, initial_pos,
                left_boundary, right_boundary,
                len_system = 1.0, add_potential=False,
                 alpha=0.3, beta=0.7, gamma=5.0e-4):
    # Some asserts related to boundaries
    ##########################################
    periodic_boundary = boundary_asserts(left_boundary,right_boundary)
    ##########################################
    # Apply boundary effects
    initial_pos = apply_boundary_effects(initial_pos,len_system,ncells,
                                     left_boundary,right_boundary)
    if periodic_boundary:
        # Tensor to keep track of flux;
        # Periodic case one NON-UNIQUE boundary
        flux = torch.zeros((ncells,))
    else:
        # All faces are unique in non-periodic case
        flux = torch.zeros((ncells+1,))
    # ASSUMING SYSTEM IS OF UNIT LENGTH
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,(ncells-0.5)*dx,ncells)
    face_centers = torch.linspace(0,len_system,ncells+1)
    # std of jump
    jump_std = torch.tensor(dt)
    jump_std = torch.sqrt(jump_std)
    # Particles cannot jump more than dx
    jump_lim = torch.tensor(dx)
    for _ in range(1,nmoves+1):
        num_par = torch.numel(initial_pos)
        jump = torch.zeros_like(initial_pos)
        ### External potential part ############
        if add_potential:
            jump += add_external_potential_effect(initial_pos, dt, alpha,
                                                  beta, gamma, len_system,
                                                  periodic_boundary) 
        ### Gaussian Noise part ##########
        jump += jump_std*torch.randn(num_par)
        # Clamp the jump to cell size
        jump = torch.clamp(jump,min=(-1+1e-6)*jump_lim,max=(1-1e-6)*jump_lim)
        flux += get_flux(face_centers,initial_pos,jump,periodic_boundary)
        # Update particle position
        initial_pos += jump
        # Apply boundary effects
        initial_pos = apply_boundary_effects(initial_pos,len_system,ncells,
                                         left_boundary,right_boundary)
        density = get_density(cell_centers,initial_pos)

    return initial_pos, density, flux


# Always uniformly distributing at cell centers
def get_initial_pos(ncells,par_per_cell,len_system=1.0):
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    cell_centers = cell_centers.unsqueeze(-1)
    cell_centers = cell_centers.expand((-1,par_per_cell))
    cell_centers = torch.reshape(cell_centers,(-1,))
    return cell_centers

# Uniformly distributing along the entire system length
def get_uni_initial_pos (ncells,par_per_cell,len_system=1.0):
    num_par = ncells*par_per_cell
    initial_pos = torch.linspace(0.,len_system-1e-6,num_par)
    return initial_pos

# Well-shaped distribution of particles
def get_well_initial_pos (ncells, par_per_cell,x_1,x_2,len_system=1.0):
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    # Range where initial number of particles is zero
    x_1 *= len_system
    x_2 *= len_system
    x_1_ind = torch.searchsorted(cell_centers,x_1)-1
    x_2_ind = torch.searchsorted(cell_centers,x_2)

    N_cell_tnsr = torch.zeros_like(cell_centers)
    N_cell_tnsr[:x_1_ind+1] = 2*par_per_cell 
    N_cell_tnsr[x_2_ind:] = 2*par_per_cell 
    initial_pos = get_particle_positions(N_cell_tnsr, dx)
    return initial_pos

# Non-zero well-shaped distribution of particles
def get_nonzero_well_initial_pos (ncells, N_low, N_high, x_1, x_2,
                                  len_system=1.0):
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    # Range where initial number of particles is zero
    x_1 *= len_system
    x_2 *= len_system
    x_1_ind = torch.searchsorted(cell_centers,x_1)-1
    x_2_ind = torch.searchsorted(cell_centers,x_2)
    N_cell_tnsr = torch.zeros_like(cell_centers)
    N_cell_tnsr[:x_1_ind+1] = N_high
    N_cell_tnsr[x_2_ind:] = N_high
    N_cell_tnsr[x_1_ind+1:x_2_ind] = N_low
    initial_pos = get_particle_positions(N_cell_tnsr, dx)
    return initial_pos

def get_linear_initial_pos (ncells, left_val, right_val, len_system=1.0):
    assert left_val >= 0
    assert right_val >= 0
    vec_par_per_cell = np.linspace(int(left_val), int(right_val),
                                   ncells,dtype=int)
    lst_initial_pos = []
    dx = len_system/ncells
    for i, num_p in enumerate(vec_par_per_cell):
        pos = dx*torch.rand(num_p)
        pos = torch.clamp(pos,1.0e-3*dx, 0.999*dx)
        pos += i*dx
        lst_initial_pos.append(pos)
    return torch.cat(lst_initial_pos,)

def get_particle_positions (N_cell_tnsr, dx):
    pos_list = []
    for ii, N_in_cell in enumerate(N_cell_tnsr):
        N_val = int(N_in_cell.item())
        if N_val > 0:
            ptcls_cell = torch.rand((N_val))*dx
            ptcls_cell += ii*dx
            pos_list.append(ptcls_cell)
    return torch.cat(pos_list)

def get_particle_positions_from_list (ncells, dx, x_list, N_list):
    len_system = ncells*dx
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    x_ind = [0,]
    for x_pos in x_list:
        aa = torch.searchsorted(cell_centers,x_pos)
        x_ind.append(aa)
    x_ind.append(ncells-1)
    N_cell_tnsr = torch.zeros(ncells)
    for ii, N_val in enumerate(N_list):
        N_cell_tnsr[x_ind[ii]:x_ind[ii+1]+1] = N_val

    return get_particle_positions(N_cell_tnsr, dx)

def update_density (dens_old, flux,
                    left_boundary, right_boundary):

    periodic_boundary = boundary_asserts(left_boundary,
                                         right_boundary)
    if periodic_boundary:
        left_flux = flux.clone()
        right_flux = torch.zeros_like(dens_old)
        right_flux[:-1] = left_flux[1:]
        right_flux[-1]  = left_flux[0]
    else:
        left_flux = torch.zeros_like(dens_old)
        left_flux[:] = flux[:-1]
        right_flux = torch.zeros_like(dens_old)
        right_flux[:] = flux[1:] 

    return dens_old + left_flux - right_flux

def test_get_density ():
    ncells=20
    len_system=1.0
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    # Random cell distribution
    N_cell_tnsr = torch.randint(low=0, high=51,
                                size=(ncells,),dtype=torch.float32)
    # particle creation
    ptcl_pos = get_particle_positions(N_cell_tnsr, dx)
    # Particle to density
    density_eval = get_density(cell_centers, ptcl_pos)
    # Check
    assert torch.allclose(density_eval, N_cell_tnsr)
    # Another check
    par_per_cell = 10
    ptcl_pos = get_uni_initial_pos(ncells, par_per_cell,
                                   len_system)

    density_eval = get_density(cell_centers, ptcl_pos)
    assert torch.sum(density_eval) == ncells*par_per_cell

def test_get_flux ():
    par_per_cell = 10
    ncells = 20
    nmoves = 10
    len_system=1.0
    dx = len_system/ncells
    dt = 0.03*dx*dx
    left_boundary =  ["periodic",0]
    right_boundary = ["periodic",0]
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    face_centers = torch.linspace(0,len_system,ncells+1)

    initial_pos = get_uni_initial_pos(ncells, par_per_cell,
                                      len_system)
    old_density = get_density(cell_centers,initial_pos)

    particles,new_density_1,flux = random_walk_v2(ncells,nmoves,dt,
                                                  initial_pos,
                                                  left_boundary,
                                                  right_boundary,
                                                  len_system=len_system,
                                                  add_potential=True)

    new_density_2 = update_density(old_density, flux,
                                   left_boundary, right_boundary)

    new_density_3 = get_density(cell_centers, particles)

    assert torch.allclose(new_density_3, new_density_1)
    assert torch.allclose(new_density_1, new_density_2)
    assert torch.allclose(new_density_2, new_density_3)

def test_well ():
    ncells=100
    len_system=1.0
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    par_per_cell = 10
    x_1 = 0.25
    x_2 = 0.75
    initial_pos = get_well_initial_pos (ncells, par_per_cell,
                                        x_1, x_2,
                                        len_system=len_system)
    density = get_density(cell_centers,initial_pos)
    print(density)

def test_nonzero_well ():
    ncells=100
    len_system=1.0
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    N_low = 5
    N_high = 25
    x_1 = 0.25
    x_2 = 0.75
    initial_pos = get_nonzero_well_initial_pos (ncells, N_low, N_high,
                                                x_1, x_2,
                                                len_system=len_system)
    density = get_density(cell_centers,initial_pos)
    print(density)

if __name__ == "__main__":
    #torch.set_default_device('cuda')
    for _ in range(20):
        test_get_density()
        test_get_flux()

    #test_nonzero_well()
    #test_well()
