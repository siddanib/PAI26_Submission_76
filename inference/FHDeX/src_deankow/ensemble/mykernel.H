#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>
#include <ExternalPotential.H>
#include <FluxMode.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_phi (int i, int j, int k,
               amrex::Array4<amrex::Real> const& phi,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
               amrex::Real /*npts_scale*/, int Ncomp,
               ExternalPotential const& external_potential)
{
    using amrex::Real;
    amrex::ignore_unused(external_potential);
    Real cellvol = dx[0]*dx[1];

#if(AMREX_SPACEDIM > 2)
    cellvol *=dx[2];
#endif


#if 0
    Real rad = Real(0.1);
    Real xcent = Real(0.5);
    Real ycent = Real(0.5);
    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0] - xcent;
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1] - ycent;

#if (AMREX_SPACEDIM > 2)
    Real zcent = Real(0.5);
    Real z = prob_lo[2] + (k+Real(0.5)) * dx[2] - zcent;
    Real r2 = (x*x + y*y + z*z) / (rad*rad);
#else
    Real z = Real(0.);
    Real r2 = (x*x + y*y) / (rad*rad);
#endif

    // Gaussian
    phi(i,j,k,0) = Real(1.) + std::exp(-r2);
#endif

#if 0
    // Jump in x-direction
    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
    if ( x > 0.5) {
        phi(i,j,k,0) = 1.5*npts_scale;
    } else {
        phi(i,j,k,0) = 0.5*npts_scale;
    }
#endif

#if 0
    // Delta function
    if (i==4 && j ==4)
    {
       phi(i,j,k,0) = 1./(dx[0]*dx[1]);
    }
    else
    {
       phi(i,j,k,0) = 0;;
    }
#endif

#if 0
    // 2D Test for multilevel
    Real rad = Real(0.2);
    Real xcent = Real(0.5);
    Real ycent = Real(0.5);
    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0] - xcent;
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1] - ycent;
    Real r2 = (x*x + y*y);
    r2 = 4.*(x-y)*(x-y)+ .25*(x+y)*(x+y);
#if (AMREX_SPACEDIM == 3)
    Real zcent = Real(0.75);
    Real z = prob_lo[2] + (k+Real(0.5)) * dx[2] - zcent;
    r2 = (x*x + y*y+z*z);
#endif

    if ( r2 > rad*rad) {
        phi(i,j,k,0) = 15./cellvol;
    } else {
        phi(i,j,k,0) = 0./cellvol;
    }
#endif

#if 0
    // 2D Test for multilevel
    Real rad = Real(0.4);
    Real radi = Real(0.1);
    Real xcent = Real(0.5);
    Real ycent = Real(0.5);
    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0] - xcent;
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1] - ycent;
    Real r2 = (x*x + y*y);
    r2 = 2.*(x-y)*(x-y)+ .5*(x+y)*(x+y);
    Real r3 = 0.5*(x-y)*(x-y)+ 2.*(x+y)*(x+y);
#if (AMREX_SPACEDIM == 3)
    Real zcent = Real(0.5);
    Real z = prob_lo[2] + (k+Real(0.5)) * dx[2] - zcent;
    r3 = (x*x + y*y+z*z);
    r2 = (x-.1)*(x-.1)+(y-.2)*(y-.2)+z*z;
#endif

    if ( r3 > rad*rad) {
        //phi(i,j,k,0) = 50./cellvol;
        phi(i,j,k,0) = 30./cellvol;
    } else if (r2 < radi*radi) {
        //phi(i,j,k,0) = 20./cellvol;
        phi(i,j,k,0) = 15./cellvol;
    } else {
        phi(i,j,k,0) = 0./cellvol;
    }
    //if(std::abs(x) <= .1 && y > 0.){
    if(x >= .3 && y > .3){
       phi(i,j,k,0) = 0.;
    } else {
       phi(i,j,k,0) = 30./cellvol;
    }

//
#endif
#if 0
//    int ioff = 25;
//    int joff = 25;
    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0] ;
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1] ;

    Real value = 200./(.01 * .01);

//    if( i-ioff >= 0 && i - ioff < 10 && j-joff >= 0 && j - joff < 50 ) {
    if( x >= .25 && x < .35 && y >= .25 && y < .75 ) {
    //if( x >= .2 && x < .3 && y >= .25 && y < .75 ) {
       //phi(i,j,k,0) = 200./cellvol;
       phi(i,j,k,0) = value;
    } else {
       phi(i,j,k,0) = 0./cellvol;
    }

//

#endif
#if 0

        int ext_pot = a_ext_pot;
        amrex::Real alpha = a_alpha;
        amrex::Real beta = a_beta;
        amrex::Real gamma = a_gamma;
        Real x = prob_lo[0] + (i+Real(0.5)) * dx[0] ;
        Real y = prob_lo[1] + (j+Real(0.5)) * dx[1] ;

    if( x >= .2 && x < .4 && y >= .25 && y < .75 ) {
        amrex::Real pot = (x-alpha)*(x-alpha)*(x-beta)*(x-beta);
        pot += (y-.5)*(y-.5)*(y-.5)*(y-.5);
        pot /= gamma;
        Real value = 20000.* 200.* std::exp(-pot);
        phi(i,j,k,0) = value;
    } else {
       phi(i,j,k,0) = 0./cellvol;
    }



#endif

#if 1
    // Uniform initialization
    Real avg_ppc = 10.;
    phi(i,j,k,0) = (avg_ppc)/cellvol;
#endif

    if (Ncomp == 2){
        phi(i,j,k,1) = phi(i,j,k,0);
    }
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxx,
                     amrex::Array4<amrex::Real> const& stochfluxx,
                     amrex::Array4<amrex::Real const> const& phi,
                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                     EnsembleDirection const& ensemble_dir,
                     AMREX_D_DECL(amrex::Real dxinv, amrex::Real dyinv,amrex::Real dzinv),
                     amrex::Real dt,
                     amrex::Real time,
                     int lo, int hi, int dom_lo, int dom_hi, int bc_lo, int bc_hi,int Ncomp,
                     ExternalPotential const& external_potential,
                     amrex::Real a_d_spde,
                     FluxMode flux_mode)
{
    int coeff_index = 0; //Ncomp-1;
    amrex::Real phibdry = amrex::Real(0.);
    amrex::Real stofact = amrex::Real(1.);

    amrex::Real phip,phim, phipn, phimn;
    phipn = amrex::Real(0.);
    phimn = amrex::Real(0.);

/*
    if (lo == dom_lo &&
        (bc_lo == amrex::BCType::foextrap ||
         bc_lo == amrex::BCType::ext_dir))
    {
        if(i == lo)
        {
            fluxx(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i-1,j,k,0)) * dxinv / 0.5;
        }
        else
        {
            fluxx(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i-1,j,k,0)) * dxinv;
        }
    }
    else if (hi == dom_hi &&
             (bc_hi == amrex::BCType::foextrap ||
              bc_hi == amrex::BCType::ext_dir))
    {
        if(i == hi+1)
        {
            fluxx(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i-1,j,k,0)) * dxinv / 0.5;
        }
        else
        {
            fluxx(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i-1,j,k,0)) * dxinv;
        }
    }
*/
    //amrex::Print() << "BC info " << i << " " << dom_lo << " " << bc_lo << " " << dom_hi << " " << bc_hi << std::endl;
    bool wall_lo = (bc_lo == amrex::BCType::foextrap);
    bool ext_dir_lo = (bc_lo == amrex::BCType::ext_dir);

    bool wall_hi = (bc_hi == amrex::BCType::foextrap);
    bool ext_dir_hi = (bc_hi == amrex::BCType::ext_dir);

    if (i == dom_lo && ext_dir_lo) {
        stofact = std::sqrt(2.);
        phim = 2.*phibdry - phi(i,j,k,coeff_index);
        phip = phi(i,j,k,coeff_index);
        phimn = phibdry;
        phipn = std::max(phip, amrex::Real(0.));
        //amrex::Print() << " A " << std::endl;

    } else if (i == dom_hi+1 && ext_dir_hi) {
        stofact = std::sqrt(2.);
        phip = 2.*phibdry - phi(i-1,j,k,coeff_index);
        phim = phi(i-1,j,k,coeff_index);
        phimn = std::max(phim, amrex::Real(0.));
        phipn = phibdry;
        //amrex::Print() << " B " << std::endl;
    } else {

        phip = phi(i,j,k,0);
        phim = phi(i-1,j,k,0);
        phipn = std::max(phi(i,j,k,coeff_index), amrex::Real(0.));
        phimn = std::max(phi(i-1,j,k,coeff_index), amrex::Real(0.));
        stofact = 1.;
        //amrex::Print() << " C " << std::endl;

    }

    if (flux_mode == FluxMode::ml) {
        // ML model predictions are in terms of N and they need to be converted to 
        // number density
        amrex::Real dvolinv = AMREX_D_TERM(dxinv,*dyinv,*dzinv);
        // Note that volume for boundary faces is half the actual volume
        dvolinv *=  stofact*stofact;
        amrex::Real ml_flux  = stochfluxx(i,j,k); 
        // Convert from number of particles to number density  
        // Note that the boundary face is also only half cell size away
        ml_flux *= (dvolinv/(stofact*stofact*dxinv*dt));
        // Need to change the sign as ML model treats flux left to right positive
        fluxx(i,j,k,coeff_index) = -ml_flux;
    }
    else {
        // Note that Diffusion Coefficient changed from 0.5 to a_d_spde
        fluxx(i,j,k,coeff_index) = a_d_spde*(phip-phim) * dxinv;
        amrex::Real sqphiavg = std::sqrt(2.*a_d_spde)*0.5*(std::sqrt(phipn)+std::sqrt(phimn));
        fluxx(i,j,k,coeff_index) += stofact * stochfluxx(i,j,k) * sqphiavg;
    }
    if (external_potential.enabled) {
       amrex::Real xloc = prob_lo[0] + i / dxinv;
       amrex::Real yloc = prob_lo[1] + (j + amrex::Real(0.5)) / dyinv;
       amrex::Real phiavg = 0.5*(phim+phip);
       // Note: External potential logic is incorrect for periodic boundaries
       fluxx(i,j,k,coeff_index) += phiavg *
           external_potential_gradient_component(external_potential,
                                       ensemble_dir, 0, time, xloc, yloc);
    }
    // If the boundary is a wall flux has to be ZERO
    if (i == dom_lo && wall_lo) {
        fluxx(i,j,k,coeff_index) = amrex::Real(0.);
    } else if (i == dom_hi+1 && wall_hi) {
        fluxx(i,j,k,coeff_index) = amrex::Real(0.);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x_quantized (int i, int j, int k,
                               amrex::Array4<amrex::Real> const& fluxx,
                               amrex::Array4<amrex::Real> const& stochfluxx,
                               amrex::Array4<amrex::Real const> const& phi,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                               EnsembleDirection const& ensemble_dir,
                               AMREX_D_DECL(amrex::Real dxinv, amrex::Real dyinv,amrex::Real dzinv),
                               amrex::Real dt,
                               amrex::Real time,
                               int lo, int hi, int dom_lo, int dom_hi, int bc_lo, int bc_hi,int Ncomp,
                               ExternalPotential const& external_potential,
                               amrex::Real a_d_spde,
                               FluxMode flux_mode,
                               amrex::RandomEngine const& engine)
{
    compute_flux_x(i, j, k, fluxx, stochfluxx, phi,
                   prob_lo,
                   ensemble_dir,
                   AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                   lo, hi, dom_lo, dom_hi, bc_lo, bc_hi, Ncomp,
                   external_potential, a_d_spde, flux_mode);
    // The flux returned has the units of m^-2s^-1. Convert this to number of particles  
    int coeff_index = 0;
    amrex::Real face_area_x = AMREX_D_TERM(amrex::Real(1.),*dyinv,*dzinv); 
    amrex::Real flux_val = fluxx(i,j,k,coeff_index);
    // Note area is inverted
    flux_val *= (dt/face_area_x);
    amrex::Real rand_val = amrex::Random(engine);
    if (rand_val > flux_val-std::floor(flux_val)) {
        flux_val =  std::floor(flux_val);
    }
    else {
        flux_val = std::ceil(flux_val);
    }
    fluxx(i,j,k,coeff_index) = flux_val;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxy,
                     amrex::Array4<amrex::Real> const& stochfluxy,
                     amrex::Array4<amrex::Real const> const& phi,
                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                     EnsembleDirection const& ensemble_dir,
                     AMREX_D_DECL(amrex::Real dxinv, amrex::Real dyinv,amrex::Real dzinv),
                     amrex::Real dt,
                     amrex::Real time,
                     int lo, int hi, int dom_lo, int dom_hi, int bc_lo, int bc_hi, int Ncomp,
                     ExternalPotential const& external_potential,
                     amrex::Real a_d_spde,
                     FluxMode flux_mode)
{
    int coeff_index = 0; //Ncomp-1;
    amrex::Real phibdry = amrex::Real(0.);
    amrex::Real stofact = amrex::Real(1.);

    amrex::Real phip,phim, phipn, phimn;
    phipn = amrex::Real(0.);
    phimn = amrex::Real(0.);

    bool wall_lo = (bc_lo == amrex::BCType::foextrap);
    bool ext_dir_lo = (bc_lo == amrex::BCType::ext_dir);

    bool wall_hi = (bc_hi == amrex::BCType::foextrap);
    bool ext_dir_hi = (bc_hi == amrex::BCType::ext_dir);

    if (j == dom_lo && ext_dir_lo)
    {
        stofact = std::sqrt(2.);
        phim = 2.*phibdry - phi(i,j,k,coeff_index);
        phip = phi(i,j,k,coeff_index);
        phimn = phibdry;
        phipn = std::max(phip, amrex::Real(0.));

    } else if (j == dom_hi+1 && ext_dir_hi) {
        stofact = std::sqrt(2.);
        phip = 2.*phibdry - phi(i,j-1,k,coeff_index);
        phim = phi(i,j-1,k,coeff_index);
        phipn = phibdry;
        phimn = std::max(phim, amrex::Real(0.));

     } else {

        phip = phi(i,j,k,coeff_index);
        phim = phi(i,j-1,k,coeff_index);
        phipn = std::max(phi(i,j,k,coeff_index), amrex::Real(0.));
        phimn = std::max(phi(i,j-1,k,coeff_index), amrex::Real(0.));
        stofact = 1.;

    }
    if (flux_mode == FluxMode::ml) {
        // ML model predictions are in terms of N and they need to be converted to 
        // number density
        amrex::Real dvolinv = AMREX_D_TERM(dxinv,*dyinv,*dzinv);
        // Note that volume for boundary faces is half the actual volume
        dvolinv *=  stofact*stofact;
        amrex::Real ml_flux  = stochfluxy(i,j,k); 
        // Convert from number of particles to number density 
        // Note that the boundary face is also only half cell size away
        ml_flux *= (dvolinv/(stofact*stofact*dyinv*dt));
        // Need to change the sign as ML model treats flux left to right positive
        fluxy(i,j,k,coeff_index) = -ml_flux;
    } else {
        // Note that Diffusion Coefficient changed from 0.5 to a_d_spde
        fluxy(i,j,k,coeff_index) = a_d_spde*(phip-phim) * dyinv;
        amrex::Real sqphiavg = std::sqrt(2.0*a_d_spde)*0.5*(std::sqrt(phipn)+std::sqrt(phimn));
        fluxy(i,j,k,coeff_index) += stofact * stochfluxy(i,j,k)* sqphiavg;
    }
    if (external_potential.enabled) {
       amrex::Real xloc = prob_lo[0] + (i + amrex::Real(0.5)) / dxinv;
       amrex::Real yloc = prob_lo[1] + j / dyinv;
       amrex::Real phiavg = 0.5*(phip+phim);
       fluxy(i,j,k,coeff_index) += phiavg *
           external_potential_gradient_component(external_potential,
                                        ensemble_dir, 1, time, xloc, yloc);
    }
    // If the boundary is a wall flux has to be ZERO
    if (j == dom_lo && wall_lo) {
        fluxy(i,j,k,coeff_index) = amrex::Real(0.);
    } else if (j == dom_hi+1 && wall_hi) {
        fluxy(i,j,k,coeff_index) = amrex::Real(0.);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y_quantized (int i, int j, int k,
                               amrex::Array4<amrex::Real> const& fluxy,
                               amrex::Array4<amrex::Real> const& stochfluxy,
                               amrex::Array4<amrex::Real const> const& phi,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                               EnsembleDirection const& ensemble_dir,
                               AMREX_D_DECL(amrex::Real dxinv, amrex::Real dyinv,amrex::Real dzinv),
                               amrex::Real dt,
                               amrex::Real time,
                               int lo, int hi, int dom_lo, int dom_hi, int bc_lo, int bc_hi, int Ncomp,
                               ExternalPotential const& external_potential,
                               amrex::Real a_d_spde,
                               FluxMode flux_mode,
                               amrex::RandomEngine const& engine)
{
    compute_flux_y(i, j, k, fluxy, stochfluxy, phi,
                   prob_lo,
                   ensemble_dir,
                   AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                   lo, hi, dom_lo, dom_hi, bc_lo, bc_hi, Ncomp,
                   external_potential, a_d_spde, flux_mode);
    // The flux returned has the units of m^-2s^-1. Convert this to number of particles  
    int coeff_index = 0;
    amrex::Real face_area_y = AMREX_D_TERM(dxinv, *amrex::Real(1.),*dzinv); 
    amrex::Real flux_val = fluxy(i,j,k,coeff_index);
    // Note area is inverted
    flux_val *= (dt/face_area_y);
    amrex::Real rand_val = amrex::Random(engine);
    if (rand_val > flux_val-std::floor(flux_val)) {
        flux_val =  std::floor(flux_val);
    }
    else {
        flux_val = std::ceil(flux_val);
    }
    fluxy(i,j,k,coeff_index) = flux_val;
}


#if (AMREX_SPACEDIM > 2)
// Review this function before enabling the 3D path.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxz,
                     amrex::Array4<amrex::Real> const& stochfluxz,
                     amrex::Array4<amrex::Real const> const& phi,
                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                     EnsembleDirection const& ensemble_dir,
                     AMREX_D_DECL(amrex::Real dxinv, amrex::Real dyinv,amrex::Real dzinv),
                     amrex::Real dt,
                     amrex::Real time,
                     int lo, int hi, int dom_lo, int dom_hi, int bc_lo, int bc_hi, int Ncomp,
                     ExternalPotential const& external_potential,
                     amrex::Real a_d_spde,
                     FluxMode flux_mode)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(prob_lo);
    amrex::ignore_unused(ensemble_dir);
    if (external_potential.enabled) {
        amrex::Abort("External potential not coded in 3D scenario yet.\n");
    }

    bool wall_lo = (bc_lo == amrex::BCType::foextrap);
    bool ext_dir_lo = (bc_lo == amrex::BCType::ext_dir);

    bool wall_hi = (bc_hi == amrex::BCType::foextrap);
    bool ext_dir_hi = (bc_hi == amrex::BCType::ext_dir);

    if (flux_mode == FluxMode::ml) {
        amrex::Abort("3D implementation is NOT implemented for ML version");
    }

    int coeff_index = Ncomp-1;
    if (lo == dom_lo &&
        (bc_lo == amrex::BCType::foextrap ||
         bc_lo == amrex::BCType::ext_dir))
    {
        if(k == lo)
        {
            fluxz(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i,j,k-1,0)) * dzinv / 0.5;
        }
        else
        {
            fluxz(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i,j,k-1,0)) * dzinv;
        }
    }
    else if (hi == dom_hi &&
             (bc_hi == amrex::BCType::foextrap ||
              bc_hi == amrex::BCType::ext_dir))
    {
        if(k == hi+1)
        {
            fluxz(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i,j,k-1,0)) * dzinv / 0.5;
        }
        else
        {
            fluxz(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i,j,k-1,0)) * dzinv;
        }
    }
    else
    {
//        amrex::Real phiavg = 2.*(phi(i,j,k)*phi(i,j,k-1))/(phi(i,j,k)+phi(i,j,k-1)+1.e-16);
        amrex::Real phip = std::max(phi(i,j,k,coeff_index),0.);
        amrex::Real phim = std::max(phi(i,j,k-1,coeff_index),0.);
        //amrex::Real phiavg = 2.*(phip*phim)/(phip+phim+1.e-16);
        //fluxz(i,j,k,0) = 0.5*(phi(i,j,k,0)-phi(i,j,k-1,0)) * dzinv
        // +stochfluxz(i,j,k) * std::sqrt(phiavg);

        amrex::Real sqphiavg = std::sqrt(2.*a_d_spde)*0.5*(std::sqrt(phip)+std::sqrt(phim));
        // Note that Diffusion Coefficient changed from 0.5 to a_d_spde
        fluxz(i,j,k,0) = a_d_spde*(phi(i,j,k,0)-phi(i,j,k-1,0)) * dzinv
         +stochfluxz(i,j,k) * sqphiavg;
        if(Ncomp == 2){
           fluxz(i,j,k,1) = 0.5*(phi(i,j,k,1)-phi(i,j,k-1,1)) * dzinv;
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z_quantized (int i, int j, int k,
                               amrex::Array4<amrex::Real> const& fluxz,
                               amrex::Array4<amrex::Real> const& stochfluxz,
                               amrex::Array4<amrex::Real const> const& phi,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                               EnsembleDirection const& ensemble_dir,
                               AMREX_D_DECL(amrex::Real dxinv, amrex::Real dyinv,amrex::Real dzinv),
                               amrex::Real dt,
                               amrex::Real time,
                               int lo, int hi, int dom_lo, int dom_hi, int bc_lo, int bc_hi, int Ncomp,
                               ExternalPotential const& external_potential,
                               amrex::Real a_d_spde,
                               FluxMode flux_mode,
                               amrex::RandomEngine const& engine)
{
    compute_flux_z(i, j, k, fluxz, stochfluxz, phi,
                   prob_lo,
                   ensemble_dir,
                   AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                   lo, hi, dom_lo, dom_hi, bc_lo, bc_hi, Ncomp,
                   external_potential, a_d_spde, flux_mode);
    // The flux returned has the units of m^-2s^-1. Convert this to number of particles  
    int coeff_index = 0;
    amrex::Real face_area_z = AMREX_D_TERM(dxinv,*dyinv,amrex::Real(1.)); 
    amrex::Real flux_val = fluxz(i,j,k,coeff_index);
    // Note area is inverted
    flux_val *= (dt/face_area_z);
    amrex::Real rand_val = amrex::Random(engine);
    if (rand_val > flux_val-std::floor(flux_val)) {
        flux_val =  std::floor(flux_val);
    }
    else {
        flux_val = std::ceil(flux_val);
    }
    fluxz(i,j,k,coeff_index) = flux_val;
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                              amrex::Array4<amrex::Real const> const& fluxy,
                              amrex::Array4<amrex::Real const> const& fluxz),
                 amrex::Real dt,
                 AMREX_D_DECL(amrex::Real dxinv,
                              amrex::Real dyinv,
                              amrex::Real dzinv),
                 int Ncomp)
{
    for (int n = 0; n < Ncomp; n++){
        phinew(i,j,k,n) = phiold(i,j,k,n)
            +  dt * dxinv * (fluxx(i+1,j  ,k,n  ) - fluxx(i,j,k,n))
            +  dt * dyinv * (fluxy(i  ,j+1,k,n  ) - fluxy(i,j,k,n))
#if (AMREX_SPACEDIM > 2)
            +  dt * dzinv * (fluxz(i  ,j  ,k+1,n) - fluxz(i,j,k,n));
#else
            ;
#endif
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi_quantized (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                              amrex::Array4<amrex::Real const> const& fluxy,
                              amrex::Array4<amrex::Real const> const& fluxz),
                 amrex::Real dt,
                 AMREX_D_DECL(amrex::Real dxinv,
                              amrex::Real dyinv,
                              amrex::Real dzinv),
                 int Ncomp)
{
    // Fluxes are already in net particles crossing in dt time
    amrex::Real dvolinv = AMREX_D_TERM(dxinv,*dyinv,*dzinv);
    for (int n = 0; n < Ncomp; n++){
        phinew(i,j,k,n) = phiold(i,j,k,n)
            +  dvolinv * (fluxx(i+1,j  ,k,n  ) - fluxx(i,j,k,n))
            +  dvolinv * (fluxy(i  ,j+1,k,n  ) - fluxy(i,j,k,n))
#if (AMREX_SPACEDIM > 2)
            +  dvolinv * (fluxz(i  ,j  ,k+1,n) - fluxz(i,j,k,n));
#else
            ;
#endif
    }
}

#endif
