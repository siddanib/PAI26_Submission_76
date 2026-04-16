#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>

#include <ExternalPotential.H>
#include <FluxMode.H>
#include <torch/script.h>

using namespace amrex;

struct MLFluxContext {
    FluxMode mode = FluxMode::gaussian;
    int history_len = 0;
    int history_ngrow = 1;
    int hist_count = 0;
    int flow_steps = 20;
    amrex::Real flow_t0 = amrex::Real(0.0);
    amrex::Real flow_t1 = amrex::Real(1.0);
    amrex::Real t_df = amrex::Real(4.0);
    amrex::Real t_loc = amrex::Real(0.0);
    amrex::Real t_scale = amrex::Real(1.0);
    amrex::Real ml_input_scale = amrex::Real(51.0);
    amrex::Real ml_output_mn_fctr = amrex::Real(0.069);
    amrex::Real ml_output_std_fctr = amrex::Real(0.2537);
    bool quantize_ml_output = false;
    const amrex::MultiFab* phi_hist = nullptr;
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> flux_hist = {};
    torch::jit::script::Module* module = nullptr;
#ifdef AMREX_USE_GPU	    
    bool use_cuda = true;
#else
    bool use_cuda = false;
#endif
};

void advance_phi (amrex::MultiFab& phi_old,
                  amrex::MultiFab& phi_new,
                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& stochFlux,
                  amrex::Real dt, amrex::Real time, amrex::Real npts_scale,
                  amrex::Geometry const& geom,
                  Vector<amrex::BCRec> const& BoundaryCondition,
                  Vector<int> const& a_ensemble_dir,
                  ExternalPotential const& external_potential,
                  Real a_d_spde,
                  MLFluxContext const* ml_ctx);

#endif
