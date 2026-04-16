#include <AmrCoreAdv.H>
#include <Kernels.H>
#include <myfunc.H>

using namespace amrex;

// Advance a single level for a single time step, updates flux registers
void
AmrCoreAdv::AdvancePhiAtLevel (int lev, Real time, Real dt_lev, int /*iteration*/, int /*ncycle*/)
{
    Array<MultiFab, AMREX_SPACEDIM> fluxes;
    Array<MultiFab, AMREX_SPACEDIM> stochFluxes;
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        BoxArray ba = grids[lev];
        ba.surroundingNodes(i);
        fluxes[i].define(ba, dmap[lev], phi_new[lev].nComp(), 0);
        fluxes[i].setVal(amrex::Real(0.));
        stochFluxes[i].define(ba, dmap[lev], phi_new[lev].nComp(), 0);
        stochFluxes[i].setVal(amrex::Real(0.));
    }

    phi_old[lev].FillBoundary(Geom(lev).periodicity());

    // We do this here so we can print the FABs for debugging
    phi_new[lev].setVal(amrex::Real(0.0));

    MLFluxContext ml_ctx;
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> flux_hist_ptrs;
    if (m_flux_mode == FluxMode::ml) {
        ml_ctx.mode = FluxMode::ml;
        ml_ctx.history_len = m_ml_history_len;
        ml_ctx.history_ngrow = m_ml_history_ngrow;
        ml_ctx.hist_count = m_ml_hist_count[lev];
        ml_ctx.flow_steps = m_ml_flow_steps;
        ml_ctx.flow_t0 = m_ml_flow_t0;
        ml_ctx.flow_t1 = m_ml_flow_t1;
        ml_ctx.t_df = m_ml_t_df;
        ml_ctx.t_loc = m_ml_t_loc;
        ml_ctx.t_scale = m_ml_t_scale;
        ml_ctx.ml_input_scale = m_ml_input_scale;
        ml_ctx.ml_output_mn_fctr = m_ml_output_mn_fctr;
        ml_ctx.ml_output_std_fctr = m_ml_output_std_fctr;
        ml_ctx.quantize_ml_output = m_quantize_ml_output;
        ml_ctx.module = m_ml_module.get();
        ml_ctx.use_cuda = m_ml_use_cuda;
        if (m_ml_history_len > 0) {
            if (!m_phi_hist[lev]) {
                amrex::Abort("ML flux mode requires phi history buffers.");
            }
            ml_ctx.phi_hist = m_phi_hist[lev].get();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (!m_flux_hist[lev][d]) {
                    amrex::Abort("ML flux mode requires flux history buffers for all directions.");
                }
                flux_hist_ptrs[d] = m_flux_hist[lev][d].get();
            }
            ml_ctx.flux_hist = flux_hist_ptrs;

            m_phi_hist[lev]->FillBoundary(Geom(lev).periodicity());
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                // Ensure shared faces are consistent before feeding ML.
                m_flux_hist[lev][d]->OverrideSync(Geom(lev).periodicity());
            }
        }
    }

    advance_phi(phi_old[lev], phi_new[lev], fluxes, stochFluxes, dt_lev,
            time, npts_scale, geom[lev], bcs,
            m_ensemble_dir, m_external_potential,
            m_d_spde, (m_flux_mode == FluxMode::ml) ? &ml_ctx : nullptr);

    CaptureSPDEFaceFluxProfile(lev, fluxes);

    if (m_flux_mode == FluxMode::ml) {
        UpdateMLPhiHistory(lev);
    }

    // Increment or decrement the flux registers by area and time-weighted fluxes
    // Note that the fluxes have already been scaled by dt and area
    // In this example we are solving phi_t = -div(+F)
    // The fluxes contain, e.g., F_{i+1/2,j} = (phi*u)_{i+1/2,j}
    // Keep this in mind when considering the different sign convention for updating
    // the flux registers from the coarse or fine grid perspective
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    if (do_reflux) {
        if (flux_reg[lev+1]) {
            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                // update the lev+1/lev flux register (index lev+1)
                flux_reg[lev+1]->CrseInit(fluxes[i],i,0,0,fluxes[i].nComp(),amrex::Real(1.0));
            }
        }
        if (flux_reg[lev]) {
            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                // update the lev/lev-1 flux register (index lev)
                flux_reg[lev]->FineAdd(fluxes[i],i,0,0,fluxes[i].nComp(),amrex::Real(-1.0));
            }
        }
    }
}
