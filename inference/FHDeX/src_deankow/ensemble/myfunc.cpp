#include "myfunc.H"
#include <AMReX.H>
#include "mykernel.H"
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include "common_functions.H"
#include "rng_functions.H"
#include <limits>
#include <torch/script.h>
#include <type_traits>

namespace {

using namespace amrex;

constexpr torch::Dtype TorchRealDType ()
{
    if constexpr (std::is_same_v<amrex::Real, float>) {
        return torch::kFloat32;
    } else {
        return torch::kFloat64;
    }
}

struct ScopedDisableFPExcept
{
    explicit ScopedDisableFPExcept (amrex::FPExcept excepts)
        : m_prev(amrex::disableFPExcept(excepts))
    {}

    ~ScopedDisableFPExcept ()
    {
        amrex::setFPExcept(m_prev);
    }

    ScopedDisableFPExcept (ScopedDisableFPExcept const&) = delete;
    ScopedDisableFPExcept& operator= (ScopedDisableFPExcept const&) = delete;

private:
    amrex::FPExcept m_prev;
};

// This function updates the flux history as the scaling is very different
// flux history is in NET PARTICLES CROSSING and NOT NUMBER DENSITY
// Do NOT update hist_count in this function
void UpdateMLFluxHistory (int hist_count, int history_len,
                          amrex::Geometry const& geom,
                          amrex::MultiFab const& flux_new,
                          amrex::MultiFab& flux_hist)
{
    if (history_len == 0) { return;}

    if (hist_count < history_len) {
        const int head = hist_count;
        amrex::MultiFab::Copy(flux_hist, flux_new, 0, head, 1, 0);
        // Keep shared faces consistent across ranks for ML history.
        flux_hist.OverrideSync(geom.periodicity());
    }
    else {
        if (history_len == 1) {
            amrex::MultiFab::Copy(flux_hist, flux_new, 0, 0, 1, 0);
        } else {
            // History full: shift left by 1 to keep most recent hist_len entries.
            amrex::MultiFab flux_hist_copy(flux_hist.boxArray(),
                                           flux_hist.DistributionMap(),
                                           flux_hist.nComp()-1, 0);
            amrex::MultiFab::Copy(flux_hist_copy, flux_hist, 1, 0,
                                  flux_hist.nComp()-1, 0);
            amrex::MultiFab::Swap(flux_hist, flux_hist_copy, 0, 0,
                                  flux_hist.nComp()-1, 0);
            const int head = history_len - 1;
            amrex::MultiFab::Copy(flux_hist, flux_new, 0, head, 1, 0);
        }
        flux_hist.OverrideSync(geom.periodicity());
    }
}

void FillMLStochFluxDir (int dir,
                         const amrex::Real dvol,
                         amrex::MultiFab& stochFlux_dir,
                         amrex::MultiFab const& phi_old,
                         amrex::MultiFab const* phi_hist,
                         amrex::MultiFab* flux_hist,
                         amrex::Geometry const& geom,
                         int hist_count, int history_len,
                         int flow_steps, amrex::Real flow_t0, amrex::Real flow_t1,
                         torch::jit::script::Module* module,
                         bool use_cuda, amrex::Real ml_input_scale,
                         amrex::Real ml_output_mn_fctr,
                         amrex::Real ml_output_std_fctr,
                         bool quantize_ml_output)
{
    if (module == nullptr) {
        amrex::Abort("ML flux mode requires a loaded TorchScript model.");
    }
    if (history_len > 0 && phi_hist == nullptr) {
        amrex::Abort("ML flux mode requires phi history buffers when history_len > 0.");
    }
    if (history_len > 0 && flux_hist == nullptr) {
        amrex::Abort("ML flux mode requires flux history buffers when history_len > 0.");
    }

    const int dens_T = hist_count + 1;
    const int flux_T = (hist_count == 0) ? 1 : hist_count;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi_old,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        bool quntz_l = quantize_ml_output;
        const Box& fbx = mfi.nodaltilebox(dir);
        const auto lo = amrex::lbound(fbx);
        const auto nbox = fbx.size();

        int ncell = AMREX_D_TERM(nbox[0], * nbox[1], * nbox[2]);

        amrex::Gpu::ManagedVector<amrex::Real> dens_buf(ncell * dens_T * 2);
        amrex::Gpu::ManagedVector<amrex::Real> flux_buf(ncell * flux_T);
        amrex::Gpu::ManagedVector<amrex::Real> x_buf(ncell);
        auto const& phi = phi_old.array(mfi);
        auto const& stoch = stochFlux_dir.array(mfi);
        auto const phi_hist_arr = (phi_hist != nullptr)
            ? phi_hist->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        auto const flux_hist_arr = (flux_hist != nullptr)
            ? flux_hist->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};

        auto dens_ptr = dens_buf.dataPtr();
        auto flux_ptr = flux_buf.dataPtr();
        auto x_ptr = x_buf.dataPtr();

        amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int ii = i - lo.x;
            int jj = j - lo.y;
            int index = jj*nbox[0] + ii;
#if AMREX_SPACEDIM == 3
            int kk = k - lo.z;
            index += kk*nbox[0]*nbox[1];
#endif
            int iL = i;
            int jL = j;
            int kL = k;
            int iR = i;
            int jR = j;
            int kR = k;

            if (dir == 0) {
                iL = i - 1;
                iR = i;
            } else if (dir == 1) {
                jL = j - 1;
                jR = j;
            } else {
                kL = k - 1;
                kR = k;
            }

            // ML model inputs are in terms of Number of particles
            // followed by scaling
            amrex::Real phiL = phi(iL, jL, kL, 0);
            phiL *= (dvol/ml_input_scale);
            amrex::Real phiR = phi(iR, jR, kR, 0);
            phiR *= (dvol/ml_input_scale);

            int dens_base = index * dens_T * 2;
            for (int h = 0; h < hist_count; ++h) {
                int comp = h;
                amrex::Real phiLh = phi_hist_arr(iL, jL, kL, comp);
            	phiLh *= (dvol/ml_input_scale);
                amrex::Real phiRh = phi_hist_arr(iR, jR, kR, comp);
                phiRh *= (dvol/ml_input_scale);
                int off = dens_base + h * 2;
                dens_ptr[off + 0] = std::max(phiLh, amrex::Real(0.));
                dens_ptr[off + 1] = std::max(phiRh, amrex::Real(0.));
            }
            int cur_off = dens_base + hist_count * 2;
            dens_ptr[cur_off + 0] = std::max(phiL, amrex::Real(0.));
            dens_ptr[cur_off + 1] = std::max(phiR, amrex::Real(0.));

            // For flux it is net particles crossing
            if (hist_count == 0) {
                flux_ptr[index] = 0.0;
            } else {
                int flux_base = index * flux_T;
                for (int h = 0; h < hist_count; ++h) {
                    int comp = h;
                    flux_ptr[flux_base + h] = flux_hist_arr(i, j, k, comp);
                    flux_ptr[flux_base + h] /= ml_input_scale;
                }
            }

            x_ptr[index] = stoch(i,j,k,0);
        });

        amrex::Gpu::synchronize();

        auto opts = torch::TensorOptions().dtype(TorchRealDType());
#ifdef AMREX_USE_CUDA
        if (use_cuda) {
            opts = opts.device(torch::Device(torch::kCUDA, amrex::Gpu::Device::deviceId()));
        } else {
            opts = opts.device(torch::kCPU);
        }
#else
        amrex::ignore_unused(use_cuda);
        opts = opts.device(torch::kCPU);
#endif
        at::Tensor dens_t     = torch::from_blob(dens_buf.dataPtr(), {ncell, dens_T, 2}, opts).clone();
        at::Tensor flux_t     = torch::from_blob(flux_buf.dataPtr(), {ncell, flux_T, 1}, opts).clone();
        at::Tensor tgt_t     = torch::from_blob(x_buf.dataPtr(),    {ncell, 1, 1},    opts).clone();

        torch::InferenceMode guard;
        const int steps = (flow_steps > 0) ? flow_steps : 1;
        amrex::Real dt = (flow_t1 - flow_t0) / static_cast<amrex::Real>(steps);
        for (int s = 0; s < steps; ++s) {
            amrex::Real tval = flow_t0 + dt * static_cast<amrex::Real>(s);
            at::Tensor time_pst = torch::full({ncell, 1, 1}, tval, opts);
            at::Tensor grad_t;
#ifdef AMREX_USE_CUDA
            grad_t = module->forward({tgt_t, dens_t, flux_t, time_pst}).toTensor();
#else
#ifdef AMREX_USE_OMP
#pragma omp critical(torch_jit_forward_cpu)
#endif
            {
                ScopedDisableFPExcept no_fpe(amrex::FPExcept::all);
                grad_t = module->forward({tgt_t, dens_t, flux_t, time_pst}).toTensor();
            }
#endif
            if (grad_t.dim() == 1) {
                grad_t = grad_t.view({ncell, 1, 1});
            } else if (grad_t.dim() == 2) {
                grad_t = grad_t.view({ncell, 1, grad_t.size(1)});
            }
            if (grad_t.dim() != 3 || grad_t.size(1) != 1 || grad_t.size(2) != 1) {
                amrex::Abort("ML flow model must return shape (ncell,1,1) or compatible.");
            }
            grad_t = torch::nan_to_num(grad_t, 0.0, 0.0, 0.0);
            tgt_t = torch::nan_to_num(tgt_t + dt * grad_t, 0.0, 0.0, 0.0);
        }
        tgt_t = tgt_t.contiguous();
        //amrex::Gpu::ManagedVector<amrex::Real> out_buf(ncell);
        //auto acc = tgt_t.accessor<amrex::Real,3>();
        //for (int n = 0; n < ncell; ++n) {
        //    out_buf[n] = acc[n][0][0];
        //}

        //auto out_ptr = out_buf.dataPtr();
        auto out_ptr = tgt_t.data_ptr<amrex::Real>();
        amrex::ParallelForRNG(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k,
                                           amrex::RandomEngine const& engine)
        {
            int ii = i - lo.x;
            int jj = j - lo.y;
            int index = jj*nbox[0] + ii;
#if AMREX_SPACEDIM == 3
            int kk = k - lo.z;
            index += kk*nbox[0]*nbox[1];
#endif
            int iL = i;
            int jL = j;
            int kL = k;
            int iR = i;
            int jR = j;
            int kR = k;

            if (dir == 0) {
                iL = i - 1;
                iR = i;
            } else if (dir == 1) {
                jL = j - 1;
                jR = j;
            } else {
                kL = k - 1;
                kR = k;
            }
            amrex::Real flux_val = out_ptr[index];
            // The output is scaled net particles crossing
            amrex::Real phiL = phi(iL, jL, kL, 0);
            phiL *= dvol;
            phiL = std::max(phiL, amrex::Real(0.));
            amrex::Real phiR = phi(iR, jR, kR, 0);
            phiR *= dvol;
            phiR = std::max(phiR, amrex::Real(0.));
            if (flux_val != flux_val ||
                flux_val > std::numeric_limits<amrex::Real>::max() ||
                flux_val < -std::numeric_limits<amrex::Real>::max()) {
                flux_val = amrex::Real(0.0);
            }
            flux_val *= ml_output_std_fctr * std::sqrt(std::max(phiL + phiR, amrex::Real(0.0)));
            flux_val += ml_output_mn_fctr*(phiL-phiR);
            // Quantization of Net particles crossing
            if (quntz_l) {
                phiL = std::floor(phiL);
                phiR = std::floor(phiR);
                amrex::Real rand_val = amrex::Random(engine);
                if (rand_val > flux_val-std::floor(flux_val)) {
                    flux_val =  std::floor(flux_val);
                }
                else {
                    flux_val = std::ceil(flux_val);
                }
            }
            flux_val = std::min(flux_val, phiL);
            flux_val = std::max(flux_val, -phiR);
            stoch(i,j,k,0) = flux_val;
        });
    }
    // The new fluxes are in terms of Net particles crossing
    // Save them to flux_hist
    if (flux_hist != nullptr) {
        UpdateMLFluxHistory(hist_count, history_len,
                            geom, stochFlux_dir, *flux_hist);
    }
}

void FillStudentTStochFluxDir (int dir,
                               amrex::MultiFab& stochFlux_dir,
                               amrex::Geometry const& geom,
                               amrex::Real df, amrex::Real loc, amrex::Real scale)
{
    for (MFIter mfi(stochFlux_dir); mfi.isValid(); ++mfi)
    {
        const Box& fbx = mfi.nodaltilebox(dir);
        auto const& stoch = stochFlux_dir.array(mfi);
        amrex::Real df_copy = df;
        amrex::Real loc_copy = loc;
        amrex::Real scale_copy = scale;
        amrex::ParallelForRNG(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k,
                                                        amrex::RandomEngine const& engine) noexcept
        {
            amrex::Real z = amrex::RandomNormal(0.0_rt, 1.0_rt, engine);
            amrex::Real u = amrex::RandomGamma(0.5_rt * df_copy, 2.0_rt, engine);
            amrex::Real t = z / std::sqrt(u / df_copy);
            stoch(i,j,k,0) = t * scale_copy + loc_copy;
        });
    }
}

}

void advance_phi (MultiFab& phi_old,
                  MultiFab& phi_new,
                  Array<MultiFab, AMREX_SPACEDIM>& flux,
                  Array<MultiFab, AMREX_SPACEDIM>& stochFlux,
                  Real dt,
                  Real time,
                  Real /*npts_scale*/,
                  Geometry const& geom,
                  Vector<BCRec> const& BoundaryCondition,
                  Vector<int> const& a_ensemble_dir,
                  ExternalPotential const& external_potential,
                  Real a_d_spde,
                  MLFluxContext const* ml_ctx)
{
    int Ncomp = phi_old.nComp();

    AMREX_D_TERM(const Real dxinv = geom.InvCellSize(0);,
                 const Real dyinv = geom.InvCellSize(1);,
                 const Real dzinv = geom.InvCellSize(2););
    AMREX_D_TERM(Real dvol = dxinv, *dyinv, *dzinv);
    dvol = Real(1.)/dvol;

    const Box& domain_bx = geom.Domain();
    const auto dom_lo = lbound(domain_bx);
    const auto dom_hi = ubound(domain_bx);
    const auto prob_lo = geom.ProbLoArray();
    EnsembleDirection ensemble_dir{
        AMREX_D_DECL(a_ensemble_dir[0], a_ensemble_dir[1], a_ensemble_dir[2])};

    //Real variance = dxinv*dyinv/(npts_scale*dt);
    Real variance = dxinv*dyinv/dt;

#if(AMREX_SPACEDIM > 2)
    variance *=dzinv;
#endif

    const bool use_ml = (ml_ctx != nullptr && ml_ctx->mode == FluxMode::ml);
    if (!use_ml) {
        // Fill stochFlux with random numbers (can skip density component 0)
        for (int d=0;d<AMREX_SPACEDIM;d++) {
            MultiFabFillRandom(stochFlux[d], 0, variance, geom);
        }
    } else {
        if (ml_ctx->history_len > 0) {
            if (ml_ctx->phi_hist == nullptr) {
                amrex::Abort("ML flux mode requires history buffers.");
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (ml_ctx->flux_hist[d] == nullptr) {
                    amrex::Abort("ML flux mode requires flux history buffers for all directions.");
                }
            }
        }
        for (int d=0; d<AMREX_SPACEDIM; ++d) {
            // Apply ML model only along non-ensemble directions
            if (a_ensemble_dir[d] == 0) {
            	FillStudentTStochFluxDir(d, stochFlux[d], geom,
            	                         ml_ctx->t_df, ml_ctx->t_loc, ml_ctx->t_scale);
            	FillMLStochFluxDir(d, dvol, stochFlux[d], phi_old, ml_ctx->phi_hist,
            	                   ml_ctx->flux_hist[d], geom, ml_ctx->hist_count,
            	                   ml_ctx->history_len,
            	                   ml_ctx->flow_steps, ml_ctx->flow_t0, ml_ctx->flow_t1,
            	                   ml_ctx->module, ml_ctx->use_cuda,
            	                   ml_ctx->ml_input_scale,
            	                   ml_ctx->ml_output_mn_fctr,
            	                   ml_ctx->ml_output_std_fctr,
            	                   ml_ctx->quantize_ml_output);
            	// Keep shared faces/nodes consistent across ranks.
            	stochFlux[d].OverrideSync(geom.periodicity());
            }
        }
    }

    const BCRec& bc = BoundaryCondition[0];

    bool quantize_fluxes = (use_ml && ml_ctx->quantize_ml_output);

    // Compute fluxes one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.nodaltilebox(0);
        auto const& fluxx = flux[0].array(mfi);
        const Box& ybx = mfi.nodaltilebox(1);
        auto const& fluxy = flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.nodaltilebox(2);
        auto const& fluxz = flux[2].array(mfi);
#endif
        auto const& stochfluxx = stochFlux[0].array(mfi);
        auto const& stochfluxy = stochFlux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& stochfluxz = stochFlux[2].array(mfi);
#endif
        const Box& bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        auto const& phi = phi_old.array(mfi);

        if (a_ensemble_dir[0] == 0) {
            if (quantize_fluxes) {
                amrex::ParallelForRNG(xbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k,
                               amrex::RandomEngine const& engine)
                    {
                        compute_flux_x_quantized(i,j,k,fluxx,stochfluxx,phi,
                                                 prob_lo,
                                                 ensemble_dir,
                                                 AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                                                 lo.x, hi.x, dom_lo.x, dom_hi.x,
                                                 bc.lo(0), bc.hi(0), Ncomp,
                                                 external_potential, a_d_spde,
                                                 use_ml ? FluxMode::ml : FluxMode::gaussian,
                                                 engine);
                    });
            } else {
                amrex::ParallelFor(xbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        compute_flux_x(i,j,k,fluxx,stochfluxx,phi,
                                       prob_lo,
                                       ensemble_dir,
                                       AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                                       lo.x, hi.x, dom_lo.x, dom_hi.x, bc.lo(0), bc.hi(0),Ncomp,
                                       external_potential, a_d_spde,
                                       use_ml ? FluxMode::ml : FluxMode::gaussian);
                    });
            }
        }

        if (a_ensemble_dir[1] == 0) {
            if (quantize_fluxes) {
                amrex::ParallelForRNG(ybx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k,
                               amrex::RandomEngine const& engine)
                    {
                        compute_flux_y_quantized(i,j,k,fluxy,stochfluxy,phi,
                                                 prob_lo,
                                                 ensemble_dir,
                                                 AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                                                 lo.y, hi.y, dom_lo.y, dom_hi.y,
                                                 bc.lo(1), bc.hi(1), Ncomp,
                                                 external_potential, a_d_spde,
                                                 use_ml ? FluxMode::ml : FluxMode::gaussian,
                                                 engine);
                    });
            } else {
                amrex::ParallelFor(ybx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        compute_flux_y(i,j,k,fluxy,stochfluxy,phi,
                                       prob_lo,
                                       ensemble_dir,
                                       AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                                       lo.y, hi.y, dom_lo.y, dom_hi.y, bc.lo(1), bc.hi(1),Ncomp,
                                       external_potential, a_d_spde,
                                       use_ml ? FluxMode::ml : FluxMode::gaussian);
                    });
            }
        }
#if (AMREX_SPACEDIM > 2)
        if (a_ensemble_dir[2] == 0) {
            if (quantize_fluxes) {
                amrex::ParallelForRNG(zbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k,
                               amrex::RandomEngine const& engine)
                    {
                        compute_flux_z_quantized(i,j,k,fluxz,stochfluxz,phi,
                                                 prob_lo,
                                                 ensemble_dir,
                                                 AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                                                 lo.z, hi.z, dom_lo.z, dom_hi.z,
                                                 bc.lo(2), bc.hi(2), Ncomp,
                                                 external_potential, a_d_spde,
                                                 use_ml ? FluxMode::ml : FluxMode::gaussian,
                                                 engine);
                    });
            } else {
                amrex::ParallelFor(zbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        compute_flux_z(i,j,k,fluxz,stochfluxz,phi,
                                       prob_lo,
                                       ensemble_dir,
                                       AMREX_D_DECL(dxinv, dyinv, dzinv), dt, time,
                                       lo.z, hi.z, dom_lo.z, dom_hi.z, bc.lo(2), bc.hi(2),Ncomp,
                                       external_potential, a_d_spde,
                                       use_ml ? FluxMode::ml : FluxMode::gaussian);
                    });
            }
        }
#endif
    }

    // Perform an OverrideSync if quantization was performed
    if (quantize_fluxes) {
        flux[0].OverrideSync(geom.periodicity());
        flux[1].OverrideSync(geom.periodicity());
#if (AMREX_SPACEDIM == 3)
        flux[2].OverrideSync(geom.periodicity());
#endif
    }
    // Advance the solution one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& fluxx = flux[0].array(mfi);
        auto const& fluxy = flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& fluxz = flux[2].array(mfi);
#endif
        auto const& phiOld = phi_old.array(mfi);
        auto const& phiNew = phi_new.array(mfi);
        if (quantize_fluxes) {
            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                update_phi_quantized(i,j,k,phiOld,phiNew,
                           AMREX_D_DECL(fluxx,fluxy,fluxz),
                           dt,
                           AMREX_D_DECL(dxinv,dyinv,dzinv),
                           Ncomp);
            });
        }
        else {
            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                update_phi(i,j,k,phiOld,phiNew,
                           AMREX_D_DECL(fluxx,fluxy,fluxz),
                           dt,
                           AMREX_D_DECL(dxinv,dyinv,dzinv),
                           Ncomp);
            });
        }
    }

    if (!quantize_fluxes) {
        Real dx, dy, dz = 1.;
        AMREX_D_TERM(dx = geom.CellSize(0);,
                     dy = geom.CellSize(1);,
                     dz = geom.CellSize(2););

        // Compute fluxes one grid at a time
        for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const Box& xbx = mfi.nodaltilebox(0);
            const Box& ybx = mfi.nodaltilebox(1);

            auto const& fluxx = flux[0].array(mfi);
            auto const& fluxy = flux[1].array(mfi);

            amrex::ParallelFor(xbx, Ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                     fluxx(i,j,k,n) *= dt * dy * dz;
                });

            amrex::ParallelFor(ybx, Ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                     fluxy(i,j,k,n) *= dt * dx * dz;
                });

#if (AMREX_SPACEDIM > 2)
            const Box& zbx = mfi.nodaltilebox(2);
            auto const& fluxz = flux[2].array(mfi);
            amrex::ParallelFor(zbx, Ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                     fluxz(i,j,k,n) *= dt * dx * dy;
                });
#endif
        }
    }
}
