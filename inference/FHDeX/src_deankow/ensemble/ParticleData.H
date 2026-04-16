#ifndef _PARTICLE_DATA_H_
#define _PARTICLE_DATA_H_

#include <string>
#include <iostream>

// #include <IndexDefines.H>
#include <AMReX_Particles.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <StochasticPC.H>

/**
 * Container holding many of the particle-related data and options
 */

struct ParticleData {
  public:
    struct FaceFluxDiagnosticsConfig {
        bool enabled = false;
        amrex::Real face_x = amrex::Real(0.0);
    };

    void init_particle_params(int max_level, int ensemble_dir_exists,
                              ExternalPotential const& external_potential)
    {
        amrex::ParmParse pp(pp_prefix);
        m_external_potential = external_potential;

        use_particles = 0;
        multilevel_run = 1;
        pp.query("use_particles", use_particles);

        if (use_particles && ensemble_dir_exists) {
            multilevel_run = 0;
        }

        if (use_particles) {
            if (max_level != 1 && multilevel_run) {
                amrex::Abort("When using particles in multilevel mode you must have max_level == 1");
            }

            if (max_level != 0 && ensemble_dir_exists) {
                amrex::Abort("When using particles in ensemble mode you must have max_level == 0");
            }
        }

        if (ensemble_dir_exists) {
            amrex::ParmParse pp_ens;
            // read in if a direction is ensemble direction
            pp_ens.queryarr("is_ensemble_dir", m_ensemble_dir, 0, AMREX_SPACEDIM);
            pp_ens.query("remove_particle_history", m_remove_particle_history);
            pp_ens.query("remove_hist_freq", m_remove_hist_freq);
        }
    }

    void init_particles(amrex::ParGDBBase* a_gdb, const amrex::BoxArray& grown_fba,
                        amrex::MultiFab& phi_crse, amrex::MultiFab& phi_fine,
                        amrex::Real time)
    {
        // Initialize stochastic particles and phi at level 1
        if (use_particles && multilevel_run) {
            const amrex::Geometry& geom_fine = a_gdb->Geom(1);
            const auto dx_fine = geom_fine.CellSize();
#if (AMREX_SPACEDIM == 2)
            amrex::Real cell_vol_fine = dx_fine[0]*dx_fine[1];
#else
            amrex::Real cell_vol_fine = dx_fine[0]*dx_fine[1]*dx_fine[2];
#endif

            // **********************************************************************
            // Define the particle class with the default gdb, then change the particle
            //        boxArray (and DM) to be the grown ba and associated dm
            // **********************************************************************
            stochastic_particles = std::make_unique<StochasticPC>(a_gdb);
            stochastic_particles->SetParticleBoxArray(1,grown_fba);
            amrex::DistributionMapping grown_dm { grown_fba, amrex::ParallelDescriptor::NProcs() };
            stochastic_particles->SetParticleDistributionMap(1,grown_dm);

            // **********************************************************************
            // Fill a temporary phi_grown with phi from "coarse" level
            // **********************************************************************
            amrex::MultiFab phi_grown(grown_fba, grown_dm, 1, 0);
            phi_crse.FillBoundary(geom_fine.periodicity());
            phi_grown.ParallelCopy(phi_crse,0,0,1,1,0,geom_fine.periodicity());

            // **********************************************************************
            // Use this grown boxArray to hold the particles
            // **********************************************************************
            stochastic_particles->InitParticles(phi_grown, m_external_potential,
                                                m_ensemble_dir, time);

            amrex::Print() << "Before Removal we have  " << stochastic_particles->TotalNumberOfParticles() <<
                              " stochastic particles." << std::endl;

            // **********************************************************************
            // Remove particles not in the real fine boxArray
            // **********************************************************************
            stochastic_particles->RemoveParticlesNotInBA(phi_fine.boxArray());

            // **********************************************************************
            // Redistribute so every particle is now in the "correct" place
            // **********************************************************************
            stochastic_particles->Redistribute();
            amrex::Print() << "Initialized " << stochastic_particles->TotalNumberOfParticles() <<
                              " stochastic particles." << std::endl;

            // **********************************************************************
            // Define phi at level 1 from the particle count
            // **********************************************************************
            int lev = 1;
            phi_fine.setVal(amrex::Real(0.0));
            stochastic_particles->Increment(phi_fine, lev);
            phi_fine.mult(1./cell_vol_fine);
            amrex::Print() << "Used the particle locations to define phi at level 1 " << std::endl;
        }
    }

    // This version initializes particles at level 0
    // Particles are created using comp_id = 0 of phi_crse,
    // but comp_id = 1 is again reset based on the initialized particles
    void init_particles(amrex::ParGDBBase* a_gdb, amrex::MultiFab& phi_crse,
                        amrex::Real time)
    {
        const int lev = 0;
        // The component of phi_crse that will be modified based on Particles
        const int comp_id = 1;
        // Initialize stochastic particles at level 0
        if (use_particles && multilevel_run == 0) {
            const amrex::Geometry& geom_crse = a_gdb->Geom(lev);
            const auto dx_crse = geom_crse.CellSize();
#if (AMREX_SPACEDIM == 2)
            amrex::Real cell_vol_crse = dx_crse[0]*dx_crse[1];
#else
            amrex::Real cell_vol_crse = dx_crse[0]*dx_crse[1]*dx_crse[2];
#endif
            // **********************************************************************
            // Define the particle class with the default gdb
            // **********************************************************************
            stochastic_particles = std::make_unique<StochasticPC>(a_gdb);
            // **********************************************************************
            // Use phi_crse to hold the particles
            // **********************************************************************
            stochastic_particles->InitParticles(phi_crse, m_external_potential,
                                                m_ensemble_dir, time, lev);
            // **********************************************************************
            // Redistribute so every particle is in the "correct" place
            // **********************************************************************
            stochastic_particles->Redistribute();
            // **********************************************************************
            // Remove particles that are not in the valid boxArray
            // **********************************************************************
            stochastic_particles->RemoveParticlesNotInBA(phi_crse.boxArray(), lev);
            // **********************************************************************
            // Define comp_id = 1 of phi_crse based on particles
            // **********************************************************************
            amrex::MultiFab phi_from_ptcls(phi_crse, amrex::make_alias, comp_id, 1);
            phi_from_ptcls.setVal(amrex::Real(0.));
            stochastic_particles->Increment(phi_from_ptcls, lev);
            phi_from_ptcls.FillBoundary(a_gdb->Geom(lev).periodicity());
            phi_from_ptcls.mult(1./cell_vol_crse);
            amrex::Print() << "Initialized " << stochastic_particles->TotalNumberOfParticles() <<
                              " stochastic particles on level = 0." << std::endl;
            // Recorrecting comp=0 using comp=1
            amrex::MultiFab::Copy(phi_crse, phi_from_ptcls,0,0,1,phi_crse.nGrow());
        }
    }

    void regrid_particles(const amrex::BoxArray& new_fine_ba, const amrex::BoxArray& new_ungrown_ba,
                          const amrex::BoxArray& old_fine_ba, amrex::MultiFab& phi_fine,
                          amrex::Real time)
    {
        amrex::DistributionMapping new_ungrown_dm { new_ungrown_ba, amrex::ParallelDescriptor::NProcs() };
        stochastic_particles->SetParticleBoxArray(1, new_ungrown_ba);
        stochastic_particles->SetParticleDistributionMap(1, new_ungrown_dm);

        stochastic_particles->Redistribute();
        stochastic_particles->RemoveParticlesAtLevel(0);
        amrex::Real factor = m_factor;
        stochastic_particles->AddParticles(phi_fine, old_fine_ba, factor,
                                           m_external_potential, m_ensemble_dir, time);
        phi_fine.setVal(amrex::Real(0.0));
        const int lev = 1;
        const auto dx = stochastic_particles->Geom(lev).CellSizeArray();

#if (AMREX_SPACEDIM == 2)
    const amrex::Real cell_vol = dx[0]*dx[1];
#else
    const amrex::Real cell_vol = dx[0]*dx[1]*dx[2];
#endif
        stochastic_particles->Increment(phi_fine, lev);
        phi_fine.mult(1./cell_vol);

        amrex::DistributionMapping new_dm { new_fine_ba, amrex::ParallelDescriptor::NProcs() };
        stochastic_particles->SetParticleBoxArray(1, new_fine_ba);
        stochastic_particles->SetParticleDistributionMap(1, new_dm);
    }

    void writePlotFile (const std::string& a_fname, amrex::MultiFab const& phi_fine) const
    {
        if (use_particles && multilevel_run) {
            bool is_checkpoint = false;
            amrex::MultiFab phi_grown(stochastic_particles->ParticleBoxArray(1),
                                      stochastic_particles->ParticleDistributionMap(1),1,0);
            const amrex::Geometry& geom_fine = stochastic_particles->GetParGDB()->Geom(1);
            phi_grown.ParallelCopy(phi_fine,0,0,1,1,0,geom_fine.periodicity());
            stochastic_particles->ColorParticlesWithPhi(phi_grown);
            stochastic_particles->Checkpoint( a_fname, "particles", is_checkpoint, particle_varnames );
        }
    }

    void Checkpoint (const std::string& a_fname) const
    {
        if (use_particles) {
            bool is_checkpoint = true;
            stochastic_particles->Checkpoint(a_fname, "stochastic_particles", is_checkpoint, particle_varnames);
        }
    }

    void Restart(amrex::ParGDBBase* a_gdb, std::string& restart_file)
    {
        if (use_particles) {
            stochastic_particles = std::make_unique<StochasticPC>(a_gdb);
            std::string particle_file("stochastic_particles");
            stochastic_particles->Restart(restart_file, particle_file);
        }
    }

    void ConfigureFaceFluxDiagnostics(FaceFluxDiagnosticsConfig const& config)
    {
        m_face_flux_diag = config;
    }

    void BeginFaceFluxDiagnostics(int lev, amrex::Real dt_lev)
    {
        amrex::ignore_unused(lev, dt_lev);
        if (!m_face_flux_diag.enabled || !stochastic_particles) {
            return;
        }

        const int ny = stochastic_particles->Geom(lev).Domain().length(1);
        m_particle_face_flux_profile.assign(ny, amrex::Real(0.0));
    }

    void EndFaceFluxDiagnostics(int lev, amrex::Real dt_lev)
    {
        if (!m_face_flux_diag.enabled) {
            return;
        }

        if (!stochastic_particles) {
            return;
        }

        const amrex::Geometry& geom = stochastic_particles->Geom(lev);
        const auto problo = geom.ProbLoArray();
        const auto probhi = geom.ProbHiArray();
        const auto dx = geom.CellSizeArray();
        const amrex::Box& domain = geom.Domain();
        const int ny = domain.length(1);
        const amrex::Real face_x = m_face_flux_diag.face_x;
        const amrex::Real lx = probhi[0] - problo[0];
        const bool periodic_x = geom.isPeriodic(0);
        const bool periodic_y = geom.isPeriodic(1);
        const amrex::Real face_area = AMREX_D_TERM(amrex::Real(1.0), * dx[1], * dx[2]);
        const auto dxi = geom.InvCellSizeArray();
        const auto is_periodic = geom.isPeriodicArray();

        amrex::Gpu::DeviceVector<amrex::Real> flux_bins_d(ny, amrex::Real(0.0));
        amrex::Real* flux_bins_ptr = flux_bins_d.dataPtr();

        for (typename StochasticPC::ParIterType pti(*stochastic_particles, lev); pti.isValid(); ++pti) {
            auto& ptile = stochastic_particles->ParticlesAt(lev, pti);
            auto& aos = ptile.GetArrayOfStructs();
            const int np = aos.numParticles();
            auto* pstruct = aos().data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                auto const& p = pstruct[i];
                amrex::Real xold = p.rdata(RealIdx::xold);
                amrex::Real yold = p.rdata(RealIdx::yold);
                amrex::Real xnew = p.pos(0);
                auto new_pos = getNewCell(p, problo, dxi, domain);
                amrex::ignore_unused(new_pos, dxi, is_periodic);

                amrex::Real disp = xnew - xold;
                if (periodic_x) {
                    if (disp > amrex::Real(0.5) * lx) {
                        disp -= lx;
                    } else if (disp < -amrex::Real(0.5) * lx) {
                        disp += lx;
                    }
                }

                amrex::Real xnew_unwrapped = xold + disp;
                amrex::Real sign = amrex::Real(0.0);
                auto crosses_face = [=] AMREX_GPU_DEVICE (amrex::Real face_loc) noexcept
                {
                    if (disp > amrex::Real(0.0)) {
                        return (xold < face_loc && xnew_unwrapped >= face_loc) ? amrex::Real(-1.0)
                                                                               : amrex::Real(0.0);
                    } else if (disp < amrex::Real(0.0)) {
                        return (xold >= face_loc && xnew_unwrapped < face_loc) ? amrex::Real(1.0)
                                                                                : amrex::Real(0.0);
                    }
                    return amrex::Real(0.0);
                };

                sign = crosses_face(face_x);
                if (periodic_x && sign == amrex::Real(0.0)) {
                    sign = crosses_face(face_x + lx);
                }
                if (periodic_x && sign == amrex::Real(0.0)) {
                    sign = crosses_face(face_x - lx);
                }

                if (sign == amrex::Real(0.0)) {
                    return;
                }

                if (periodic_y) {
                    amrex::Real ly = probhi[1] - problo[1];
                    if (yold < problo[1]) {
                        yold += ly;
                    } else if (yold >= probhi[1]) {
                        yold -= ly;
                    }
                }

                int j = static_cast<int>(amrex::Math::floor((yold - problo[1]) / dx[1]));
                j = amrex::max(0, amrex::min(ny - 1, j));
                amrex::Gpu::Atomic::AddNoRet(&flux_bins_ptr[j], sign / (dt_lev * face_area));
            });
        }

        amrex::Gpu::streamSynchronize();
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                         flux_bins_d.begin(), flux_bins_d.end(),
                         m_particle_face_flux_profile.begin());
        amrex::ParallelDescriptor::ReduceRealSum(m_particle_face_flux_profile.data(), ny);
    }

    [[nodiscard]] bool UsingParticles() const
    {
        return use_particles != 0;
    }

    [[nodiscard]] amrex::Vector<amrex::Real> const& GetFaceFluxDiagnosticsProfile() const
    {
        return m_particle_face_flux_profile;
    }

    void advance_particles(int lev, amrex::Real time, amrex::Real dt_lev, amrex::Real cell_vol,
                           amrex::MultiFab& phi_crse_old,
                           amrex::MultiFab& phi_crse_new,
                           amrex::MultiFab& phi_fine)
    {
        // Update stochastic particles
        if (use_particles && multilevel_run) {
            AMREX_ALWAYS_ASSERT(lev == 1);
            const amrex::Geometry& geom_fine = stochastic_particles->GetParGDB()->Geom(lev);

            // **********************************************************************
            // Fill a temporary phi_grown with phi from "coarse" level
            // **********************************************************************
            // amrex::Print() << "PARTICLE BOXARRAY " << stochastic_particles->ParticleBoxArray(0) << std::endl;
            // amrex::Print() << "PARTICLE BOXARRAY " << stochastic_particles->ParticleBoxArray(1) << std::endl;
            amrex::MultiFab phi_grown(stochastic_particles->ParticleBoxArray(1),
                                      stochastic_particles->ParticleDistributionMap(1),1,0);
            phi_crse_old.FillBoundary(geom_fine.periodicity());
            phi_grown.ParallelCopy(phi_crse_old,0,0,1,1,0,geom_fine.periodicity());

            // **********************************************************************
            // Create new particles in first layer of ghost cells around the level 1 grids
            // **********************************************************************
            amrex::BoxArray ba_to_exclude(phi_fine.boxArray());

            amrex::Print() << "Adding particles in advance " << std::endl;

            amrex::Real factor = m_factor;
            stochastic_particles->AddParticles(phi_grown, ba_to_exclude, factor,
                                               m_external_potential, m_ensemble_dir, time);

            // **********************************************************************
            // Advect the particles in the valid region and in the ghost region
            // **********************************************************************
            BeginFaceFluxDiagnostics(lev, dt_lev);
            stochastic_particles->AdvectWithRandomWalk(lev, dt_lev, time,
                                        m_external_potential, m_ensemble_dir);
            EndFaceFluxDiagnostics(lev, dt_lev);

            // **********************************************************************
            // Account for particles that went from crse --> fine BEFORE Redistribute
            // **********************************************************************
            amrex::MultiFab phi_grown2(phi_grown.boxArray(),phi_grown.DistributionMap(),1,1);
            phi_grown2.setVal(amrex::Real(0.));

            stochastic_particles->RefluxCrseToFine(phi_fine.boxArray(), phi_grown2);

            phi_grown2.mult(1./cell_vol,0,1,1);

            phi_crse_new.ParallelAdd(phi_grown2, 0, 0, 1, amrex::IntVect::TheUnitVector(),
                                     amrex::IntVect::TheUnitVector());

            // **********************************************************************
            // Put particles on the correct levels and in the right boxes
            // **********************************************************************
            stochastic_particles->Redistribute();

            // **********************************************************************
            // Remove particles on the coarsest level
            // **********************************************************************

            //amrex::Print() << "Before Removal we have  " << stochastic_particles->TotalNumberOfParticles() <<
            //                  " stochastic particles." << std::endl;

            int crse_lev = 0;
            stochastic_particles->RemoveParticlesAtLevel(crse_lev);

            //amrex::Print() << " After Removal we have  " << stochastic_particles->TotalNumberOfParticles() <<
            //                  " stochastic particles." << std::endl;

            // **********************************************************************
            // Account for particles that went from fine --> crse
            // **********************************************************************

            phi_grown.setVal(amrex::Real(0.));

            stochastic_particles->RefluxFineToCrse(phi_fine.boxArray(), phi_grown);

            phi_grown.mult(1./cell_vol);
#if (AMREX_SPACEDIM == 2)
            print_state(phi_grown, amrex::IntVect(31,16));
#else
            print_state(phi_grown, amrex::IntVect(31,16,0));
#endif

            phi_crse_new.ParallelAdd(phi_grown,0,0,1);

            // **********************************************************************
            // Remove particles that are not in the valid boxArray used by phi_fine
            // **********************************************************************
            stochastic_particles->RemoveParticlesNotInBA(phi_fine.boxArray());

            // **********************************************************************
            // Define phi at level 1 from the particle count
            // **********************************************************************
            phi_grown.setVal(amrex::Real(0.));
            stochastic_particles->Increment(phi_grown, lev);

            // *****************************************************************************
            // Overwrite phi at level 0 from phi at level 1 only in the valid level 1 region
            // *****************************************************************************
            phi_fine.ParallelCopy(phi_grown,0,0,1);

            phi_fine.mult(1./cell_vol);
            amrex::Print() << "Used the particle locations to define phi at level 1 " << std::endl;
        }
    }

    void advance_particles(int lev, amrex::Real time, amrex::Real dt_lev, amrex::Real cell_vol,
                           amrex::MultiFab& phi_crse_new, int a_istep)
    {
        // The component of phi_crse_new that will be modified based on Particles
        const int comp_id = 1;
        // Update stochastic particles
        if (use_particles && (multilevel_run == 0)) {
            //***********************************************************************
            // Delete particles and re-add them to remove history effects
            //***********************************************************************
            if (m_remove_particle_history &&
                (a_istep % m_remove_hist_freq == 0)) {
                // Store phi based on current particle location
                amrex::MultiFab phi_crse_old(phi_crse_new.boxArray(),
                                             phi_crse_new.DistributionMap(),
                                             1, phi_crse_new.nGrow());
                phi_crse_old.setVal(amrex::Real(0.));
                stochastic_particles->Increment(phi_crse_old, lev);
                phi_crse_old.FillBoundary(
                        stochastic_particles->GetParGDB()->Geom(lev).periodicity());
                phi_crse_old.mult(1./cell_vol);
                // Delete the current state of ALL particles
                stochastic_particles->RemoveParticlesAtLevel(lev);
                // Add them again based on phi_crse_old
                stochastic_particles->InitParticles(phi_crse_old, m_external_potential,
                                                    m_ensemble_dir, time, lev);
                // **********************************************************************
                // Redistribute so every particle is in the "correct" place
                // **********************************************************************
                stochastic_particles->Redistribute();
                // **********************************************************************
                // Remove particles that are not in the valid boxArray
                // **********************************************************************
                stochastic_particles->RemoveParticlesNotInBA(phi_crse_old.boxArray(), lev);
            }

            // **********************************************************************
            // Advect the particles in the valid region and in the ghost region
            // **********************************************************************
            BeginFaceFluxDiagnostics(lev, dt_lev);
            stochastic_particles->AdvectWithRandomWalk(lev, dt_lev, time,
                                        m_external_potential, m_ensemble_dir);
            EndFaceFluxDiagnostics(lev, dt_lev);
            // **********************************************************************
            // Put particles on the correct levels and in the right boxes
            // **********************************************************************
            stochastic_particles->Redistribute();
            // **********************************************************************
            // Remove particles that are not in the valid boxArray
            // **********************************************************************
            stochastic_particles->RemoveParticlesNotInBA(phi_crse_new.boxArray(), lev);
            // **********************************************************************
            // Define comp_id = 1 of phi_crse_new based on particles
            // **********************************************************************
            amrex::MultiFab phi_from_ptcls(phi_crse_new, amrex::make_alias, comp_id, 1);
            phi_from_ptcls.setVal(amrex::Real(0.));
            stochastic_particles->Increment(phi_from_ptcls, lev);
            phi_from_ptcls.FillBoundary(
                    stochastic_particles->GetParGDB()->Geom(lev).periodicity());
            phi_from_ptcls.mult(1./cell_vol);
        }
    }

    void Redistribute()
    {
        if (use_particles) {
            stochastic_particles->Redistribute();
        }
    }

    std::string pp_prefix {"amr"};

    int use_particles;
    int multilevel_run;
    std::unique_ptr<StochasticPC> stochastic_particles;
    amrex::Vector<std::string> particle_varnames = {"xold", "yold", "zold"};
    // This variable decides if a particular direction is ensemble direction
    // grid spacing along ensemble directions needs to be UNITY
    amrex::Vector<int> m_ensemble_dir = {AMREX_D_DECL(0, 0, 0)};
    // This bool controls if the positions of particles are
    // reinitialized after every time step. ONLY used in ensemble mode
    bool m_remove_particle_history = false;
    int m_remove_hist_freq = 1;
    ExternalPotential m_external_potential;
    amrex::Real m_factor = amrex::Real(0.9);
    FaceFluxDiagnosticsConfig m_face_flux_diag;
    amrex::Vector<amrex::Real> m_particle_face_flux_profile;
};
#endif
