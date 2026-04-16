
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Reduce.H>
#include <torch/script.h>

#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <AmrCoreAdv.H>
#include <Kernels.H>
#include <mykernel.H>
#include "chrono"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <type_traits>

using namespace amrex;
using namespace std::chrono;

namespace {

constexpr torch::Dtype TorchRealDType ()
{
    if constexpr (std::is_same_v<amrex::Real, float>) {
        return torch::kFloat32;
    } else {
        return torch::kFloat64;
    }
}

void CaptureSPDEFaceFluxProfileImpl (
    int face_index,
    amrex::Geometry const& geom,
    amrex::MultiFab const& flux_x,
    amrex::Vector<amrex::Real>& spde_face_flux)
{
    const amrex::Box& domain = geom.Domain();
    const int jlo = domain.smallEnd(1);
    const int ny = domain.length(1);
    auto overlap_mask = flux_x.OverlapMask(geom.periodicity());

    amrex::Gpu::DeviceVector<amrex::Real> face_flux_d(ny, amrex::Real(0.0));
    amrex::Real* face_flux_ptr = face_flux_d.dataPtr();

    for (amrex::MFIter mfi(flux_x); mfi.isValid(); ++mfi) {
        const amrex::Box& xbx = mfi.nodaltilebox(0);
        if (face_index < xbx.smallEnd(0) || face_index > xbx.bigEnd(0)) {
            continue;
        }

        amrex::Box line_box(amrex::IntVect(AMREX_D_DECL(face_index, xbx.smallEnd(1), 0)),
                            amrex::IntVect(AMREX_D_DECL(face_index, xbx.bigEnd(1), 0)));
        auto const& fluxx = flux_x.const_array(mfi);
        auto const& mask = overlap_mask->const_array(mfi);
        amrex::ParallelFor(line_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::ignore_unused(i, k);
            const amrex::Real cell_count = amrex::max(mask(face_index, j, 0), amrex::Real(1.0));
            amrex::Gpu::Atomic::AddNoRet(
                &face_flux_ptr[j - jlo],
                fluxx(face_index, j, 0, 0) / cell_count);
        });
    }

    amrex::Gpu::streamSynchronize();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     face_flux_d.begin(), face_flux_d.end(),
                     spde_face_flux.begin());
    amrex::ParallelDescriptor::ReduceRealSum(spde_face_flux.data(), ny);
}

void ComputeReducedDensitiesImpl (
    amrex::MultiFab const& phi_mf,
    amrex::Geometry const& gm,
    amrex::Real x_min,
    amrex::Real x_max,
    amrex::Vector<amrex::Real>& spde_reduced_density,
    amrex::Vector<amrex::Real>& particle_reduced_density)
{
    const amrex::Box& domain = gm.Domain();
    const int jlo = domain.smallEnd(1);
    const int ny = domain.length(1);
    const auto prob_lo = gm.ProbLoArray();
    const auto dx = gm.CellSizeArray();
    amrex::Gpu::DeviceVector<amrex::Real> spde_sum_d(ny, amrex::Real(0.0));
    amrex::Gpu::DeviceVector<amrex::Real> particle_sum_d(ny, amrex::Real(0.0));
    amrex::Real* spde_sum_ptr = spde_sum_d.dataPtr();
    amrex::Real* particle_sum_ptr = particle_sum_d.dataPtr();

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            if (x >= x_min && x <= x_max) {
                amrex::Gpu::Atomic::AddNoRet(&spde_sum_ptr[j - jlo], phi(i,j,k,0));
                amrex::Gpu::Atomic::AddNoRet(&particle_sum_ptr[j - jlo], phi(i,j,k,1));
            }
        });
    }

    amrex::Gpu::streamSynchronize();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     spde_sum_d.begin(), spde_sum_d.end(),
                     spde_reduced_density.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     particle_sum_d.begin(), particle_sum_d.end(),
                     particle_reduced_density.begin());
    amrex::ParallelDescriptor::ReduceRealSum(spde_reduced_density.data(), ny);
    amrex::ParallelDescriptor::ReduceRealSum(particle_reduced_density.data(), ny);
}

}

// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRe boundary condition object
AmrCoreAdv::AmrCoreAdv ()
{

    // periodic boundaries
    //int bc_lo[] = {BCType::int_dir, BCType::int_dir, BCType::int_dir};
    //int bc_hi[] = {BCType::int_dir, BCType::int_dir, BCType::int_dir};
    amrex::Vector<int> bc_lo(AMREX_SPACEDIM,0);
    amrex::Vector<int> bc_hi(AMREX_SPACEDIM,0);

/*
    // walls (Neumann)
    int bc_lo[] = {amrex::BCType::foextrap, amrex::BCType::foextrap, amrex::BCType::foextrap};
    int bc_hi[] = {amrex::BCType::foextrap, amrex::BCType::foextrap, amrex::BCType::foextrap};
*/
    // walls Dirichlet
    //int bc_lo[] = {amrex::BCType::ext_dir, amrex::BCType::ext_dir, amrex::BCType::ext_dir};
    //int bc_hi[] = {amrex::BCType::ext_dir, amrex::BCType::ext_dir, amrex::BCType::ext_dir};

    ReadParameters(bc_lo,bc_hi);

    /////////////////////////////////////////
    //Initialise rngs
    /////////////////////////////////////////
    int restart = -1;

    if (restart < 0) {

        if (seed > 0) {
            // initializes the seed for C++ random number calls
            InitRandom(seed+ParallelDescriptor::MyProc(),
                       ParallelDescriptor::NProcs(),
                       seed+ParallelDescriptor::MyProc());
        } else if (seed == 0) {
            // initializes the seed for C++ random number calls based on the clock
            auto now = time_point_cast<nanoseconds>(system_clock::now());
            int randSeed = now.time_since_epoch().count();
            // broadcast the same root seed to all processors
            ParallelDescriptor::Bcast(&randSeed,1,ParallelDescriptor::IOProcessorNumber());
            InitRandom(randSeed+ParallelDescriptor::MyProc(),
                       ParallelDescriptor::NProcs(),
                       randSeed+ParallelDescriptor::MyProc());
        } else {
            Abort("Must supply non-negative seed");
        }
    }

    // Geometry on all levels has been defined already.

    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);
    if (do_subcycle) {
        for (int lev = 1; lev <= max_level; ++lev) {
            nsubsteps[lev] = MaxRefRatio(lev-1);
        }
    }

    t_new.resize(nlevs_max, amrex::Real(0.0));
    t_old.resize(nlevs_max, amrex::Real(-1.e100));
    dt.resize(nlevs_max, amrex::Real(1.e100));

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);
    m_phi_hist.resize(nlevs_max);
    m_flux_hist.resize(nlevs_max);
    m_ml_hist_count.resize(nlevs_max, 0);

    bcs.resize(1);     // Setup 1-component
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        if (bc_lo[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_lo[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_lo[idim] == BCType::ext_dir ) {  // external Dirichlet
            bcs[0].setLo(idim, bc_lo[idim]);
        }
        else {
            amrex::Abort("Invalid bc_lo");
        }

        // hi-side BCSs
        if (bc_hi[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_hi[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_hi[idim] == BCType::ext_dir ) {  // external Dirichlet
            bcs[0].setHi(idim, bc_hi[idim]);
        }
        else {
            amrex::Abort("Invalid bc_hi");
        }
    }

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] is never actually used in the reflux operation
    flux_reg.resize(nlevs_max+1);

    // fillpatcher[lev] is for filling data on level lev using the data on
    // lev-1 and lev.
    fillpatcher.resize(nlevs_max+1);

    if (m_flux_mode == FluxMode::ml) {
        if (m_ml_model_file.empty()) {
            amrex::Abort("flux_mode=ml requires ml_model_file to be set.");
        }
#ifdef AMREX_USE_CUDA
        const auto ml_device = torch::Device(torch::kCUDA, amrex::Gpu::Device::deviceId());
#endif
        const auto finalize_model = [&] ()
        {
            m_ml_module->to(TorchRealDType());
            m_ml_module->eval();
        };
        const auto load_model = [&] ()
        {
#ifdef AMREX_USE_CUDA
            m_ml_module = std::make_unique<torch::jit::script::Module>(
                torch::jit::load(m_ml_model_file, ml_device));
#else
            m_ml_module = std::make_unique<torch::jit::script::Module>(
                torch::jit::load(m_ml_model_file));
#endif
            finalize_model();
        };
        try {
#ifdef AMREX_USE_CUDA
            load_model();
#else
#ifdef AMREX_USE_MPI
            Vector<char> model_bytes;
            MPI_Comm node_comm = MPI_COMM_NULL;
            int ierr = MPI_Comm_split_type(ParallelDescriptor::Communicator(),
                                           MPI_COMM_TYPE_SHARED,
                                           ParallelDescriptor::MyProc(),
                                           MPI_INFO_NULL,
                                           &node_comm);
            if (ierr != MPI_SUCCESS) {
                amrex::ParallelDescriptor::MPI_Error(__FILE__, __LINE__,
                                                     "MPI_Comm_split_type(ParallelDescriptor::Communicator(), MPI_COMM_TYPE_SHARED, ParallelDescriptor::MyProc(), MPI_INFO_NULL, &node_comm)",
                                                     ierr);
            }
            ParallelContext::push(node_comm, ParallelContext::frames.size(), 0);
            const auto pop_node_comm = [&] () noexcept
            {
                ParallelContext::pop();
                int free_ierr = MPI_Comm_free(&node_comm);
                if (free_ierr != MPI_SUCCESS) {
                    amrex::ParallelDescriptor::MPI_Error(__FILE__, __LINE__,
                                                         "MPI_Comm_free(&node_comm)",
                                                         free_ierr);
                }
            };
            try {
                if (ParallelContext::IOProcessorSub()) {
                    std::ifstream is(m_ml_model_file, std::ios::binary);
                    if (!is.good()) {
                        amrex::Abort("Error opening the TorchScript model file on node-local reader.");
                    }
                    is.seekg(0, std::ios::end);
                    const std::streamoff nbytes = is.tellg();
                    if (nbytes < 0) {
                        amrex::Abort("Error determining TorchScript model file size on node-local reader.");
                    }
                    is.seekg(0, std::ios::beg);
                    model_bytes.resize(static_cast<std::size_t>(nbytes));
                    if (!model_bytes.empty()) {
                        is.read(model_bytes.dataPtr(), nbytes);
                    }
                    if (!is.good() && !is.eof()) {
                        amrex::Abort("Error reading the TorchScript model file on node-local reader.");
                    }
                }

                std::uint64_t model_nbytes = model_bytes.size();
                ParallelDescriptor::Bcast(&model_nbytes, 1,
                                          ParallelContext::IOProcessorNumberSub(),
                                          ParallelContext::CommunicatorSub());
                if (!ParallelContext::IOProcessorSub()) {
                    model_bytes.resize(static_cast<std::size_t>(model_nbytes));
                }
                if (model_nbytes > 0) {
                    ParallelDescriptor::Bcast(model_bytes.dataPtr(), model_nbytes,
                                              ParallelContext::IOProcessorNumberSub(),
                                              ParallelContext::CommunicatorSub());
                }
                const int local_nprocs = ParallelContext::NProcsSub();
                const int local_rank = ParallelContext::MyProcSub();
                const int batch_size = std::max(1, m_ml_load_batch_size);
                for (int start = 0; start < local_nprocs; start += batch_size) {
                    const int stop = std::min(start + batch_size, local_nprocs);
                    if (local_rank >= start && local_rank < stop) {
                        std::string model_blob(model_bytes.data(), model_bytes.size());
                        std::istringstream model_stream(model_blob, std::ios::binary);
                        m_ml_module = std::make_unique<torch::jit::script::Module>(
                            torch::jit::load(model_stream));
                        finalize_model();
                    }
                    ParallelDescriptor::Barrier(ParallelContext::CommunicatorSub());
                }
                pop_node_comm();
            } catch (...) {
                pop_node_comm();
                throw;
            }
#else
            std::string model_blob;
            {
                std::ifstream is(m_ml_model_file, std::ios::binary);
                if (!is.good()) {
                    amrex::Abort("Error opening the TorchScript model file.");
                }
                std::ostringstream os;
                os << is.rdbuf();
                if (!is.good() && !is.eof()) {
                    amrex::Abort("Error reading the TorchScript model file.");
                }
                model_blob = os.str();
            }
            std::istringstream model_stream(model_blob, std::ios::binary);
            m_ml_module = std::make_unique<torch::jit::script::Module>(
                torch::jit::load(model_stream));
            finalize_model();
#endif
#endif
        }
        catch (const c10::Error&) {
            amrex::Abort("Error loading the TorchScript model.");
        }

#ifdef AMREX_USE_CUDA
        m_ml_use_cuda = true;
        m_ml_module->to(ml_device);
#else
        m_ml_use_cuda = false;
        amrex::Print() << "ML model loaded via node-local AMReX broadcast with local deserialize batch size "
                       << m_ml_load_batch_size << ": " << m_ml_model_file << "\n";
#endif
#ifdef AMREX_USE_CUDA
        amrex::Print() << "ML model loaded: " << m_ml_model_file << "\n";
#endif
    }
}

AmrCoreAdv::~AmrCoreAdv () = default;

// advance solution to final time
void
AmrCoreAdv::Evolve ()
{
    Real cur_time = t_new[0];
    int last_plot_file_step = 0;

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << std::endl;

        ComputeDt();

        int lev = 0;
        int iteration = 1;
        timeStepNoSubcycling(cur_time, iteration);

        cur_time += dt[0];

        // sum phi to check conservation
        Real sum_phi_old = phi_old[0].sum();
        Real sum_phi_new = phi_new[0].sum();

        amrex::Print() << "Coarse STEP " << step+1 << " ends." << " TIME = " << cur_time
                       << " DT = " << dt[0] << " Sum_old Sum_new Diff (Phi) = "    << std::setw(20) << std::setprecision(12)
                       << std::scientific <<  sum_phi_old << " " << std::setw(2l) << std::setprecision(12)
                       << std::scientific <<  sum_phi_new << " " << std::setw(2l) << std::setprecision(12)
                       << std::scientific << (sum_phi_new - sum_phi_old) << std::endl;
        if (alg_type != 0 ) {
          // sum phi to check conservation
          sum_phi_old = phi_old[0].sum(1);
          sum_phi_new = phi_new[0].sum(1);

          amrex::Print() << "Ensemble Sum_old Sum_new Diff (Phi1) = "    << std::setw(20) << std::setprecision(12)
                         << std::scientific <<  sum_phi_old << " " << std::setw(2l) << std::setprecision(12)
                         << std::scientific <<  sum_phi_new << " " << std::setw(2l) << std::setprecision(12)
                         << std::scientific << (sum_phi_new - sum_phi_old) << std::endl;
          }

        // sync up time
        for (lev = 0; lev <= finest_level; ++lev) {
            t_new[lev] = cur_time;
        }

        if (plot_int > 0 && (step+1) % plot_int == 0) {
            last_plot_file_step = step+1;
            WritePlotFile();
        }

        if (chk_int > 0 && (step+1) % chk_int == 0) {
            WriteCheckpointFile();
        }

#ifdef AMREX_MEM_PROFILING
        {
            std::ostringstream ss;
            ss << "[STEP " << step+1 << "]";
            MemProfiler::report(ss.str());
        }
#endif

        if (cur_time >= stop_time - 1.e-6*dt[0]) { break; }
    }

    if (plot_int > 0 && istep[0] > last_plot_file_step) {
        WritePlotFile();
    }
}

// initializes multilevel data
void
AmrCoreAdv::InitData ()
{
    if (restart_chkfile.empty()) {
        // start simulation from the beginning
        const Real time = 0.0;

        // Note that InitFromScratch allocates the space for phi at each level,
        //      but only initializes phi at level 0, not at level > 0.
        //      So we can't do average down until we create the particles,
        //      then use the particles to define the level 1 phi
        InitFromScratch(time);

#ifdef AMREX_PARTICLES
        if (max_level > 0) {
            particleData.init_particles((amrex::ParGDBBase*)GetParGDB(), grown_fba, phi_new[0], phi_new[1],
                                        time);
        }
        else {
            particleData.init_particles((amrex::ParGDBBase*)GetParGDB(), phi_new[0], time);
        }
#endif
        AverageDown();
        phi_new[0].FillBoundary();

        MultiFab::Copy(phi_old[0], phi_new[0],0,0,1,0);
        phi_old[0].FillBoundary();

        if (chk_int > 0) {
            WriteCheckpointFile();
        }
    }
    else {
        // restart from a checkpoint
        ReadCheckpointFile();
    }
    InitDiagnostics();
    if (plot_int > 0) {
        WritePlotFile();
    }
}

void AmrCoreAdv::MakeFBA(const BoxArray& ba)
{
    int lev = 1;
    Box domain(Geom(lev).Domain());
    BoxList valid_bl(ba);
    BoxList com_bl = GetBndryCells(ba,1);
#if (AMREX_SPACEDIM == 2)
    Vector<IntVect> pshifts(9);
#else
    Vector<IntVect> pshifts(27);
#endif

    BoxList com_bl_fixed;

    //
    // Loop over boxes created by GetBndryCells call -- note that if periodic
    // some of these boxes may intersect the valid_bl so we remove those intersections
    // by intersecting with the copmlement of the valid ba
    //
    for (auto& b : com_bl) {
        Box bx(b);

        //
        // First intersect the existing box with the domain and keep that
        // Note that GetBndryCells would not include any cells inside the domain
        // that are part of the original ba
        //
        Box b1 = bx & domain;
        if (!b1.isEmpty()) {
            com_bl_fixed.push_back(b1);
        }

        //
        // Next add the pieces that were outside the domain in a periodic direction
        // Note that GetBndryCells DOES include cells outside the domain
        // that are part of the original ba if shifted periodically
        //
        geom[lev].periodicShift(domain, bx, pshifts);
        for (int n = 0; n < pshifts.size(); n++) {
            Box bx_shift(b);
            bx_shift.shift(pshifts[n]);
            Box b2 = bx_shift & domain;
            if (!b2.isEmpty()) {
                // Now we have to make sure we don't include any intersection of this b2
                // with the valid boxArray
                BoxList bl_comp = complementIn(b2,valid_bl);
                for (auto& b_comp : bl_comp) {
                    Box bx_comp(b_comp);
                    if (!bx_comp.isEmpty()) {
                        com_bl_fixed.push_back(bx_comp);
                    }
                }
            }
        }
    }

    //
    // Remove any duplicated regions in the boundary cells
    //
    com_bl_fixed.simplify();

    //
    // Add the valid boxes
    //
    com_bl_fixed.catenate(valid_bl);
    grown_fba.define(com_bl_fixed);
}

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data (overrides the pure virtual function in AmrCore)
// regrid  --> RemakeLevel            (if level already existed)
// regrid  --> MakeNewLevelFromCoarse (if adding new level)
void
AmrCoreAdv::MakeNewLevelFromCoarse (int lev, Real time, const BoxArray& ba,
                                    const DistributionMapping& dm)
{
    const int ncomp = phi_new[lev-1].nComp();
    const int ng = phi_new[lev-1].nGrow();

    amrex::Print() << " CREATE LEVEL " << lev << " " << ba << std::endl;

    phi_new[lev].define(ba, dm, ncomp, ng);
    phi_old[lev].define(ba, dm, ncomp, ng);

    t_new[lev] = time;
    t_old[lev] = time - amrex::Real(1.e200);

    if (lev > 0 && do_reflux) {
        flux_reg[lev] = std::make_unique<FluxRegister>(ba, dm, refRatio(lev-1), lev, ncomp);
    }

    InitMLHistoryLevel(lev, ba, dm);

    FillCoarsePatch(lev, time, phi_new[lev], 0, ncomp);
}

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data (overrides the pure virtual function in AmrCore)
// regrid  --> RemakeLevel            (if level already existed)
// regrid  --> MakeNewLevelFromCoarse (if adding new level)
void
AmrCoreAdv::RemakeLevel (int lev, Real time, const BoxArray& ba,
                         const DistributionMapping& dm)
{
    const int ncomp = phi_new[lev].nComp();
    const int ng = phi_new[lev].nGrow();

    BoxArray old_fine_ba = phi_old[1].boxArray();
    amrex::Print() << " REGRIDDING: NEW GRIDS AT LEVEL " << lev << " " << ba << std::endl;

    if (lev == 1) {
        MakeFBA(ba);
    }

    MultiFab new_state(ba, dm, ncomp, ng);
    MultiFab old_state(ba, dm, ncomp, ng);

    // Must use fillpatch_function
    FillPatch(lev, time, new_state, 0, ncomp, FillPatchType::fillpatch_function);

    std::swap(new_state, phi_new[lev]);
    std::swap(old_state, phi_old[lev]);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    if (lev > 0 && do_reflux) {
        flux_reg[lev] = std::make_unique<FluxRegister>(ba, dm, refRatio(lev-1), lev, ncomp);
    }

    if (m_flux_mode == FluxMode::ml && m_ml_history_len > 0) {
        auto new_hist = std::make_unique<MultiFab>(ba, dm, m_ml_history_len, m_ml_history_ngrow);
        new_hist->setVal(amrex::Real(0.0));
        if (m_phi_hist[lev]) {
            new_hist->ParallelCopy(*m_phi_hist[lev], 0, 0, m_ml_history_len);
        }
        m_phi_hist[lev].swap(new_hist);

        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            BoxArray fba = ba;
            fba.surroundingNodes(d);
            auto new_flux_hist = std::make_unique<MultiFab>(fba, dm, m_ml_history_len, m_ml_history_ngrow);
            new_flux_hist->setVal(amrex::Real(0.0));
            if (m_flux_hist[lev][d]) {
                new_flux_hist->ParallelCopy(*m_flux_hist[lev][d], 0, 0, m_ml_history_len);
            }
            m_flux_hist[lev][d].swap(new_flux_hist);
        }
    }

#ifdef AMREX_PARTICLES
        if (lev == 1) {
            particleData.regrid_particles(grown_fba, ba, old_fine_ba, phi_new[1], time);
        }
#endif
}

// Delete level data
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::ClearLevel (int lev)
{
    phi_new[lev].clear();
    phi_old[lev].clear();
    if (m_flux_mode == FluxMode::ml) {
        m_phi_hist[lev].reset();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            m_flux_hist[lev][d].reset();
        }
        m_ml_hist_count[lev] = 0;
    }
    flux_reg[lev].reset(nullptr);
    fillpatcher[lev].reset(nullptr);
}

// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization.
// overrides the pure virtual function in AmrCore
// main.cpp --> AmrCoreAdv::InitData --> InitFromScratch --> MakeNewGrids --> MakeNewLevelFromScratch
//                                              restart  --> MakeNewGrids --> MakeNewLevelFromScratch
void AmrCoreAdv::MakeNewLevelFromScratch (int lev, Real time, const BoxArray& ba,
                                          const DistributionMapping& dm)
{
    const int ng = 1;

    amrex::Print() << " GRIDS AT LEVEL " << lev << " " << ba << std::endl;

    // ncomp = number of components for each array
    int ncomp;
    if (alg_type == 0) {
       ncomp = 1;
    } else {
       ncomp = 2;
    }

    if (lev == 1) {
        MakeFBA(ba);
    }

    phi_new[lev].define(ba, dm, ncomp, ng);
    phi_old[lev].define(ba, dm, ncomp, ng);

    t_new[lev] = time;
    t_old[lev] = time - amrex::Real(1.e200);

    if (lev > 0 && do_reflux) {
        flux_reg[lev] = std::make_unique<FluxRegister>(ba, dm, refRatio(lev-1), lev, ncomp);
    }

    InitMLHistoryLevel(lev, ba, dm);

    const auto problo = Geom(lev).ProbLoArray();
    const auto dx     = Geom(lev).CellSizeArray();

    int Ncomp = phi_new[lev].nComp();

    ExternalPotential external_potential = m_external_potential;

    if (lev == 0) {
        if (m_init_type != InitType::uniform) {
            ValidateInitializationParameters();
        }

        amrex::Gpu::DeviceVector<amrex::Real> init_positions_d;
        amrex::Gpu::DeviceVector<amrex::Real> init_particles_d;
        amrex::Real const* init_positions_ptr = nullptr;
        amrex::Real const* init_particles_ptr = nullptr;
        int num_init_positions = 0;
        if (m_init_type == InitType::piecewise_x) {
            init_positions_d.resize(m_init_positions.size());
            init_particles_d.resize(m_init_particles_per_interval.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             m_init_positions.begin(), m_init_positions.end(),
                             init_positions_d.begin());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             m_init_particles_per_interval.begin(), m_init_particles_per_interval.end(),
                             init_particles_d.begin());
            init_positions_ptr = init_positions_d.data();
            init_particles_ptr = init_particles_d.data();
            num_init_positions = static_cast<int>(m_init_positions.size());
        }

        if (m_init_type == InitType::external_potential_x) {
            InitializeExternalPotentialXLevel0(phi_new[lev], time);
        } else {
            for (MFIter mfi(phi_new[lev]); mfi.isValid(); ++mfi)
            {
                const Box& vbx = mfi.validbox();
                auto const& phi_arr = phi_new[lev].array(mfi);
                auto npts_scale_local = npts_scale;
                auto init_type = m_init_type;
                amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (init_type == InitType::piecewise_x) {
                        amrex::Real cellvol = dx[0]*dx[1];
#if (AMREX_SPACEDIM > 2)
                        cellvol *= dx[2];
#endif
                        amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
                        int interval = num_init_positions;
                        for (int n = 0; n < num_init_positions; ++n) {
                            if (x < init_positions_ptr[n]) {
                                interval = n;
                                break;
                            }
                        }

                        phi_arr(i,j,k,0) = init_particles_ptr[interval] / cellvol;
                        if (Ncomp == 2) {
                            phi_arr(i,j,k,1) = phi_arr(i,j,k,0);
                        }
                    } else {
                        init_phi(i,j,k,phi_arr,dx,problo,npts_scale_local,Ncomp,
                                 external_potential);
                    }
                });
            }
        }
    } else {
        phi_new[lev].ParallelCopy(phi_new[lev-1], 0, 0, phi_new[lev].nComp());
    }
}

void
AmrCoreAdv::InitializeExternalPotentialXLevel0 (amrex::MultiFab& phi, amrex::Real time)
{
#if (AMREX_SPACEDIM != 2)
    amrex::ignore_unused(phi, time);
    amrex::Abort("external_potential_x initialization is only implemented for AMREX_SPACEDIM == 2.");
#else
    const auto prob_lo = Geom(0).ProbLoArray();
    const auto dx = Geom(0).CellSizeArray();
    const Box& domain = Geom(0).Domain();
    const amrex::Real x_min = m_init_x_range[0];
    const amrex::Real x_max = m_init_x_range[1];
    const int ncomp = phi.nComp();
    const bool y_is_ensemble = (m_ensemble_dir[1] != 0);
    EnsembleDirection ens_dir{AMREX_D_DECL(m_ensemble_dir[0], m_ensemble_dir[1], m_ensemble_dir[2])};

    const amrex::Real cell_volume = y_is_ensemble ? dx[0] : dx[0] * dx[1];

    const BoxArray& ba = phi.boxArray();
    amrex::Long num_selected_cells = 0;
    amrex::Real min_potential = std::numeric_limits<amrex::Real>::max();
    bool use_overlap_selection = false;

    auto cell_is_selected = [&] (int i) noexcept
    {
        const amrex::Real center_x = prob_lo[0]
            + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
        if (!use_overlap_selection) {
            return center_x >= x_min && center_x <= x_max;
        }

        const amrex::Real cell_lo = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
        const amrex::Real cell_hi = cell_lo + dx[0];
        return cell_hi > x_min && cell_lo < x_max;
    };

    auto accumulate_selected_cells = [&] ()
    {
        num_selected_cells = 0;
        min_potential = std::numeric_limits<amrex::Real>::max();

        if (y_is_ensemble) {
            const int ilo = domain.smallEnd(0);
            const int ihi = domain.bigEnd(0);
            const amrex::Real y_ref = prob_lo[1]
                + (static_cast<amrex::Real>(domain.smallEnd(1)) + amrex::Real(0.5)) * dx[1];
            for (int i = ilo; i <= ihi; ++i) {
                if (!cell_is_selected(i)) {
                    continue;
                }

                const amrex::Real x = prob_lo[0]
                    + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
                const amrex::Real potential =
                    external_potential_value(m_external_potential, ens_dir, time, x, y_ref);
                min_potential = std::min(min_potential, potential);
                ++num_selected_cells;
            }
        } else {
            // Traverse the global BoxArray so initialization is independent of the
            // current DistributionMapping.
            for (int bi = 0; bi < ba.size(); ++bi) {
                const Box& vbx = ba[bi];
                for (int j = vbx.smallEnd(1); j <= vbx.bigEnd(1); ++j) {
                    for (int i = vbx.smallEnd(0); i <= vbx.bigEnd(0); ++i) {
                        if (!cell_is_selected(i)) {
                            continue;
                        }

                        const amrex::Real x = prob_lo[0]
                            + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
                        const amrex::Real y = prob_lo[1]
                            + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
                        const amrex::Real potential =
                            external_potential_value(m_external_potential, ens_dir, time, x, y);
                        min_potential = std::min(min_potential, potential);
                        ++num_selected_cells;
                    }
                }
            }
        }
    };

    accumulate_selected_cells();
    if (num_selected_cells == 0) {
        use_overlap_selection = true;
        accumulate_selected_cells();
    }

    if (num_selected_cells == 0) {
        std::ostringstream oss;
        oss << "external_potential_x initialization selected zero level-0 cells. "
            << "init_x_range=[" << x_min << ", " << x_max << "], "
            << "geom_prob_x=[" << Geom(0).ProbLo(0) << ", " << Geom(0).ProbHi(0) << "], "
            << "dx=" << dx[0] << ", "
            << "domain_i=[" << domain.smallEnd(0) << ", " << domain.bigEnd(0) << "], "
            << "first_center="
            << (prob_lo[0] + (static_cast<amrex::Real>(domain.smallEnd(0)) + amrex::Real(0.5)) * dx[0])
            << ", last_center="
            << (prob_lo[0] + (static_cast<amrex::Real>(domain.bigEnd(0)) + amrex::Real(0.5)) * dx[0]);
        amrex::Abort(oss.str());
    }
    amrex::ParallelDescriptor::ReduceRealMin(min_potential);

    amrex::Real weight_sum = amrex::Real(0.0);
    if (y_is_ensemble) {
        const int ilo = domain.smallEnd(0);
        const int ihi = domain.bigEnd(0);
        const amrex::Real y_ref = prob_lo[1]
            + (static_cast<amrex::Real>(domain.smallEnd(1)) + amrex::Real(0.5)) * dx[1];
        for (int i = ilo; i <= ihi; ++i) {
            if (!cell_is_selected(i)) {
                continue;
            }

            const amrex::Real x = prob_lo[0]
                + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            const amrex::Real potential =
                external_potential_value(m_external_potential, ens_dir, time, x, y_ref);
            weight_sum += std::exp(amrex::Real(-2.0) * (potential - min_potential));
        }
    } else {
        for (int bi = 0; bi < ba.size(); ++bi) {
            const Box& vbx = ba[bi];
            for (int j = vbx.smallEnd(1); j <= vbx.bigEnd(1); ++j) {
                for (int i = vbx.smallEnd(0); i <= vbx.bigEnd(0); ++i) {
                    if (!cell_is_selected(i)) {
                        continue;
                    }

                    const amrex::Real x = prob_lo[0]
                        + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
                    const amrex::Real y = prob_lo[1]
                        + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
                    const amrex::Real potential =
                        external_potential_value(m_external_potential, ens_dir, time, x, y);
                    weight_sum += std::exp(amrex::Real(-2.0) * (potential - min_potential));
                }
            }
        }
    }

    if (!(weight_sum > amrex::Real(0.0))) {
        amrex::Abort("external_potential_x initialization produced zero total probability weight.");
    }

    phi.setVal(amrex::Real(0.0));
    const ExternalPotential external_potential = m_external_potential;
    const amrex::Real total_particles = m_init_total_particles;
    const amrex::Real min_potential_local = min_potential;
    const amrex::Real weight_sum_local = weight_sum;
    const bool use_overlap_selection_local = use_overlap_selection;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& vbx = mfi.validbox();
        auto const& phi_arr = phi.array(mfi);
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::Real center_x = prob_lo[0]
                + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0];
            bool cell_selected = center_x >= x_min && center_x <= x_max;
            if (use_overlap_selection_local) {
                const amrex::Real cell_lo = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
                const amrex::Real cell_hi = cell_lo + dx[0];
                cell_selected = cell_hi > x_min && cell_lo < x_max;
            }
            if (!cell_selected) {
                return;
            }

            const amrex::Real x = center_x;
            const amrex::Real y = prob_lo[1]
                + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
            const amrex::Real potential =
                external_potential_value(external_potential, ens_dir, time, x, y);
            const amrex::Real weight =
                std::exp(amrex::Real(-2.0) * (potential - min_potential_local));
            const amrex::Real density =
                total_particles * weight / (weight_sum_local * cell_volume);
            phi_arr(i,j,k,0) = density;
            if (ncomp == 2) {
                phi_arr(i,j,k,1) = density;
            }
        });
    }
#endif
}

void
AmrCoreAdv::InitMLHistoryLevel (int lev, const BoxArray& ba, const DistributionMapping& dm)
{
    if (m_flux_mode != FluxMode::ml || m_ml_history_len <= 0) {
        return;
    }

    m_phi_hist[lev] = std::make_unique<MultiFab>(ba, dm, m_ml_history_len, m_ml_history_ngrow);
    m_phi_hist[lev]->setVal(amrex::Real(0.0));

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        BoxArray fba = ba;
        fba.surroundingNodes(d);
        m_flux_hist[lev][d] = std::make_unique<MultiFab>(fba, dm, m_ml_history_len, 0);
        m_flux_hist[lev][d]->setVal(amrex::Real(0.0));
    }

    m_ml_hist_count[lev] = 0;
}

// phi and flux histories are being updated at different locations because
// flux history should NOT include external potential effects
// Furthermore, flux history is in NET PARTICLES CROSSING and NOT NUMBER DENSITY
void
AmrCoreAdv::UpdateMLPhiHistory (int lev)
{
    if (m_flux_mode != FluxMode::ml || m_ml_history_len <= 0) {
        return;
    }

    AMREX_ASSERT(m_phi_hist[lev]);

    auto& phi_hist = *m_phi_hist[lev];
    const int hist_len = m_ml_history_len;
    const int count = m_ml_hist_count[lev];

    if (count < hist_len) {
        const int head = count;

        amrex::MultiFab::Copy(phi_hist, phi_old[lev], 0, head, 1, 0);
        phi_hist.FillBoundary(Geom(lev).periodicity());

        m_ml_hist_count[lev] = count + 1;
    }
    else {
        if (hist_len == 1) {
            amrex::MultiFab::Copy(phi_hist, phi_old[lev], 0, 0, 1, 0);
        } else {
            // History full: shift left by 1 to keep most recent hist_len entries.
            amrex::MultiFab phi_hist_copy(phi_hist.boxArray(),
                                          phi_hist.DistributionMap(),
                                          phi_hist.nComp()-1, 0);
            amrex::MultiFab::Copy(phi_hist_copy, phi_hist, 1, 0,
                                  phi_hist.nComp()-1, 0);
            amrex::MultiFab::Swap(phi_hist, phi_hist_copy, 0, 0,
                                  phi_hist.nComp()-1, 0);

            const int head = hist_len - 1;
            amrex::MultiFab::Copy(phi_hist, phi_old[lev], 0, head, 1, 0);
        }

        phi_hist.FillBoundary(Geom(lev).periodicity());

        m_ml_hist_count[lev] = hist_len;
    }
}

// tag all cells for refinement
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::ErrorEst (int lev, TagBoxArray& tags, Real /*time*/, int /*ngrow*/)
{
    static bool first = true;
    static Vector<Real> phierr;

    // only do this during the first call to ErrorEst
    if (first)
    {
        first = false;
        // read in an array of "phierr", which is the tagging threshold
        // in this example, we tag values of "phi" which are greater than phierr
        // for that particular level
        // in subroutine state_error, you could use more elaborate tagging, such
        // as more advanced logical expressions, or gradients, etc.
        ParmParse pp("adv");
        int n = pp.countval("phierr");
        if (n > 0) {
            pp.getarr("phierr", phierr, 0, n);
        }
    }

    if (lev >= phierr.size()) { return; }

//    const int clearval = TagBox::CLEAR;
    const int   tagval = TagBox::SET;

    const MultiFab& state = phi_new[lev];

    const Real* dx  =  geom[lev].CellSize();
#if (AMREX_SPACEDIM == 2)
    const Real cell_vol = dx[0]*dx[1];
#else
    const Real cell_vol = dx[0]*dx[1]*dx[2];
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {

        for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx  = mfi.tilebox();
            const auto statefab = state.array(mfi);
            const auto tagfab  = tags.array(mfi);
            Real phierror = phierr[lev]/cell_vol;

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                state_error(i, j, k, tagfab, statefab, phierror, tagval);
            });
        }
    }
}

// read in some parameters from inputs file
void
AmrCoreAdv::ReadParameters ( amrex::Vector<int>& bc_lo, amrex::Vector<int>& bc_hi)
{
    {
        ParmParse pp;  // Traditionally, max_step and stop_time do not have prefix.
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);

        npts_scale = amrex::Real(1.0);
        pp.queryAdd("npts_scale", npts_scale);

        alg_type = 0;
        pp.queryAdd("alg_type", alg_type);

        m_init_type = InitType::uniform;
        pp.query("init_type", m_init_type);
        pp.queryarr("init_positions", m_init_positions);
        pp.queryarr("init_particles_per_interval", m_init_particles_per_interval);
        pp.queryarr("init_x_range", m_init_x_range);
        pp.query("init_total_particles", m_init_total_particles);

        m_flux_mode = FluxMode::gaussian;
        pp.query("flux_mode", m_flux_mode);

        pp.query("ml_model_file", m_ml_model_file);
        m_ml_load_batch_size = 1;
        pp.query("ml_load_batch_size", m_ml_load_batch_size);
        m_ml_history_len = 10;
        pp.query("ml_history_len", m_ml_history_len);
        m_ml_history_ngrow = 1;
        pp.query("ml_history_ngrow", m_ml_history_ngrow);
        if (m_ml_history_len < 0 || m_ml_history_ngrow < 0) {
            amrex::Abort("ml_history_len and ml_history_ngrow must be non-negative.");
        }
        if (m_flux_mode == FluxMode::ml && m_ml_history_len < 0) {
            amrex::Abort("ml_history_len must be >= 0 when flux_mode=ml.");
        }
        if (m_flux_mode == FluxMode::ml && m_ml_history_ngrow < 1) {
            amrex::Abort("ml_history_ngrow must be >= 1 when flux_mode=ml.");
        }
        if (m_ml_load_batch_size < 1) {
            amrex::Abort("ml_load_batch_size must be >= 1.");
        }

        m_ml_flow_steps = 100;
        pp.query("ml_flow_steps", m_ml_flow_steps);
        m_ml_flow_t0 = amrex::Real(0.0);
        pp.query("ml_flow_t0", m_ml_flow_t0);
        m_ml_flow_t1 = amrex::Real(1.0);
        pp.query("ml_flow_t1", m_ml_flow_t1);
        if (m_ml_flow_steps < 1) {
            amrex::Abort("ml_flow_steps must be >= 1.");
        }
        if (m_ml_flow_t1 <= m_ml_flow_t0) {
            amrex::Abort("ml_flow_t1 must be greater than ml_flow_t0.");
        }

        m_ml_t_df = amrex::Real(4.0);
        pp.query("ml_t_df", m_ml_t_df);
        m_ml_t_loc = amrex::Real(0.0);
        pp.query("ml_t_loc", m_ml_t_loc);
        m_ml_t_scale = amrex::Real(1.0);
        pp.query("ml_t_scale", m_ml_t_scale);
        m_ml_input_scale = amrex::Real(51.0);
        pp.query("ml_input_scale", m_ml_input_scale);
        m_ml_output_mn_fctr = amrex::Real(0.069);
        pp.query("ml_output_mn_fctr", m_ml_output_mn_fctr);
        m_ml_output_std_fctr = amrex::Real(0.2537);
        pp.query("ml_output_std_fctr", m_ml_output_std_fctr);
        m_quantize_ml_output = false;
        pp.query("quantize_ml_output", m_quantize_ml_output);
        if (m_ml_t_df <= 0.0) {
            amrex::Abort("ml_t_df must be > 0.");
        }
        if (m_ml_t_scale <= 0.0) {
            amrex::Abort("ml_t_scale must be > 0.");
        }
        if (m_ml_input_scale <= 0.0) {
            amrex::Abort("ml_input_scale must be > 0.");
        }
        if (m_ml_output_std_fctr <= 0.0) {
            amrex::Abort("ml_output_std_fctr must be > 0.");
        }

        pp.query("diag_enable", m_diag.enabled);
        pp.query("diag_flux_x", m_diag.flux_x);
        pp.query("diag_x_min", m_diag.x_min);
        pp.query("diag_x_max", m_diag.x_max);
        pp.query("diag_file", m_diag.file);

        // read in BC; see Src/Base/AMReX_BC_TYPES.H for supported types
        pp.queryarr("bc_lo", bc_lo);
        pp.queryarr("bc_hi", bc_hi);

        seed = 0;
        pp.queryAdd("seed", seed);

        // read in if a direction is ensemble direction
        pp.queryarr("is_ensemble_dir", m_ensemble_dir, 0, AMREX_SPACEDIM);
        // Some asserts for m_ensemble_dir
        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
            if (m_ensemble_dir[idir]) {
                amrex::Print()<<"max_level: "<<max_level<<"\n";
                AMREX_ALWAYS_ASSERT(max_level == 0);
                AMREX_ALWAYS_ASSERT(Geom(0).CellSize(idir) == Real(1.0));
            }
        }

        pp.get("diffusion_coefficient",m_d_spde);
    }

    {
        ParmParse pp("amr"); // Traditionally, these have prefix, amr.

        pp.query("regrid_int", regrid_int);
        pp.query("plot_file", plot_file);
        pp.query("plot_int", plot_int);
        pp.query("chk_file", chk_file);
        pp.query("chk_int", chk_int);
        pp.query("restart",restart_chkfile);
    }

    {
        ParmParse pp("adv");

        pp.query("cfl", cfl);
        pp.query("do_reflux", do_reflux);
        pp.query("do_subcycle", do_subcycle);
    }

    {
        ParmParse pp("ext_pot");

        pp.query("exists", m_external_potential.enabled);
        if (m_external_potential.enabled) {
            if (AMREX_SPACEDIM != 2) {
                amrex::Abort("External Potential is coded for 2D.\n");
            }
            std::string potential_type = "quartic_2d";
            pp.query("type", potential_type);
            m_external_potential.type = ExternalPotentialTypeFromString(potential_type);
            if (m_external_potential.type != ExternalPotentialType::quartic_2d) {
                amrex::Abort("Unsupported ext_pot.type: " + potential_type);
            }
            pp.get("alpha", m_external_potential.alpha);
            pp.get("beta", m_external_potential.beta);
            pp.get("gamma", m_external_potential.gamma);
        } else {
            m_external_potential.type = ExternalPotentialType::none;
        }
    }

#ifdef AMREX_PARTICLES
        int a_ensemble_dir_exists = 0;
        for (int edir : m_ensemble_dir) {
            a_ensemble_dir_exists += edir;
        }
        if (a_ensemble_dir_exists) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(alg_type != 0,
                "Ensemble mode with particles requires alg_type != 0, i.e., ncomp = 2");
        }
        particleData.init_particle_params(max_level, a_ensemble_dir_exists,
                                          m_external_potential);
#endif
}

void
AmrCoreAdv::ValidateInitializationParameters () const
{
    if (m_init_type == InitType::uniform) {
        return;
    }

    if (m_init_type == InitType::external_potential_x) {
        if (!m_external_potential.enabled) {
            amrex::Abort("external_potential_x initialization requires ext_pot.exists = 1.");
        }
        if (AMREX_SPACEDIM != 2) {
            amrex::Abort("external_potential_x initialization is only implemented for AMREX_SPACEDIM == 2.");
        }
        if (m_init_x_range.size() != 2) {
            amrex::Abort("external_potential_x initialization requires init_x_range to contain exactly two entries.");
        }
        if (m_ensemble_dir[0] != 0) {
            amrex::Abort("external_potential_x initialization does not support x as an ensemble direction.");
        }

        const amrex::Real x_min = m_init_x_range[0];
        const amrex::Real x_max = m_init_x_range[1];
        const amrex::Real prob_lo = Geom(0).ProbLo(0);
        const amrex::Real prob_hi = Geom(0).ProbHi(0);
        if (x_min >= x_max) {
            amrex::Abort("external_potential_x initialization requires init_x_range[0] < init_x_range[1].");
        }
        if (x_min < prob_lo || x_max > prob_hi) {
            amrex::Abort("external_potential_x initialization requires init_x_range to lie inside the level-0 x-domain.");
        }
        if (m_init_total_particles < amrex::Real(0.0)) {
            amrex::Abort("external_potential_x initialization requires init_total_particles >= 0.");
        }

        const amrex::Real rounded_total = std::round(m_init_total_particles);
        const amrex::Real tol = amrex::Real(1.e-12)
            * std::max(amrex::Real(1.0), std::abs(m_init_total_particles));
        if (std::abs(m_init_total_particles - rounded_total) > tol) {
            amrex::Abort("external_potential_x initialization requires init_total_particles to be an integer value.");
        }
        return;
    }

    if (m_init_particles_per_interval.size() != m_init_positions.size() + 1) {
        amrex::Abort("piecewise_x initialization requires init_particles_per_interval to have exactly one more entry than init_positions.");
    }

    const amrex::Real prob_lo = Geom(0).ProbLo(0);
    const amrex::Real prob_hi = Geom(0).ProbHi(0);
    amrex::Real prev = prob_lo;
    for (amrex::Real xpos : m_init_positions) {
        if (xpos < prob_lo || xpos > prob_hi) {
            amrex::Abort("All init_positions entries must lie inside the level-0 x-domain.");
        }
        if (xpos <= prev) {
            amrex::Abort("init_positions must be strictly increasing.");
        }
        prev = xpos;
    }
}

void
AmrCoreAdv::ValidateDiagnosticsMode () const
{
    if (!m_diag.enabled) {
        return;
    }

    if (alg_type == 0) {
        amrex::Abort("Diagnostics require alg_type != 0.");
    }

    if (max_level != 0) {
        amrex::Abort("Diagnostics require amr.max_level = 0.");
    }

#ifdef AMREX_PARTICLES
    if (!particleData.UsingParticles()) {
        amrex::Abort("Diagnostics require amr.use_particles = 1.");
    }
#else
    amrex::Abort("Diagnostics require particle support.");
#endif
}

int
AmrCoreAdv::MapPhysicalXToFaceIndex (amrex::Geometry const& geom, amrex::Real x) const
{
    const amrex::Real prob_lo = geom.ProbLo(0);
    const amrex::Real prob_hi = geom.ProbHi(0);
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Box& domain = geom.Domain();

    if (x < prob_lo - 1.e-12 || x > prob_hi + 1.e-12) {
        amrex::Abort("diag_flux_x is outside the level-0 x-domain.");
    }

    const amrex::Real rel = (x - prob_lo) / dx;
    const int face = static_cast<int>(std::llround(rel)) + domain.smallEnd(0);
    const int face_lo = domain.smallEnd(0);
    const int face_hi = domain.bigEnd(0) + 1;
    if (face < face_lo || face > face_hi) {
        amrex::Abort("diag_flux_x does not map to a valid level-0 x-face.");
    }

    return face;
}

void
AmrCoreAdv::ConfigureParticleDiagnostics ()
{
    if (!m_diag.enabled) {
        return;
    }

#ifdef AMREX_PARTICLES
    ParticleData::FaceFluxDiagnosticsConfig config;
    config.enabled = true;
    config.face_x = Geom(0).ProbLo(0)
        + static_cast<amrex::Real>(m_diag.face_index - Geom(0).Domain().smallEnd(0)) * Geom(0).CellSize(0);
    particleData.ConfigureFaceFluxDiagnostics(config);
#endif
}

void
AmrCoreAdv::InitDiagnostics ()
{
    if (!m_diag.enabled) {
        return;
    }

    ValidateDiagnosticsMode();
    if (m_diag.x_min > m_diag.x_max) {
        amrex::Abort("diag_x_min must be <= diag_x_max.");
    }
    if (m_diag.x_min < Geom(0).ProbLo(0) || m_diag.x_max > Geom(0).ProbHi(0)) {
        amrex::Abort("diag_x_min/diag_x_max must lie inside the level-0 x-domain.");
    }
    m_diag.face_index = MapPhysicalXToFaceIndex(Geom(0), m_diag.flux_x);
    const int ny = Geom(0).Domain().length(1);
    m_diag.spde_face_flux.assign(ny, amrex::Real(0.0));
    m_diag.particle_face_flux.assign(ny, amrex::Real(0.0));
    m_diag.spde_reduced_density.assign(ny, amrex::Real(0.0));
    m_diag.particle_reduced_density.assign(ny, amrex::Real(0.0));
    m_diag.header_written = false;
    if (restart_chkfile.empty() && amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream os(m_diag.file, std::ios::out | std::ios::trunc);
        if (!os.good()) {
            amrex::Abort("Unable to initialize diagnostics file.");
        }
    }
    ConfigureParticleDiagnostics();
}

void
AmrCoreAdv::CaptureSPDEFaceFluxProfile (int lev,
                                        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& fluxes)
{
    if (!m_diag.enabled) {
        return;
    }

    if (lev != 0) {
        return;
    }

    CaptureSPDEFaceFluxProfileImpl(m_diag.face_index, Geom(lev), fluxes[0], m_diag.spde_face_flux);
}

void
AmrCoreAdv::ComputeReducedDensities (int lev)
{
    if (!m_diag.enabled) {
        return;
    }

    std::fill(m_diag.spde_reduced_density.begin(), m_diag.spde_reduced_density.end(), amrex::Real(0.0));
    std::fill(m_diag.particle_reduced_density.begin(), m_diag.particle_reduced_density.end(), amrex::Real(0.0));
    ComputeReducedDensitiesImpl(phi_new[lev], Geom(lev), m_diag.x_min, m_diag.x_max,
                                m_diag.spde_reduced_density, m_diag.particle_reduced_density);

#ifdef AMREX_PARTICLES
    auto const& particle_flux = particleData.GetFaceFluxDiagnosticsProfile();
    std::fill(m_diag.particle_face_flux.begin(), m_diag.particle_face_flux.end(), amrex::Real(0.0));
    const std::size_t ncopy = std::min(static_cast<std::size_t>(m_diag.particle_face_flux.size()),
                                       static_cast<std::size_t>(particle_flux.size()));
    for (std::size_t n = 0; n < ncopy; ++n) {
        m_diag.particle_face_flux[n] = particle_flux[n];
    }
#endif
}

void
AmrCoreAdv::WriteStepDiagnostics (int lev, amrex::Real time)
{
    if (!m_diag.enabled) {
        return;
    }

    if (lev != 0 || !amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }

    std::ios_base::openmode mode = std::ios::out | std::ios::app;
    bool write_header = false;
    {
        std::ifstream is(m_diag.file, std::ios::in | std::ios::ate);
        if (!is.good()) {
            write_header = true;
        } else if (is.tellg() == std::streampos(0)) {
            write_header = true;
        }
    }
    std::ofstream os(m_diag.file, mode);
    if (!os.good()) {
        amrex::Abort("Unable to open diagnostics file.");
    }

    if (write_header) {
        os << "# step time y_index y_coord spde_flux_x particle_crossing_flux "
              "spde_reduced_density particle_reduced_density\n";
    }
    m_diag.header_written = true;

    const amrex::Geometry& gm = Geom(lev);
    const amrex::Real ylo = gm.ProbLo(1);
    const amrex::Real dy = gm.CellSize(1);
    const amrex::Box& domain = gm.Domain();
    const int jlo = domain.smallEnd(1);
    const int ny = domain.length(1);
    const std::size_t rows = std::min({static_cast<std::size_t>(m_diag.spde_face_flux.size()),
                                       static_cast<std::size_t>(m_diag.particle_face_flux.size()),
                                       static_cast<std::size_t>(m_diag.spde_reduced_density.size()),
                                       static_cast<std::size_t>(m_diag.particle_reduced_density.size())});

    os << std::setprecision(17);
    for (int n = 0; n < amrex::min(ny, static_cast<int>(rows)); ++n) {
        const int j = jlo + n;
        const amrex::Real y = ylo + (static_cast<amrex::Real>(j - jlo) + amrex::Real(0.5)) * dy;
        os << istep[lev] + 1 << " "
           << time << " "
           << j << " "
           << y << " "
           << m_diag.spde_face_flux[n] << " "
           << m_diag.particle_face_flux[n] << " "
           << m_diag.spde_reduced_density[n] << " "
           << m_diag.particle_reduced_density[n] << "\n";
    }
}

// set covered coarse cells to be the average of overlying fine cells
void
AmrCoreAdv::AverageDown ()
{
    for (int lev = finest_level; lev >= 1; --lev) {
        phi_new[lev-1].ParallelCopy(phi_new[lev], 0, 0, phi_new[lev].nComp());
    }
}

// more flexible version of AverageDown() that lets you average down across multiple levels
void
AmrCoreAdv::AverageDownTo (int crse_lev)
{
    phi_new[crse_lev].ParallelCopy(phi_new[crse_lev+1], 0, 0, phi_new[crse_lev].nComp());
}

// compute a new multifab by coping in phi from valid region and filling ghost cells
// works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
void
AmrCoreAdv::FillPatch (int lev, Real time, MultiFab& mf, int icomp, int ncomp,
                       FillPatchType fptype)
{
    if (lev == 0)
    {
        Vector<MultiFab*> smf;
        Vector<Real> stime;
        GetData(0, time, smf, stime);

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > physbc(geom[lev],bcs,gpu_bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                        geom[lev], physbc, 0);
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev],bcs,bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                        geom[lev], physbc, 0);
        }
    }
    else
    {
        Vector<MultiFab*> cmf, fmf;
        Vector<Real> ctime, ftime;
        GetData(lev-1, time, cmf, ctime);
        GetData(lev  , time, fmf, ftime);

        Interpolater* mapper = &cell_cons_interp;

        if (fptype == FillPatchType::fillpatch_class) {
            if (fillpatcher[lev] == nullptr) {
                fillpatcher[lev] = std::make_unique<FillPatcher<MultiFab>>
                    (boxArray(lev  ), DistributionMap(lev  ), Geom(lev  ),
                     boxArray(lev-1), DistributionMap(lev-1), Geom(lev-1),
                     mf.nGrowVect(), mf.nComp(), mapper);
            }
        }

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

            if (fptype == FillPatchType::fillpatch_class) {
                fillpatcher[lev]->fill(mf, mf.nGrowVect(), time,
                                       cmf, ctime, fmf, ftime, 0, icomp, ncomp,
                                       cphysbc, 0, fphysbc, 0, bcs, 0);
            } else {
                amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                          0, icomp, ncomp, geom[lev-1], geom[lev],
                                          cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                          mapper, bcs, 0);
            }
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
            PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

            if (fptype == FillPatchType::fillpatch_class) {
                fillpatcher[lev]->fill(mf, mf.nGrowVect(), time,
                                       cmf, ctime, fmf, ftime, 0, icomp, ncomp,
                                       cphysbc, 0, fphysbc, 0, bcs, 0);
            } else {
                amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                          0, icomp, ncomp, geom[lev-1], geom[lev],
                                          cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                          mapper, bcs, 0);
            }
        }
    }
}

// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
void
AmrCoreAdv::FillCoarsePatch (int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    BL_ASSERT(lev > 0);

    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetData(lev-1, time, cmf, ctime);
    Interpolater* mapper = &cell_cons_interp;

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    if(Gpu::inLaunchRegion())
    {
        GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
    else
    {
        CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
        PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
        PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
}

void
AmrCoreAdv::GetData (int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    data.clear();
    datatime.clear();

    if (amrex::almostEqual(time, t_new[lev], 5))
    {
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_new[lev]);
    }
    else if (amrex::almostEqual(time, t_old[lev], 5))
    {
        data.push_back(&phi_old[lev]);
        datatime.push_back(t_old[lev]);
    }
    else
    {
        data.push_back(&phi_old[lev]);
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }
}

// Advance all the levels with the same dt
void
AmrCoreAdv::timeStepNoSubcycling (Real time, int iteration)
{
    if (max_level > 0 && regrid_int > 0)  // We may need to regrid
    {
        if (istep[0] % regrid_int == 0)
        {
            amrex::Print() << "Regridding at step " << istep[0] << std::endl;
            regrid(0, time);

            AverageDown();

            Real sum_phi_reg_new = phi_new[0].sum();
            Real sum_phi_reg_old = phi_old[0].sum();
            amrex::Print() << " Sum(Phi) new / old / diff / %diff  after regrid = " << std::setw(24) <<  std::setprecision(16) << std::scientific <<
                   sum_phi_reg_new << " " << sum_phi_reg_old << " " << (sum_phi_reg_new-sum_phi_reg_old) << " " <<
                   (sum_phi_reg_new-sum_phi_reg_old)/sum_phi_reg_old << std::endl;
        }
    }

    // Advance phi at level 0 only
    int  lev = 0;
    int nsub = 1;
    std::swap(phi_old[lev], phi_new[lev]);

    if (Verbose()) {
       amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
       amrex::Print() << "ADVANCE with time = " << t_new[lev] << " dt = " << dt[0] << std::endl;
    }

    AdvancePhiAtLevel(lev, time, dt[lev], iteration, nsub);

    if (finest_level > 0) {
        flux_reg[lev+1]->Reflux(phi_new[lev], amrex::Real(1.0), 0, 0, phi_new[lev].nComp(), geom[lev]);
    }

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
    }

    int lev_for_particles = finest_level;
#ifdef AMREX_PARTICLES
    // We set this to finest_level rather than 1 so that we can run with max_level = 0;
    //    in that case the advance_particle routine will return without doing anything
    if (finest_level > 0) {
        std::swap(phi_old[lev_for_particles], phi_new[lev_for_particles]);
    }
    const Real* dx  =  geom[lev_for_particles].CellSize();
#if (AMREX_SPACEDIM == 2)
    const Real cell_vol = dx[0]*dx[1];
#else
    const Real cell_vol = dx[0]*dx[1]*dx[2];
#endif
    if (finest_level > 0) {
        particleData.advance_particles(lev_for_particles, time, dt[lev_for_particles], cell_vol,
                                   phi_old[0], phi_new[0], phi_new[lev_for_particles]);
    }
    else {
        particleData.advance_particles(lev_for_particles, time, dt[lev_for_particles],
                                       cell_vol, phi_new[lev_for_particles],
                                       istep[lev_for_particles]);
    }
    particleData.Redistribute();
#endif

    ComputeReducedDensities(lev);
    WriteStepDiagnostics(lev, time + dt[lev]);

    // Make sure the coarser levels are consistent with the finer levels
    AverageDown ();

    for (auto& fp : fillpatcher) {
        fp.reset(); // Because the data have changed.
    }

    for (int ilev = 0; ilev <= finest_level; ilev++) {
        ++istep[ilev];
    }

    if (Verbose())
    {
        amrex::Print() << "[Level " << lev_for_particles << " step " << istep[lev_for_particles] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev_for_particles) << " cells" << std::endl;
    }
}

// a wrapper for EstTimeStep
void
AmrCoreAdv::ComputeDt ()
{
    Vector<Real> dt_tmp(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        dt_tmp[lev] = EstTimeStep(lev, t_new[lev]);
    }
    ParallelDescriptor::ReduceRealMin(dt_tmp.data(), int(dt_tmp.size()));

    constexpr Real change_max = 1.1;
    Real dt_0 = dt_tmp[0];
    int n_factor = 1;

    for (int lev = 0; lev <= finest_level; ++lev) {
        dt_tmp[lev] = std::min(dt_tmp[lev], change_max*dt[lev]);
        n_factor *= nsubsteps[lev];
        dt_0 = std::min(dt_0, n_factor*dt_tmp[lev]);
    }

    // Limit dt's by the value of stop_time.
    const Real eps = 1.e-3*dt_0;

    if (t_new[0] + dt_0 > stop_time - eps) {
        dt_0 = stop_time - t_new[0];
    }

    dt[0] = dt_0;

    for (int lev = 1; lev <= finest_level; ++lev) {
        dt[lev] = dt[lev-1] / nsubsteps[lev];
    }
}

// compute dt from CFL considerations
Real
AmrCoreAdv::EstTimeStep (int lev, Real /*time*/)
{
    BL_PROFILE("AmrCoreAdv::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();

    const Real* dx  =  geom[lev].CellSize();

    Real coeff = AMREX_D_TERM(   2./(dx[0]*dx[0]),
                               + 2./(dx[1]*dx[1]),
                               + 2./(dx[2]*dx[2]) );

    int ensemble_run = 0;
    for (int edir : m_ensemble_dir) {
        ensemble_run += edir;
    }
    if (ensemble_run) {
        coeff = 0.;
        for (int idir=0; idir < AMREX_SPACEDIM; ++idir) {
            if (m_ensemble_dir[idir] == 0) {
                coeff += 2./(dx[idir]*dx[idir]);
            }
        }
    }

    Real est = 1.0 / (2.0*coeff);
    dt_est = amrex::min(dt_est, est);

    dt_est *= cfl;

    return dt_est;
}

// write plotfile to disk
void
AmrCoreAdv::WritePlotFile () const
{
    const std::string& plotfilename = amrex::Concatenate(plot_file, istep[0], 6);

    // Vector of MultiFabs
    Vector<MultiFab> mf(finest_level+1);
    int ncomp_mf = 2; int src_comp = 0;
    // Check if this is an ensemble run
    int ensemble_run = 0;
    for (int edir : m_ensemble_dir) {
        ensemble_run += edir;
    }
    if (ensemble_run) { ncomp_mf += 1;}

    for (int lev = 0; lev <= finest_level; ++lev) {
        mf[lev].define(grids[lev], dmap[lev], ncomp_mf, 0);
        MultiFab::Copy(mf[lev],phi_new[lev],src_comp,0,1,0);
        MultiFab::Copy(mf[lev],phi_new[lev],src_comp,1,1,0);
        if (ensemble_run) {
            MultiFab::Copy(mf[lev],phi_new[lev],src_comp+1,2,1,0);
        }

        // Set the fine data in "phi0" to -1 so we can test on that value and plot particles over blank space
        if (lev == 1) {
            mf[lev].setVal(-1.0,1,1,0);
        }
    }


    Vector<std::string> varnames = {"phi", "phi0"};
    if (ensemble_run) {
        varnames.push_back("phi1");
    }

    amrex::Print() << "Writing plotfile " << plotfilename << "\n";

    if (finest_level == 0)
    {
        int fake_finest_level = 0;
        WriteMultiLevelPlotfile(plotfilename, fake_finest_level+1, GetVecOfConstPtrs(mf), varnames,
                                Geom(), t_new[0], istep, refRatio());
    } else {

        PhysBCFunctNoOp null_bc_for_fill;

        Vector<IntVect>   r2(finest_level);
        Vector<Geometry>  g2(finest_level+1);
        Vector<MultiFab> mf2(finest_level+1);

        mf2[0].define(grids[0], dmap[0], ncomp_mf, 0);

        // Copy level 0 as is
        MultiFab::Copy(mf2[0],mf[0],0,0,ncomp_mf,0);

        // Define a new multi-level array of Geometry's so that we pass the new "domain" at lev > 0
        Array<int,AMREX_SPACEDIM> periodicity =
                     {AMREX_D_DECL(Geom()[0].isPeriodic(0),Geom()[0].isPeriodic(1),Geom()[0].isPeriodic(2))};
        g2[0].define(Geom()[0].Domain(),&(Geom()[0].ProbDomain()),0,periodicity.data());

        r2[0] = IntVect(AMREX_D_DECL(2,2,2));
        for (int lev = 1; lev <= finest_level; ++lev) {
            if (lev > 1) {
                r2[lev-1][0] = r2[lev-2][0] * 2;
                r2[lev-1][1] = r2[lev-2][1] * 2;
#if (AMREX_SPACEDIM > 2)
                r2[lev-1][2] = r2[lev-2][2] * 2;
#endif
            }

            mf2[lev].define(refine(grids[lev],r2[lev-1]), dmap[lev], ncomp_mf, 0);

            // Set the new problem domain
            Box d2(Geom()[lev].Domain());
            d2.refine(r2[lev-1]);

            g2[lev].define(d2,&(Geom()[lev].ProbDomain()),0,periodicity.data());
        }

        amrex::Vector<amrex::BCRec> bcs_temp;
        bcs_temp.resize(2);     // Setup for 2 components in mf
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            bcs_temp[0].setLo(idim, bcs[0].lo()[idim]);
            bcs_temp[1].setLo(idim, bcs[0].lo()[idim]);

            bcs_temp[0].setHi(idim, bcs[0].hi()[idim]);
            bcs_temp[1].setHi(idim, bcs[0].hi()[idim]);
        }

        // Do piecewise interpolation of mf into mf2
        for (int lev = 1; lev <= finest_level; ++lev) {
            Interpolater* mapper_c = &pc_interp;
            InterpFromCoarseLevel(mf2[lev], t_new[lev], mf[lev],
                                  0, 0, ncomp_mf,
                                  geom[lev], g2[lev],
                                  null_bc_for_fill, 0, null_bc_for_fill, 0,
                                  r2[lev-1], mapper_c, bcs_temp, 0);
        }

        // Define an effective ref_ratio which is isotropic to be passed into WriteMultiLevelPlotfile
        Vector<IntVect> rr(finest_level);
        for (int lev = 0; lev < finest_level; ++lev) {
            rr[lev] = IntVect(AMREX_D_DECL(2,2,2));
        }

       WriteMultiLevelPlotfile(plotfilename, finest_level+1,
                                   GetVecOfConstPtrs(mf2), varnames,
                                   g2, t_new[0], istep, rr);

    }

#ifdef AMREX_PARTICLES
   particleData.writePlotFile(plotfilename, phi_new[finest_level]);
#endif
}

void
AmrCoreAdv::WriteCheckpointFile () const
{

    // chk00010            write a checkpoint file with this root directory
    // chk00010/Header     this contains information you need to save (e.g., finest_level, t_new, etc.) and also
    //                     the BoxArrays at each level
    // chk00010/Level_0/
    // chk00010/Level_1/
    // etc.                these subdirectories will hold the MultiFab data at each level of refinement

    // checkpoint file name, e.g., chk00010
    const std::string& checkpointname = amrex::Concatenate(chk_file,istep[0]);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

    const int nlevels = finest_level+1;

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

    // write Header file
   if (ParallelDescriptor::IOProcessor()) {

       std::string HeaderFileName(checkpointname + "/Header");
       VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
       std::ofstream HeaderFile;
       HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
       HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                                               std::ofstream::trunc |
                                               std::ofstream::binary);
       if( ! HeaderFile.good()) {
           amrex::FileOpenFailed(HeaderFileName);
       }

       HeaderFile.precision(17);

       // write out title line
       HeaderFile << "Checkpoint file for AmrCoreAdv\n";

       // write out finest_level
       HeaderFile << finest_level << "\n";

       // write out array of istep
       for (int i = 0; i < istep.size(); ++i) {
           HeaderFile << istep[i] << " ";
       }
       HeaderFile << "\n";

       // write out array of dt
       for (int i = 0; i < dt.size(); ++i) {
           HeaderFile << dt[i] << " ";
       }
       HeaderFile << "\n";

       // write out array of t_new
       for (int i = 0; i < t_new.size(); ++i) {
           HeaderFile << t_new[i] << " ";
       }
       HeaderFile << "\n";

       if (m_flux_mode == FluxMode::ml && m_ml_history_len > 0) {
           HeaderFile << "ml_history_len " << m_ml_history_len << "\n";
           HeaderFile << "ml_history_ngrow " << m_ml_history_ngrow << "\n";
           HeaderFile << "ml_hist_count ";
           for (int lev = 0; lev <= finest_level; ++lev) {
               HeaderFile << m_ml_hist_count[lev] << " ";
           }
           HeaderFile << "\n";
       }

       // write the BoxArray at each level
       for (int lev = 0; lev <= finest_level; ++lev) {
           boxArray(lev).writeOn(HeaderFile);
           HeaderFile << '\n';
       }
   }

   // write the MultiFab data to, e.g., chk00010/Level_0/
   for (int lev = 0; lev <= finest_level; ++lev) {
       VisMF::Write(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "phi"));
       if (m_flux_mode == FluxMode::ml && m_ml_history_len > 0) {
           VisMF::Write(*m_phi_hist[lev],
                        amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "phi_hist"));
           const char* flux_names[] = {"flux_hist_x", "flux_hist_y", "flux_hist_z"};
           for (int d = 0; d < AMREX_SPACEDIM; ++d) {
               VisMF::Write(*m_flux_hist[lev][d],
                            amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", flux_names[d]));
           }
       }
   }

#ifdef AMREX_PARTICLES
   particleData.Checkpoint(checkpointname);
#endif

}

namespace {
// utility to skip to next line in Header
void GotoNextLine (std::istream& is)
{
    constexpr std::streamsize bl_ignore_max { 100000 };
    is.ignore(bl_ignore_max, '\n');
}
}

void
AmrCoreAdv::ReadCheckpointFile ()
{
    amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

    // Header
    std::string File(restart_chkfile + "/Header");

    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    // read in title line
    std::getline(is, line);

    // read in finest_level
    is >> finest_level;
    GotoNextLine(is);

    // read in array of istep
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            istep[i++] = std::stoi(word);
        }
    }

    // read in array of dt
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            dt[i++] = std::stod(word);
        }
    }

    // read in array of t_new
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            t_new[i++] = std::stod(word);
        }
    }

    int header_ml_history_len = m_ml_history_len;
    int header_ml_history_ngrow = m_ml_history_ngrow;
    Vector<int> header_ml_hist_count;
    bool has_ml_header = false;
    std::streampos pos = is.tellg();
    std::string peek;
    std::getline(is, peek);
    if (peek.empty()) {
        std::getline(is, peek);
    }
    if (peek.rfind("ml_history_len", 0) == 0) {
        has_ml_header = true;
        {
            std::istringstream lis(peek);
            std::string key;
            lis >> key >> header_ml_history_len;
        }

        std::getline(is, line);
        {
            std::istringstream lis(line);
            std::string key;
            lis >> key >> header_ml_history_ngrow;
        }

        std::getline(is, line);
        {
            std::istringstream lis(line);
            std::string key;
            lis >> key;
            if (key == "ml_hist_head") {
                // Backward-compat: ignore stored head values.
                // Next line should contain ml_hist_count.
                std::getline(is, line);
                std::istringstream lis2(line);
                std::string key2;
                lis2 >> key2;
                int v;
                while (lis2 >> v) {
                    header_ml_hist_count.push_back(v);
                }
            } else if (key == "ml_hist_count") {
                int v;
                while (lis >> v) {
                    header_ml_hist_count.push_back(v);
                }
            }
        }
    } else {
        is.seekg(pos);
    }

    if (m_flux_mode == FluxMode::ml && has_ml_header) {
        m_ml_history_len = header_ml_history_len;
        m_ml_history_ngrow = header_ml_history_ngrow;
        if (m_ml_history_len < 0) {
            amrex::Abort("Restart checkpoint has invalid ml_history_len for flux_mode=ml.");
        }
        if (m_ml_history_ngrow < 1) {
            amrex::Abort("Restart checkpoint has invalid ml_history_ngrow for flux_mode=ml.");
        }
        for (int lev = 0; lev <= finest_level; ++lev) {
            if (lev < header_ml_hist_count.size()) {
                m_ml_hist_count[lev] = header_ml_hist_count[lev];
            }
        }
    } else if (m_flux_mode == FluxMode::ml && !has_ml_header) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            m_ml_hist_count[lev] = 0;
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) {

        // read in level 'lev' BoxArray from Header
        BoxArray ba;
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        DistributionMapping dm { ba, ParallelDescriptor::NProcs() };

        // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);

        // build MultiFab and FluxRegister data
        int ncomp = 1;
        // Check if this is an ensemble run
        int ensemble_run = 0;
        for (int edir : m_ensemble_dir) {
            ensemble_run += edir;
        }
        if (ensemble_run) { ncomp = 2;}

        int ng = 1;
        phi_old[lev].define(grids[lev], dmap[lev], ncomp, ng);
        phi_new[lev].define(grids[lev], dmap[lev], ncomp, ng);

        if (lev > 0 && do_reflux) {
            flux_reg[lev] = std::make_unique<FluxRegister>(grids[lev], dmap[lev], refRatio(lev-1), lev, ncomp);
        }

        if (m_flux_mode == FluxMode::ml && m_ml_history_len > 0) {
            m_phi_hist[lev] = std::make_unique<MultiFab>(grids[lev], dmap[lev], m_ml_history_len, m_ml_history_ngrow);
            m_phi_hist[lev]->setVal(amrex::Real(0.0));
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                BoxArray fba = grids[lev];
                fba.surroundingNodes(d);
                m_flux_hist[lev][d] = std::make_unique<MultiFab>(fba, dmap[lev], m_ml_history_len, m_ml_history_ngrow);
                m_flux_hist[lev][d]->setVal(amrex::Real(0.0));
            }
        }
    }

    // read in the MultiFab data
    for (int lev = 0; lev <= finest_level; ++lev) {
        VisMF::Read(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "phi"));
        if (m_flux_mode == FluxMode::ml && m_ml_history_len > 0 && has_ml_header) {
            VisMF::Read(*m_phi_hist[lev],
                        amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "phi_hist"));
            const char* flux_names[] = {"flux_hist_x", "flux_hist_y", "flux_hist_z"};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                VisMF::Read(*m_flux_hist[lev][d],
                            amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", flux_names[d]));
            }
        }
    }

#ifdef AMREX_PARTICLES
    particleData.Restart((amrex::ParGDBBase*)GetParGDB(),restart_chkfile);
#endif


}
