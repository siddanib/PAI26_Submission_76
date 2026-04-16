#ifndef EXTERNAL_POTENTIAL_H_
#define EXTERNAL_POTENTIAL_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <string>

enum class ExternalPotentialType {
    none = 0,
    quartic_2d
};

using EnsembleDirection = amrex::GpuArray<int, AMREX_SPACEDIM>;

struct ExternalPotential {
    bool enabled = false;
    ExternalPotentialType type = ExternalPotentialType::none;
    amrex::Real alpha = amrex::Real(0.0);
    amrex::Real beta = amrex::Real(0.0);
    amrex::Real gamma = amrex::Real(0.0);
};

inline std::string ExternalPotentialTypeToString (ExternalPotentialType type)
{
    switch (type) {
    case ExternalPotentialType::none:
        return "none";
    case ExternalPotentialType::quartic_2d:
        return "quartic_2d";
    }

    return "unknown";
}

inline ExternalPotentialType ExternalPotentialTypeFromString (std::string const& type_name)
{
    if (type_name == "quartic_2d") {
        return ExternalPotentialType::quartic_2d;
    }
    if (type_name == "none") {
        return ExternalPotentialType::none;
    }

    return static_cast<ExternalPotentialType>(-1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real external_potential_value (ExternalPotential const& pot,
                                      EnsembleDirection const& ensemble_dir,
                                      amrex::Real time,
                                      amrex::Real x,
                                      amrex::Real y,
                                      amrex::Real z = amrex::Real(0.0)) noexcept
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(z);

    if (!pot.enabled || pot.type == ExternalPotentialType::none) {
        return amrex::Real(0.0);
    }

    if (pot.type == ExternalPotentialType::quartic_2d) {
        amrex::Real xshift_a = x - pot.alpha;
        amrex::Real xshift_b = x - pot.beta;
        amrex::Real yshift = y - amrex::Real(0.5);
        amrex::Real potential = amrex::Real(0.);
        if (ensemble_dir[0] == 0) {
            potential += xshift_a * xshift_a * xshift_b * xshift_b;
        }
        if (ensemble_dir[1] == 0) {
            potential += yshift * yshift * yshift * yshift;
        }

        return potential/pot.gamma;
    }

    return amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real external_potential_gradient_component (ExternalPotential const& pot,
                                                   EnsembleDirection const& ensemble_dir,
                                                   int dir,
                                                   amrex::Real time,
                                                   amrex::Real x,
                                                   amrex::Real y,
                                                   amrex::Real z = amrex::Real(0.0)) noexcept
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(z);

    if (!pot.enabled || pot.type == ExternalPotentialType::none) {
        return amrex::Real(0.0);
    }
    if (ensemble_dir[dir] != 0) {
        return amrex::Real(0.0);
    }

    if (pot.type == ExternalPotentialType::quartic_2d) {
        if (dir == 0) {
            return amrex::Real(2.0) * (x - pot.beta) * (x - pot.alpha)
                 * (amrex::Real(2.0) * x - pot.alpha - pot.beta) / pot.gamma;
        }
        if (dir == 1) {
            amrex::Real centered = y - amrex::Real(0.5);
            return amrex::Real(4.0) * centered * centered * centered / pot.gamma;
        }
    }

    return amrex::Real(0.0);
}

#endif
