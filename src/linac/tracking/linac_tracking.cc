/////////////////////////////////////////////////////////////////////////////
//
// FILE NAME
//   linac_tracking.cc
//
// AUTHOR
//   Andrei Shishlo, ORNL, shishlo@ornl.gov
//   2017.01.24
//
// DESCRIPTION
//   Define elementary functions for tracking through
//   specific linac accelerator elements without using
//   TEAPOT algorithms. It is necessary for the cases
//   with a huge energy spread in the bunch.
//
/////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
//
// Include files
//
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
//
// Local Functions:
//
///////////////////////////////////////////////////////////////////////////

#include "linac/tracking/linac_tracking.hh"

#include <cmath>

#include "orbit/Bunch.hh"
#include "orbit/OrbitConst.hh"
#include "orbit/SyncPart.hh"

namespace linac_tracking
{

namespace
{

double momentum_squared(double kinetic_energy, double mass)
{
    return kinetic_energy * (kinetic_energy + 2.0 * mass);
}

double longitudinal_momentum_squared(double p2, double p_s2, double transverse_momentum2)
{
    // std::fma retains precision near cancellation in Pz^2 = P^2 - Px^2 - Py^2.
    return std::fma(-p_s2, transverse_momentum2, p2);
}

struct LongitudinalUpdate
{
    LongitudinalUpdate(double length, double mass, double reference_total_energy,
                       double reference_momentum)
        : length(length), reference_total_energy(reference_total_energy),
          reference_momentum(reference_momentum), mass_squared(mass * mass),
          reference_energy_squared_momentum_squared(reference_total_energy *
                                                    reference_total_energy *
                                                    (reference_momentum * reference_momentum))
    {
    }

    double delta_z(double energy_offset, double total_energy, double longitudinal_momentum,
                   double normalized_transverse_momentum_squared) const
    {
        // Rationalize 1 - beta_s/beta_z to preserve near-reference precision.
        // Numerator: m^2*dE*(2E_s + dE) - E_s^2 * p_s^2 * (xp^2 + yp^2).
        const double numerator = std::fma(
            -reference_energy_squared_momentum_squared, normalized_transverse_momentum_squared,
            mass_squared * energy_offset * (2.0 * reference_total_energy + energy_offset));
        const double reference_energy_times_pz = reference_total_energy * longitudinal_momentum;
        return length * numerator /
               ((reference_energy_times_pz + reference_momentum * total_energy) *
                reference_energy_times_pz);
    }

    const double length;
    const double reference_total_energy;
    const double reference_momentum;
    const double mass_squared;
    const double reference_energy_squared_momentum_squared;
};

struct PlaneMatrix
{
    double m11;
    double m12;
    double m21;
};

} // namespace

///////////////////////////////////////////////////////////////////////////
// NAME
//   linac_drift
//
// DESCRIPTION
//   Drifts a particle bunch to a fixed longitudinal plane.
//
// PARAMETERS
//   bunch = reference to the macro-particle bunch
//   length = length of the drift
//
// RETURNS
//   Nothing
//
///////////////////////////////////////////////////////////////////////////
///

void linac_drift(Bunch* bunch, double length)
{
    if (length == 0.)
    {
        return;
    }

    SyncPart* syncPart = bunch->getSyncPart();

    const double beta_s = syncPart->getBeta();
    const double mass = syncPart->getMass();
    const double e_kin_s = syncPart->getEnergy();
    const double e_total_s = e_kin_s + mass;
    const double p_s = syncPart->getMomentum();
    const double p_s2 = p_s * p_s;
    const LongitudinalUpdate longitudinal_update(length, mass, e_total_s, p_s);
    const double length_p_s = length * p_s;

    syncPart->setTime(syncPart->getTime() + length / (OrbitConst::c * beta_s));

    double** arr = bunch->coordArr();
    const int bunch_size = bunch->getSize();

    int invalid_state = 0;
#pragma omp parallel for reduction(| : invalid_state)
    for (int i = 0; i < bunch_size; ++i)
    {
        const double xp = arr[i][1]; // x' := P_x / P_s
        const double yp = arr[i][3]; // y' := P_y / P_s
        const double dE = arr[i][5];
        const double e_kin = e_kin_s + dE;
        if (e_kin <= 0.)
        {
            invalid_state |= 1;
            continue;
        }

        const double e_total = e_kin + mass;
        const double p2 = momentum_squared(e_kin, mass);
        const double transverse_momentum2 = xp * xp + yp * yp;

        const double pz2 = longitudinal_momentum_squared(p2, p_s2, transverse_momentum2);
        if (pz2 <= 0.)
        {
            invalid_state |= 2;
            continue;
        }

        const double pz = std::sqrt(pz2);
        const double transverse_scale = length_p_s / pz;
        arr[i][0] += xp * transverse_scale;
        arr[i][2] += yp * transverse_scale;

        arr[i][4] += longitudinal_update.delta_z(dE, e_total, pz, transverse_momentum2);
    }

    if (invalid_state & 1)
    {
        ORBIT_MPI_Abort("linac_drift - particle has non-positive kinetic energy.");
    }
    if (invalid_state & 2)
    {
        ORBIT_MPI_Abort("linac_drift - particle cannot reach the forward longitudinal plane.");
    }
}

////////////////////////////
// NAME
//   linac_quad1
//
// DESCRIPTION
//   Quadrupole element one: linear transport matrix for linac
//
// PARAMETERS
//   bunch  = reference to the macro-particle bunch
//   length = length of transport
//   kq = quadrupole field strength [m^(-2)]
//   kq = G/(B*rho) B*rho = 3.33564*P0/charge [T*m] where P0 in GeV/c
//   G = dB/dr [T/m] quad gradient
//
// RETURNS
//   Nothing
//
///////////////////////////////////////////////////////////////////////////

void linac_quad1(Bunch* bunch, double length, double kq, [[maybe_unused]] int useCharge)
{
    if (length == 0.)
    {
        return;
    }

    if (kq == 0. || bunch->getCharge() == 0.)
    {
        linac_drift(bunch, length);
        return;
    }

    SyncPart* syncPart = bunch->getSyncPart();
    const double beta_s = syncPart->getBeta();
    const double mass = syncPart->getMass();
    const double Ekin_s = syncPart->getEnergy();
    const double Etotal_s = Ekin_s + mass;
    const double p_s = syncPart->getMomentum();
    const double p_s2 = p_s * p_s;
    const LongitudinalUpdate longitudinal_update(length, mass, Etotal_s, p_s);

    syncPart->setTime(syncPart->getTime() + length / (OrbitConst::c * beta_s));

    //coordinate array [part. index][x,xp,y,yp,z,dE]
    double** arr = bunch->coordArr();
    const int nParts = bunch->getSize();
    const bool x_focusing = kq > 0.0;
    const double abs_kq = std::fabs(kq);

    int invalid_state = 0;
#pragma omp parallel for reduction(| : invalid_state)
    for (int i = 0; i < nParts; ++i)
    {

        const double dE = arr[i][5];
        const double Ekin = Ekin_s + dE;
        if (Ekin <= 0.)
        {
            invalid_state |= 1;
            continue;
        }

        const double Etotal = Ekin + mass;
        const double p2 = momentum_squared(Ekin, mass);
        const double transverse_momentum2 = arr[i][1] * arr[i][1] + arr[i][3] * arr[i][3];
        const double pz2 = longitudinal_momentum_squared(p2, p_s2, transverse_momentum2);
        if (pz2 <= 0.)
        {
            invalid_state |= 2;
            continue;
        }

        const double pz = std::sqrt(pz2);
        const double coeff = p_s / pz;

        const double sqrt_kq = std::sqrt(abs_kq * coeff);
        const double kqlength = sqrt_kq * length;

        const double c = std::cos(kqlength);
        const double s = std::sin(kqlength);
        const double ch = std::cosh(kqlength);
        const double sh = std::sinh(kqlength);
        const PlaneMatrix focusing{c, s / sqrt_kq, -s * sqrt_kq};
        const PlaneMatrix defocusing{ch, sh / sqrt_kq, sh * sqrt_kq};
        const PlaneMatrix& mx = x_focusing ? focusing : defocusing;
        const PlaneMatrix& my = x_focusing ? defocusing : focusing;

        const double x_init = arr[i][0];
        const double xp_init = arr[i][1] * coeff;
        const double y_init = arr[i][2];
        const double yp_init = arr[i][3] * coeff;

        arr[i][0] = x_init * mx.m11 + xp_init * mx.m12;
        arr[i][1] = (x_init * mx.m21 + xp_init * mx.m11) / coeff;
        arr[i][2] = y_init * my.m11 + yp_init * my.m12;
        arr[i][3] = (y_init * my.m21 + yp_init * my.m11) / coeff;

        arr[i][4] += longitudinal_update.delta_z(dE, Etotal, pz, transverse_momentum2);
    }

    if (invalid_state & 1)
    {
        ORBIT_MPI_Abort("linac_quad1 - particle has non-positive kinetic energy.");
    }
    if (invalid_state & 2)
    {
        ORBIT_MPI_Abort("linac_quad1 - particle cannot reach the forward longitudinal plane.");
    }
}

///////////////////////////////////////////////////////////////////////////
// NAME
//  linac_ quad2
//
// DESCRIPTION
//   Quadrupole element two: nonlinear piece for linac
//
// PARAMETERS
//   bunch  = reference to the macro-particle bunch
//   length = length of the element
//
// RETURNS
//   Nothing
//
///////////////////////////////////////////////////////////////////////////

void linac_quad2(Bunch* bunch, double length)
{
    //there are no non-linear components
}

////////////////////////////
// NAME
//   linac_quad3
//
// DESCRIPTION
//   Quadrupole element 3: longitudinal field component of the quad field
//   Bz(x,y,z) = x*y*dG(z)/dz
//   This function performs the transverse kicks correction, so the length
//   is just a parameter.
//
// PARAMETERS
//   bunch  = reference to the macro-particle bunch
//   length = length of transport
//   G = dB/dr [T/m] quad gradient
//   0.299792458 = (1/v_light) * 10^9
//
// RETURNS
//   Nothing
//
///////////////////////////////////////////////////////////////////////////

void linac_quad3(Bunch* bunch, double length, double dB_dz)
{
    const double charge = bunch->getCharge();

    if (dB_dz == 0. || charge == 0.)
    {
        return;
    }

    SyncPart* syncPart = bunch->getSyncPart();
    const double kick_coeff = OrbitConst::c * charge * dB_dz * length / syncPart->getMomentum();

    //coordinate array [part. index][x,xp,y,yp,z,dE]
    double** arr = bunch->coordArr();
    const int nParts = bunch->getSize();

#pragma omp parallel for
    for (int i = 0; i < nParts; ++i)
    {
        const double coef_xy = kick_coeff * arr[i][0] * arr[i][2];
        arr[i][1] += arr[i][3] * coef_xy;
        arr[i][3] += -arr[i][1] * coef_xy;
    }
}

///////////////////////////////////////////////////////////////////////////
// NAME
//   kick
//
// DESCRIPTION
//   Kicks a particle bunch
//
// PARAMETERS
//   bunch = reference to the macro-particle bunch
//   kx = strength of the horizontal kick in rad
//   ky = strength of the vertical kick in rad
//   kE = strength of the energy kick in GeV
//
// RETURNS
//   Nothing
//
///////////////////////////////////////////////////////////////////////////

void kick(Bunch* bunch, double kx, double ky, double kE, int /*useCharge*/)
{
    SyncPart* syncPart = bunch->getSyncPart();

    const double mass = syncPart->getMass();
    const double Ekin_s = syncPart->getEnergy();
    const double p_s = syncPart->getMomentum();
    const double p2_s = p_s * p_s;

    //coordinate array [part. index][x,xp,y,yp,z,dE]
    double** arr = bunch->coordArr();
    const int bunch_size = bunch->getSize();

#pragma omp parallel for
    for (int i = 0; i < bunch_size; ++i)
    {
        const double dE = arr[i][5];
        const double Ekin = Ekin_s + dE;
        const double p2 = Ekin * (Ekin + 2.0 * mass);
        const double coeff = p_s / std::sqrt(longitudinal_momentum_squared(
                                       p2, p2_s, arr[i][1] * arr[i][1] + arr[i][3] * arr[i][3]));

        arr[i][1] += kx * coeff;
        arr[i][3] += ky * coeff;
        arr[i][5] += kE;
    }
}
} //end of namespace linac_tracking
