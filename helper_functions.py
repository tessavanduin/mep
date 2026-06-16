"""
helper_functions.py  --  units, electron-source amplitude, and diagnostics
==========================================================================

This module collects the physical-unit bookkeeping for the EELS-in-MEEP
calculation.  The single most important thing to get right in this whole
project is the *consistent* set of units, because the paper reports
**absolute** probabilities with no fit parameters.

MEEP natural units (used throughout the FDTD run)
-------------------------------------------------
MEEP solves Maxwell's equations with  c = eps0 = mu0 = 1  and all lengths
expressed in units of a chosen length scale `a` (here a = 426 nm).
Consequences we rely on below:
    * velocity in MEEP units  ==  v / c   ==  beta
    * a MEEP frequency f corresponds to SI angular frequency  omega = 2*pi*f*c/a
    * a MEEP time t corresponds to SI time  t * a / c
    * a MEEP length x corresponds to SI length  x * a

Conventions for the Fourier transform (must match the post-processing!)
-----------------------------------------------------------------------
We use the SAME convention as Eq. (1) of the paper:
    E_hat(omega) = \int E(t) e^{-i omega t} dt        (forward transform)
    Gamma(omega) = (e v / pi hbar omega) Re \int E_hat(v t) e^{+i omega t} dt
"""

import numpy as np
import meep as mp

# ---------------------------------------------------------------- constants (SI)
c          = 299792458.0          # speed of light            [m/s]
h_bar      = 1.054571817e-34      # reduced Planck constant   [J s]
q_e        = 1.602176634e-19      # elementary charge         [C]
m_e        = 9.1093837139e-31     # electron mass             [kg]
epsilon_0  = 8.8541878128e-12     # vacuum permittivity       [F/m]
alpha_fs   = q_e**2 / (4*np.pi*epsilon_0*h_bar*c)   # fine-structure constant ~1/137


# --------------------------------------------------------------- unit conversions
def E_to_freq_meep(E_eV: float, a: float) -> float:
    """Photon energy (eV) -> MEEP frequency f (units of c/a).

    omega = E/hbar,  f_meep = omega a / (2 pi c) = E q_e a / (h_bar 2 pi c).
    (This is what the old `E_to_omega` actually returned -- it was a MEEP
    *frequency*, not an angular frequency, so it is renamed here for clarity.)
    """
    return E_eV * q_e * a / (h_bar * 2 * np.pi * c)


def E_to_speed(E_kin_eV: float) -> float:
    """Electron kinetic energy (eV) -> relativistic beta = v/c (MEEP velocity)."""
    gamma = 1.0 + E_kin_eV * q_e / (m_e * c**2)
    return np.sqrt(1.0 - 1.0 / gamma**2)


# ------------------------------------------------- electron current source (MEEP)
#
# The electron is a moving point charge q = -e.  In the FDTD it is injected as a
# J_x current source that is relocated to the electron's voxel every time step
# (see EELS_3D.py).  While the source sits in a given voxel the charge has to
# transit that voxel, so the *constant* current density it must carry is fixed by
# charge conservation:
#
#     charge through one voxel  =  j_x * A_transverse * tau   =  q
#     A_transverse = (1/resolution)^2 ,   tau = dx / v = 1/(resolution * beta)
# =>  j_x = q * resolution^3 * beta            (all in MEEP units)
#
# `Q_E_MEEP` is the electron charge expressed in MEEP units.  We simply set it to
# 1 and absorb the SI value of the charge into the final conversion constant
# `gamma_si_prefactor()` below.  The vacuum flux-box check (see
# `enclosed_charge`) MUST return Q_E_MEEP for the normalisation to be exact; if
# it returns q_eff != Q_E_MEEP, divide the induced field by q_eff once.
Q_E_MEEP = 1.0


def electron_source_amplitude(resolution: float, beta: float) -> float:
    """MEEP amplitude (current density J_x) for a single electron of charge -e.

    NOTE: this is the piece the student had commented out.  Leaving it at the
    default of 1.0 is what destroyed the absolute normalisation.
    """
    return -Q_E_MEEP * resolution**3 * beta


def gamma_si_prefactor(omega_si: np.ndarray) -> np.ndarray:
    """Overall SI prefactor that turns the (dimensionless) MEEP trajectory sum
    into the loss probability per unit *angular* frequency [s].

    Derivation (see README_EELS_physics.md):

        E_SI       = (e / (eps0 a^2)) * E_meep         (Coulomb-law field unit)
        E_hat_SI   = (a/c) * E_SI_units * E_hat_meep   (extra a/c from dt)
        dx_SI      = a * dx_meep
        Gamma(w)   = (e / (pi hbar w_SI)) Re sum_j E_hat_SI(x_j) e^{i w t_e} dx_SI

    Collecting factors the length scale `a` cancels completely and one is left
    with the remarkably clean result

        Gamma(omega) = (4 * alpha / omega_SI) * Re[ sum_j E_hat_meep * phase * dx_meep ]

    i.e. EELS probabilities scale with the fine-structure constant, as they must.
    The post-processing supplies the bracketed (dimensionless) MEEP sum; this
    function supplies 4*alpha/omega.
    """
    out = np.zeros_like(np.asarray(omega_si, dtype=float))
    nz = np.asarray(omega_si) != 0
    out[nz] = 4.0 * alpha_fs / np.asarray(omega_si)[nz]
    return out


# ---------------------------------------------------------- VACUUM-ONLY diagnostic
#
# A flux box computes the closed-surface integral of the field.  By Gauss's law
#       \oint D . dA = Q_free   (free charge enclosed).
# That makes it a perfectly good *charge meter* and nothing more.  It is NOT the
# EELS observable and it must NEVER appear in the normalisation of the induced
# field (the student divided E by it, which corrupts the empty-subtraction and,
# in a dielectric, also picks up bound charge).
#
# Use it ONCE, in vacuum, to verify the source really injects one electron.
# We integrate D (mp.Dx/Dy/Dz); in vacuum D = eps0 E so in MEEP units D == E,
# but using D makes the "free charge" interpretation correct also if you ever
# run the check with material present.

def create_flux_box(cent: mp.Vector3, b: mp.Vector3):
    """Return the six faces of a box centred at `cent` with side lengths `b`,
    together with the D-component normal to each face and its outward sign."""
    g = b / 2
    surfaces = [
        mp.Volume(center=cent + mp.Vector3( g.x, 0, 0), size=b - mp.Vector3(b.x, 0, 0)),
        mp.Volume(center=cent + mp.Vector3(-g.x, 0, 0), size=b - mp.Vector3(b.x, 0, 0)),
        mp.Volume(center=cent + mp.Vector3(0,  g.y, 0), size=b - mp.Vector3(0, b.y, 0)),
        mp.Volume(center=cent + mp.Vector3(0, -g.y, 0), size=b - mp.Vector3(0, b.y, 0)),
        mp.Volume(center=cent + mp.Vector3(0, 0,  g.z), size=b - mp.Vector3(0, 0, b.z)),
        mp.Volume(center=cent + mp.Vector3(0, 0, -g.z), size=b - mp.Vector3(0, 0, b.z)),
    ]
    components = [mp.Dx, mp.Dx, mp.Dy, mp.Dy, mp.Dz, mp.Dz]
    signs      = [1, -1, 1, -1, 1, -1]
    return surfaces, components, signs


def enclosed_charge(sim: mp.Simulation, flux_box: tuple, ds: float) -> float:
    """Closed-surface integral  \oint D . dA  = enclosed free charge (MEEP units).

    Intended use: a single vacuum run, to confirm the moving source injects
    Q_E_MEEP per electron.  `ds` is the MEEP surface element (a/resolution)**2.
    """
    flux = 0.0
    for surf, comp, sign in zip(*flux_box):
        field = sim.get_array(comp, vol=surf)
        flux += sign * np.sum(field)
    return flux * ds