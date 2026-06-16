"""
eels_postprocess.py  --  turn recorded FDTD fields into an EELS spectrum.

Pipeline (matches Eq. (1) of Bezard et al., ACS Nano 2024):

    1. induced field   E_ind(x,t) = E_crystal(x,t) - E_empty(x,t)
       (identical source normalisation in both runs -> the bare electron field
        cancels exactly.  NO division by any flux box.)
    2. temporal FT     E_hat(x,omega) = \int E_ind(x,t) e^{-i omega t} dt
    3. trajectory sum  Gamma(omega) = (4 alpha / omega_SI)
                                       * Re sum_x E_hat(x,omega) e^{i omega t_e(x)} dx
       with t_e(x) = (x - x_start)/v  the time the electron is at pixel x.
    4. Gaussian broadening to the ~30 meV experimental resolution.

The factor (4 alpha / omega_SI) is the complete, derived SI normalisation
(see helper_functions.gamma_si_prefactor and README_EELS_physics.md); all the
ad-hoc factors of the first version (conversion_factor, /h_bar/1000*dt,
dividing by 'flux') are gone.
"""

import glob
import numpy as np
import h5py

from helper_functions import (
    c, q_e, h_bar, alpha_fs,
    E_to_freq_meep, gamma_si_prefactor,
)

try:
    from scipy.interpolate import pade as _scipy_pade
except Exception:                                   # pragma: no cover
    _scipy_pade = None


# --------------------------------------------------------------------- loading
def load_run(path):
    """Load one EELS_3D-*.h5 file -> (E_field[x, t], attrs dict)."""
    with h5py.File(path, "r") as f:
        # MEEP appends the time axis as the last dimension of 'ex'.
        ex = f["ex"][()]
        attrs = dict(f.attrs)
        attrs["x_pix_meep"] = f["x_pix_meep"][()]
    if ex.ndim > 2:
        ex = ex.reshape(-1, ex.shape[-1])           # (n_pixels, n_time)
    return ex, attrs


def induced_field(crystal_path, empty_path):
    """E_crystal - E_empty.  Both must share the same (x, t) sampling."""
    Ec, attrs = load_run(crystal_path)
    Ee, _      = load_run(empty_path)
    n = min(Ec.shape[1], Ee.shape[1])               # guard against off-by-one steps
    return Ec[:, :n] - Ee[:, :n], attrs


# --------------------------------------------------- temporal Fourier transform
def temporal_FT(E_ind, attrs, energies_eV, method="dft"):
    """E_ind[x, t] -> E_hat[x, omega] in MEEP units (continuous-FT normalised).

    `energies_eV` is the requested energy-loss grid (eV).
    method = 'dft'  -> plain discrete FT  (robust; use this to validate)
             'pade' -> Pade approximant of the DTFT (sharper lines from a
                       truncated/ring-down-limited series).
    """
    a_si      = attrs["a_nm"] * 1e-9
    dt_meep   = float(attrs["dt_meep"])
    n_t       = E_ind.shape[1]
    t_meep    = np.arange(n_t) * dt_meep            # absolute MEEP time, t=0 at sim start

    # phase per time sample: omega_meep * dt_meep = 2 pi f_meep dt_meep
    f_meep    = E_to_freq_meep(np.asarray(energies_eV, float), a_si)   # MEEP freq
    omega_dt  = 2 * np.pi * f_meep * dt_meep                            # per sample

    E_hat = np.empty((E_ind.shape[0], len(energies_eV)), dtype=np.complex128)

    if method == "dft":
        # E_hat(omega) = sum_n E_n e^{-i omega t_n} dt   (continuous-FT normalised)
        phase = np.exp(-1j * np.outer(t_meep, 2 * np.pi * f_meep))      # (n_t, n_w)
        E_hat[:] = (E_ind @ phase) * dt_meep
    elif method == "pade":
        if _scipy_pade is None:
            raise RuntimeError("scipy is required for the Pade method.")
        z = np.exp(-1j * omega_dt)
        order = (n_t - 1) // 2
        for i, series in enumerate(E_ind):
            P, Q = _scipy_pade(series.astype(np.complex128), order)
            E_hat[i] = (P(z) / Q(z)) * dt_meep
    else:
        raise ValueError("method must be 'dft' or 'pade'")
    return E_hat


# ------------------------------------------------------------ trajectory sum
def compute_gamma(E_hat, attrs, energies_eV):
    """Project E_hat(x, omega) onto the electron and apply the SI prefactor.

    Returns Gamma in probability per unit *energy* [1/eV], matching the paper's
    "% per eV" axis after multiplying by 100.
    """
    x_meep   = attrs["x_pix_meep"]                  # MEEP length
    dx_meep  = float(np.mean(np.diff(x_meep)))
    beta     = float(attrs["beta"])                 # v/c
    start    = float(attrs["start_pos"])            # electron x at t=0 (MEEP)
    a_si     = attrs["a_nm"] * 1e-9

    energies_eV = np.asarray(energies_eV, float)
    omega_si = energies_eV * q_e / h_bar            # rad/s
    f_meep   = E_to_freq_meep(energies_eV, a_si)
    omega_meep = 2 * np.pi * f_meep                 # MEEP angular freq

    # electron arrival time at each pixel (MEEP time) -> projection phase
    t_e = (x_meep - start) / beta                   # (n_x,)
    # phase[x, w] = exp(+i omega_meep * t_e(x))
    phase = np.exp(1j * np.outer(t_e, omega_meep))  # (n_x, n_w)

    # dimensionless MEEP trajectory sum
    S = np.real(np.sum(E_hat * phase, axis=0) * dx_meep)        # (n_w,)

    # Gamma per angular frequency [s] = (4 alpha / omega_SI) * S
    gamma_omega = gamma_si_prefactor(omega_si) * S
    # convert to per unit energy: dE = hbar d(omega)  -> Gamma_E = Gamma_omega / hbar
    gamma_E = gamma_omega / h_bar * q_e             # [1/eV]  (q_e converts 1/J -> 1/eV)
    return gamma_E


# ------------------------------------------------------------ broadening
def gaussian_broaden(energies_eV, gamma, fwhm_eV=30e-3):
    """Convolve with the experimental Gaussian (default 30 meV FWHM)."""
    de = np.mean(np.diff(energies_eV))
    sigma = fwhm_eV / (2 * np.sqrt(2 * np.log(2)))
    n = int(np.ceil(4 * sigma / de))
    k = np.arange(-n, n + 1) * de
    g = np.exp(-k**2 / (2 * sigma**2))
    g /= g.sum()
    return np.convolve(gamma, g, mode="same")


# ------------------------------------------------------------ convenience
def spectrum(crystal_path, empty_path, energies_eV=None,
             method="dft", fwhm_eV=30e-3, broaden=True):
    """Full chain: paths -> (energies, Gamma_percent_per_eV)."""
    if energies_eV is None:
        energies_eV = np.linspace(0.0, 2.1, 1500)
    E_ind, attrs = induced_field(crystal_path, empty_path)
    E_hat = temporal_FT(E_ind, attrs, energies_eV, method=method)
    gamma = compute_gamma(E_hat, attrs, energies_eV)
    if broaden:
        gamma = gaussian_broaden(energies_eV, gamma, fwhm_eV)
    return energies_eV, gamma * 100.0               # -> % per eV


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    empty = sorted(glob.glob("EELS_3D-out/EELS_3D-EMPTY*.h5"))[0]
    for crystal in sorted(glob.glob("EELS_3D-out/EELS_3D-CRYSTAL*.h5")):
        E, G = spectrum(crystal, empty, method="dft")
        plt.figure()
        plt.plot(E, G)
        plt.axhline(0, ls="--", lw=0.5, color="grey")
        plt.xlabel("Energy loss (eV)")
        plt.ylabel("Probability (% / eV)")
        plt.title(crystal.split("/")[-1])
    plt.show()