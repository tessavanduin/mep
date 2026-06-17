import numpy as np

# physical constants
_C = 299792458.0          # m/s
_H_EV = 4.135667696e-15     # eV*s
_ME_C2 = 510998.95          # electron rest energy, eV


def electron_beta(E_kin_keV):
    """Relativistic beta = v/c for a given electron kinetic energy in keV."""
    E_kin = E_kin_keV * 1e3  # eV
    gamma = 1.0 + E_kin / _ME_C2
    return np.sqrt(1.0 - 1.0 / gamma**2)


def meep_freq_to_eV(f_meep, a_nm):
    """MEEP frequency (units c/a) -> photon energy in eV. a_nm in nm."""
    a_m = a_nm * 1e-9
    lam_m = a_m / np.asarray(f_meep)      # wavelength in metres
    return _H_EV * _C / lam_m


def eV_to_meep_freq(E_eV, a_nm):
    a_m = a_nm * 1e-9
    lam_m = _H_EV * _C / np.asarray(E_eV)
    return a_m / lam_m