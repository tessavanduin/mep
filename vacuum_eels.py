import numpy as np
import meep as mp

from electron_source import make_electron_path, moving_gaussian
from post_processing import temporal_ft, assemble_gamma, gaussian_convolve
from helper_functions import electron_beta, eV_to_meep_freq


def run_path_recording(geometry, cell, beta, y0, z0, resolution, dpml, T_extra):
    x_start = -cell.x / 2.0 + dpml
    x_end = cell.x / 2.0 - dpml
    path_len = x_end - x_start
    transit = path_len / beta
    amp0 = 1.0 / beta

    electron_path = make_electron_path(x_start, beta, y0, z0)
    step_callback = moving_gaussian(electron_path, amp0)

    sim = mp.Simulation(
        cell_size=cell, geometry=geometry, sources=[],
        boundary_layers=[mp.PML(dpml)], resolution=resolution,
        force_complex_fields=True,
    )

    line_vol = mp.Volume(center=mp.Vector3(0.5 * (x_start + x_end), y0, z0),
                         size=mp.Vector3(path_len, 0, 0))
    rec_times, rec_fields = [], []

    def record(sim_obj):
        rec_times.append(sim_obj.meep_time())
        rec_fields.append(np.asarray(
            sim_obj.get_array(mp.Ex, vol=line_vol), dtype=complex))

    sim.run(step_callback, mp.at_every(sim.Courant / resolution, record),
            until=transit + T_extra)

    rec_fields = np.array(rec_fields)
    xs = np.linspace(x_start, x_end, rec_fields.shape[1])
    return np.array(rec_times), xs, rec_fields


if __name__ == "__main__":
    a_nm = 426
    beta = electron_beta(100.0)        # 100 keV
    cell = mp.Vector3(40, 8, 8)        # empty cell, in units of a
    dpml = 1.0
    y0 = z0 = 0.0
    resolution = a_nm / 18.0
    T_extra = 200.0

    # ONE vacuum run — geometry is None (empty)
    t, xs, E = run_path_recording(None, cell, beta, y0, z0,
                                  resolution, dpml, T_extra)
    print("field shape:", E.shape)   # (n_timesteps, n_pixels)

    # NO subtraction in vacuum: the recorded field IS the electron field
    E_ind = E

    # transform + eq.1 assembly + broadening
    E_eV = np.linspace(0.4, 1.0, 600)
    omegas = 2 * np.pi * eV_to_meep_freq(E_eV, a_nm)
    E_xw = temporal_ft(t, E_ind, omegas)
    gamma = assemble_gamma(E_xw, xs, omegas, beta, a_nm)
    gamma_conv = gaussian_convolve(E_eV, gamma, fwhm_eV=0.030)

    np.savez("eels_vacuum.npz", E_eV=E_eV, gamma=gamma, gamma_conv=gamma_conv)
    print("saved eels_vacuum.npz")