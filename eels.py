import argparse
import numpy as np
from geometries import SlottedTriangleLattice, SlottedTriangleLatticeCavity
from helper_functions import electron_beta, meep_freq_to_eV, eV_to_meep_freq
from electron_source import make_electron_path, pulse_chain, moving_gaussian
from post_processing import temporal_ft, temporal_ft_pade, temporal_ft_pade_scipy, assemble_gamma, gaussian_convolve

try:
    import meep as mp
except ImportError:
    mp = None  # allows static import / unit conversions without MEEP installed


# physical constants
_C = 299792458.0          # m/s
_H_EV = 4.135667696e-15     # eV*s


# Build geometry + cell 

def build_phc(args):
    a_nm = args.a
    a = 1.0
    h = np.sqrt(3) * a
    thickness = args.d / a_nm
    r = args.r
    shift = (args.W - 1) / 2 * h
    sw = args.s / a_nm
    crystal_x_width = args.x

    if args.cavity:
        dom = SlottedTriangleLatticeCavity(r, a, thickness, shift, sw,
                                           index=args.n, width=crystal_x_width)
    else:
        dom = SlottedTriangleLattice(r, a, thickness, shift, sw,
                                     index=args.n, width=crystal_x_width)
    return dom.geometry, dom.cell, thickness, a_nm


def make_cell(struct_cell, thickness, dpml, pad_xy, pad_z):
    """Full simulation cell = structure + air padding (y,z) + PML everywhere.
    x keeps the structure length plus PML so the electron flies straight
    through; no extra air pad on x is needed (the path lives inside the cell).
    """
    return mp.Vector3(
        struct_cell.x + 2 * dpml,                 # x: structure + PML on both ends
        struct_cell.y + 2 * (pad_xy + dpml),      # y: + air pad + PML
        thickness + 2 * (pad_z + dpml),           # z: slab + air pad + PML
    )


# One FDTD run: 

def run_path_recording(geometry, cell, beta, y0, z0, fcen, df, resolution,
                       dpml, T_extra, label="", source_model="pulse-chain"):
    
    pml = [mp.PML(dpml)]

    # electron path: enter just inside one PML, exit just inside the other
    x_start = -cell.x / 2.0 + dpml
    x_end = cell.x / 2.0 - dpml
    path_len = x_end - x_start
    transit = path_len / beta
    amp0 = 1.0 / beta             # ~ e/v current weight (abs cal still pending)

    electron_path = make_electron_path(x_start, beta, y0, z0)

    static_sources = []
    step_callback = None

    if source_model == "pulse-chain":
        static_sources = pulse_chain(path_len, resolution, x_start, x_end,
                                    beta, fcen, df, amp0, y0, z0)

    elif source_model == "moving-gaussian":
        step_callback = moving_gaussian(electron_path, amp0)

    else:
        raise ValueError(f"unknown source_model: {source_model} "
                        "(use 'pulse-chain' or 'moving-gaussian')")

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=static_sources,        # empty for moving models
        boundary_layers=pml,
        resolution=resolution,
        force_complex_fields=True,
    )

    # path pixels to sample
    total_time = transit + T_extra

    rec_times = []
    rec_fields = []
    line_vol = mp.Volume(center=mp.Vector3(0.5 * (x_start + x_end), y0, z0),
                         size=mp.Vector3(path_len, 0, 0))

    def record(sim_obj):
        ex_line = sim_obj.get_array(mp.Ex, vol=line_vol)
        rec_times.append(sim_obj.meep_time())
        rec_fields.append(np.asarray(ex_line, dtype=complex))

    run_args = [mp.at_every(sim.Courant / resolution, record)]
    if step_callback is not None:
        run_args = [step_callback] + run_args
    sim.run(*run_args, until=total_time)

    rec_fields = np.array(rec_fields)
    nx_actual = rec_fields.shape[1]
    xs = np.linspace(x_start, x_end, nx_actual)
    return np.array(rec_times), xs, rec_fields



# Driver

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-a", type=int, default=426, help="lattice constant a [nm]")
    p.add_argument("-d", type=int, default=220, help="slab thickness [nm]")
    p.add_argument("-x", type=int, default=36, help="crystal width [units of a]")
    p.add_argument("-W", type=float, default=1.2, help="waveguide widening factor")
    p.add_argument("-s", type=int, default=100, help="slot width [nm]")
    p.add_argument("-r", type=float, default=0.245, help="hole radius [units of a]")
    p.add_argument("-n", type=float, default=3.45, help="refractive index")
    p.add_argument("-v", type=float, default=100.0, help="electron energy [keV]")
    
    p.add_argument("--cavity", action="store_true", help="use cavity geometry")
    p.add_argument("--Emin", type=float, default=0.40)
    p.add_argument("--Emax", type=float, default=1.00)
    p.add_argument("--nE", type=int, default=600)
    p.add_argument("--Textra", type=float, default=400.0)
    p.add_argument("--out", type=str, default="eels_spectrum.npz")
    args = p.parse_args()

    resolution = args.a / 18.0
    beta = electron_beta(args.v)

    geometry, struct_cell, thickness, a_nm = build_phc(args)
    dpml = 1.0
    cell = make_cell(struct_cell, thickness, dpml, 1.0, 1.5 * thickness)
    y0, z0 = 0.0, 0.0

    # cavity run, then vacuum run, then subtract
    t, xs, E_c = run_path_recording(geometry, cell, beta, y0, z0,
                                    resolution, dpml, args.Textra)
    _, _, E_v = run_path_recording(None, cell, beta, y0, z0,
                                   resolution, dpml, args.Textra)
    E_ind = E_c - E_v

    # temporal FT + eq.1 assembly + broadening
    E_eV = np.linspace(max(args.Emin, 1e-3), args.Emax, args.nE)
    omegas = 2 * np.pi * eV_to_meep_freq(E_eV, args.a)
    E_xw = temporal_ft(t, E_ind, omegas)
    gamma = assemble_gamma(E_xw, xs, omegas, beta, a_nm)
    gamma_conv = gaussian_convolve(E_eV, gamma, fwhm_eV=0.030)

    np.savez(args.out, E_eV=E_eV, gamma=gamma, gamma_conv=gamma_conv)
    print(f"[done] saved {args.out}")


if __name__ == "__main__":
    main()