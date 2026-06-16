"""
EELS_3D.py  --  FDTD field generation for free-electron EELS of a slotted
photonic-crystal cavity (Bezard et al., ACS Nano 2024).

What this script produces
-------------------------
For one (y, z) impact parameter it records E_x(x, t) along the electron line
for the WHOLE simulation, including the cavity ring-down after the electron has
left.  Post-processing (eels_postprocess.py) subtracts an identical empty run,
Fourier-transforms in time, and evaluates the loss probability Gamma(omega).

Physics fixes relative to the first student version
---------------------------------------------------
1. The electron current source has its amplitude SET (it was commented out), so
   the absolute normalisation is meaningful.  By default it is a SMOOTHED
   finite-size Gaussian source (Meep issue #1118) to suppress the voxel-jump
   artefacts of a moving point charge; --source-sigma 0 recovers the bare point
   source.  Either way the source carries exactly one electron.
2. The field is recorded for the full run, NOT only while the electron is inside
   the crystal.  High-Q modes ring after the electron leaves and that ring-down
   carries the line shape; truncating it destroys the spectrum.
3. The flux/divergence boxes are removed from the data path; they survive only
   as an OPTIONAL one-off vacuum charge check (--charge-check).
4. Empty and crystal runs use the IDENTICAL source normalisation, so the bare
   electron field cancels exactly in the subtraction.
5. The recording geometry (pixel positions, dt, electron start) is written into
   the HDF5 file so the post-processing needs no guessing.
"""

import argparse
import numpy as np
import h5py
import meep as mp

from geometries import SlottedTriangleLattice, SlottedTriangleLatticeCavity
from helper_functions import (
    E_to_speed,
    electron_source_amplitude,
    gaussian_current_amp_func,
    create_flux_box,
    enclosed_charge,
    Q_E_MEEP,
)


def main(args):
    a_nm = args.a

    # ---- geometry (all lengths in units of the lattice constant a) ----------
    a         = 1.0
    h         = np.sqrt(3) * a
    thickness = args.d / a_nm
    r         = args.r
    shift     = (args.W - 1) / 2 * h
    sw        = args.s / a_nm
    crystal_x_width = args.x

    if args.cavity:
        print("Using geometry: SlottedTriangleLatticeCavity")
        domain = SlottedTriangleLatticeCavity(r, a, thickness, shift, sw,
                                              index=args.n, width=crystal_x_width)
    else:
        print("Using geometry: SlottedTriangleLattice")
        domain = SlottedTriangleLattice(r, a, thickness, shift, sw,
                                        index=args.n, width=crystal_x_width)
    geometry, cell = domain.geometry, domain.cell

    cell = cell + mp.Vector3(1, 1, 1) * 12 * thickness
    if args.test and not args.plot:
        cell = mp.Vector3(2, 2, 2)

    # 18 nm target resolution (paper value)
    resolution = np.ceil(a_nm / 18)
    print(f"RESOLUTION: {resolution} px/a  = {a_nm/resolution:.2f} nm")
    print(f"Accelerating voltage: {args.v} kV")

    dpml = thickness
    pml_layers = [mp.PML(thickness=dpml)]
    if args.empty:
        geometry = None

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        symmetries=None,
                        resolution=resolution)

    if args.plot:
        sim.plot3D() if sim.dimensions == 3 else sim.plot2D()
        return

    # ---- electron trajectory along x ---------------------------------------
    beta = E_to_speed(args.v * 1e3)                       # v/c (MEEP velocity)
    electron_path_length = cell.x - 2 * dpml
    start_pos = -0.5 * electron_path_length               # x at simulation t = 0

    def electron_x(t):
        return start_pos + beta * t

    def in_cell(t):
        return start_pos <= electron_x(t) <= -start_pos

    transit_time = electron_path_length / beta

    # ---- the electron as a moving J_x current source -----------------------
    # By default a SMOOTHED (finite-size Gaussian) source is used to suppress the
    # voxel-jump artefacts of a moving point charge (Meep issue #1118).  Set
    # --source-sigma 0 to recover the bare point source (e.g. for convergence
    # studies).  Either way the source carries exactly one electron, so the
    # absolute normalisation is preserved -- verify with --charge-check.
    src_time = mp.ContinuousSource(frequency=1e-7, width=0, is_integrated=True)
    sigma = args.source_sigma / resolution                # voxels -> units of a

    if sigma > 0:
        amp_func = gaussian_current_amp_func(sigma, beta)
        src_size = mp.Vector3(1, 1, 1) * (6 * sigma)      # contain the Gaussian
    else:
        amp_func, src_size = None, mp.Vector3()           # point source
    point_amp = electron_source_amplitude(resolution, beta)

    def move_source(sim: mp.Simulation):
        t = sim.meep_time()
        if not in_cell(t):
            sim.change_sources([])                        # switch off after exit
            return
        center = mp.Vector3(electron_x(t), args.y0, args.z0)
        if sigma > 0:
            src = mp.Source(src_time, component=mp.Ex, center=center,
                            size=src_size, amp_func=amp_func)
        else:
            src = mp.Source(src_time, component=mp.Ex, center=center,
                            amplitude=point_amp)
        sim.change_sources([src])

    # ---- optional VACUUM charge check (the only legitimate flux-box use) ----
    if args.charge_check:
        ds = (a / resolution) ** 2
        box = create_flux_box(mp.Vector3(0, args.y0, args.z0),
                              mp.Vector3(0.2, 0.2, 0.2))
        q_log = []

        def check_charge(sim):
            q_log.append(enclosed_charge(sim, box, ds))

        sim.run(move_source, mp.at_every(transit_time / 50, check_charge),
                until=transit_time)
        q_log = np.array(q_log)
        print(f"[charge-check] mean enclosed charge = {np.nanmean(q_log):.4f} "
              f"(target Q_E_MEEP = {Q_E_MEEP}). "
              f"If this differs, divide the induced field by this value once.")
        return

    # ---- recording window --------------------------------------------------
    # Record E_x on the whole monitor line, at EVERY step, for the full run.
    # The run continues past the electron transit by `ringdown_factor` to
    # capture mode ring-down (capped, because ultra-high-Q lines cannot be
    # resolved this way -- see README; use the modal method for those).
    monitor_width = min(crystal_x_width, electron_path_length)
    total_time = transit_time * (1 + args.ringdown_factor)

    fname = (f"{'EMPTY' if args.empty else 'CRYSTAL'}"
             f"_a{crystal_x_width}-r{int(round(r*1000))}"
             f"_{'c1' if args.cavity else 'c0'}"
             f"_y{int(round(args.y0*1000))}_z{int(round(args.z0*1000))}")
    sim.use_output_directory()

    monitor_vol = mp.Volume(mp.Vector3(0, args.y0, args.z0),
                            mp.Vector3(monitor_width, 0, 0))

    sim.run(
        move_source,
        mp.to_appended(fname, mp.in_volume(monitor_vol, mp.output_efield_x)),
        until=total_time,
    )

    # ---- record the geometry needed by the post-processing -----------------
    dt = sim.fields.dt                       # MEEP time step
    npix = int(round(monitor_width * resolution)) + 1
    x_pix = np.linspace(-monitor_width / 2, monitor_width / 2, npix)  # MEEP units

    with h5py.File(f"EELS_3D-out/EELS_3D-{fname}.h5", "r+") as f:
        for key, val in {
            "a_nm": a_nm, "resolution": resolution, "beta": beta,
            "dt_meep": dt, "start_pos": start_pos, "monitor_width": monitor_width,
            "y0": args.y0, "z0": args.z0, "total_time": total_time,
            "source_sigma_vox": args.source_sigma,
        }.items():
            f.attrs[key] = val
        f.require_dataset("x_pix_meep", x_pix.shape, dtype="<f8", data=x_pix)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--plot", action="store_true")
    p.add_argument("-e", "--empty", action="store_true",
                   help="Vacuum run with the same cell, for the subtraction.")
    p.add_argument("-c", "--cavity", action="store_true")
    p.add_argument("-t", "--test", action="store_true")
    p.add_argument("--charge-check", action="store_true",
                   help="Vacuum-only: verify the source injects one electron.")
    p.add_argument("--ringdown-factor", type=float, default=2.0,
                   help="Extra run time as a multiple of the transit time.")
    p.add_argument("--source-sigma", type=float, default=1.0,
                   help="Width of the smoothed electron source, in VOXELS "
                        "(Meep #1118 artefact suppression). 0 = point source. "
                        "Keep ~1; must stay << slot width / lattice a.")
    p.add_argument("-a", type=int, default=426)
    p.add_argument("-d", type=int, default=220)
    p.add_argument("-x", type=int, default=36)
    p.add_argument("-W", type=float, default=1.2)
    p.add_argument("-s", type=int, default=100)
    p.add_argument("-r", type=float, default=0.245)
    p.add_argument("-n", type=float, default=3.45)
    p.add_argument("-v", type=float, default=100.0, help="Beam voltage [kV].")
    p.add_argument("--y0", type=float, default=0.0, help="Impact parameter y [a].")
    p.add_argument("--z0", type=float, default=0.0, help="Impact parameter z [a].")
    main(p.parse_args())