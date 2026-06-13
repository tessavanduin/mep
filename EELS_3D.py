# EELS_3D_perp.py
# 3D EELS with z-directed electron beam through slotted PhC slab
# Uses mirror symmetries in x and y to reduce domain

import argparse
import meep as mp
import numpy as np
import h5py

from geometries import SlottedTriangleLattice, SlottedTriangleLatticeCavity
from helper_functions import E_to_speed

q_e = 1.60217646e-19


def build_phc_geometry_3d(args):
    a_nm = args.a          # e.g. 426 nm
    a = 1.0                # normalized lattice constant
    h = np.sqrt(3) * a
    thickness = args.d / a_nm
    r = args.r
    shift = (args.W - 1) / 2 * h
    sw = args.s / a_nm
    crystal_x_width = args.x

    if args.cavity:
        print("Using geometry: SlottedTriangleLatticeCavity")
        sim_dom = SlottedTriangleLatticeCavity(
            r, a, thickness, shift, sw, index=args.n, width=crystal_x_width
        )
    else:
        print("Using geometry: SlottedTriangleLattice")
        sim_dom = SlottedTriangleLattice(
            r, a, thickness, shift, sw, index=args.n, width=crystal_x_width
        )

    geometry, cell = sim_dom.geometry, sim_dom.cell

    # add modest air padding in all directions
    pad_xy = 3 * thickness
    pad_z = 3 * thickness
    cell_3d = mp.Vector3(cell.x + 2 * pad_xy,
                         cell.y + 2 * pad_xy,
                         cell.z + 2 * pad_z)

    return geometry, cell_3d, a_nm


def main(args):
    # --- geometry & cell ---
    geometry, cell_size, a_nm = build_phc_geometry_3d(args)

    # --- resolution ---
    resolution = int(np.ceil(a_nm /24))  # ~20 nm
    print(f"RESOLUTION: {resolution} = {a_nm/resolution:.1f} nm")
    print("Accelerating voltage:", args.v, "kV")

    # --- PML & symmetries ---
    dpml = max(cell_size.z / 6.0, 1.0)
    pml_layers = [mp.PML(thickness=dpml)]

    symmetries = [
        mp.Mirror(mp.X, phase=+1),
        mp.Mirror(mp.Y, phase=+1),
    ]

    # --- electron beam along z ---
    v = E_to_speed(args.v * 1e3)  # physical speed; here sets scale
    # start below the slab, end above it
    z_span = cell_size.z - 2 * dpml
    electron_path_length = z_span
    start_z = -0.5 * electron_path_length 
    x_e = 0.0
    y_e = 0.0

    def electron_path(t):
        return mp.Vector3(x_e, y_e, start_z + v * t)

    # dummy source for initialization
    source_tmp = [
        mp.Source(
            mp.ContinuousSource(frequency=1e-10),
            component=mp.Ez,
            center=electron_path(0.0),
        )
    ]

    # --- simulation setup ---
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=None if args.empty else geometry,
        boundary_layers=pml_layers,
        resolution=resolution,
        dimensions=3,
        symmetries=symmetries,
        sources=source_tmp,
        default_material=mp.Medium(index=1.0),
    )

    # --- moving source (beam along z) ---
    def move_source(sim):
        sim.change_sources([
            mp.Source(
                mp.ContinuousSource(frequency=1e-10),
                component=mp.Ez,
                center=electron_path(sim.meep_time()),
            )
        ])

    # --- field sampling: full 3D Ez cube (minus PML) ---
    times = []
    field_snapshots_Ez = []
    field_snapshots_Ex = []
    field_snapshots_Ey = []

    def sample_field(sim):
        Ez_arr = sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cell_size.x - 2*dpml,
                            cell_size.y - 2*dpml,
                            cell_size.z - 2*dpml),
            component=mp.Ez,
        )

        Ex_arr = sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cell_size.x - 2*dpml,
                            cell_size.y - 2*dpml,
                            cell_size.z - 2*dpml),
            component=mp.Ex,
        )

        Ey_arr = sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cell_size.x - 2*dpml,
                            cell_size.y - 2*dpml,
                            cell_size.z - 2*dpml),
            component=mp.Ey,
        )
        times.append(sim.meep_time())
        field_snapshots_Ez.append(Ez_arr)
        field_snapshots_Ex.append(Ex_arr)
        field_snapshots_Ey.append(Ey_arr)


    total_time = electron_path_length / v

    sim.run(
        move_source,
        mp.at_every(1/resolution, sample_field),
        until=total_time,
    )

    # --- save HDF5 ---
    filename = f"EELS_3D_{'Empty' if args.empty else 'Crystal'}_{'Cavity' if args.cavity else 'NoCavity'}_a{args.x}_r{int(args.r*1000)}"
    with h5py.File(f"{filename}.h5", "w") as f:
        f.create_dataset("Ez", data=np.array(field_snapshots_Ez))
        f.create_dataset("Ex", data=np.array(field_snapshots_Ex))
        f.create_dataset("Ey", data=np.array(field_snapshots_Ey))
        f.create_dataset("time", data=np.array(times))

        f.attrs["cell_x"] = cell_size.x
        f.attrs["cell_y"] = cell_size.y
        f.attrs["cell_z"] = cell_size.z
        f.attrs["resolution"] = resolution
        f.attrs["v_electron"] = v


    print("Done. Saved:", filename + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--empty", action="store_true",
                        help="Run with empty domain (no PhC).")
    parser.add_argument("-c", "--cavity", action="store_true",
                        help="Use cavity geometry.")
    parser.add_argument("-a", type=int, default=426,
                        help="Lattice constant in nm.")
    parser.add_argument("-d", type=int, default=220,
                        help="Slab thickness in nm.")
    parser.add_argument("-x", type=int, default=36,
                        help="X length of the crystal (in units of a).")
    parser.add_argument("-W", type=float, default=1.2,
                        help="Width of the center waveguide (in units of a).")
    parser.add_argument("-s", type=int, default=100,
                        help="Slot width in nm.")
    parser.add_argument("-r", type=float, default=0.245,
                        help="Radius/a.")
    parser.add_argument("-n", type=float, default=3.45,
                        help="Refractive index of slab.")
    parser.add_argument("-v", type=float, default=100.0,
                        help="Electron accelerating voltage in kV.")
    args = parser.parse_args()
    main(args)
