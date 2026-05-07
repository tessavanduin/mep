import argparse
import meep as mp
import numpy as np
from geometries import *
from helper_functions import E_to_speed

q_e =  1.60217646e-19

def build_sim(args, empty=False):
    # --- Parameters ---
    a_nm = args.a           # Default: 426
    a = 1                   # Lattice constant
    h = np.sqrt(3)*a        # Unit cell height

    thickness = args.d/a_nm # Slab thickness
    r = args.r              # Radius of holes, r = 0.245*a
    shift = (args.W-1)/2*h  # Amount by which the two halves are shifted up and down (0.1 creates a W1.2 wvg)
    sw = args.s/a_nm        # Slot width, sw = 100nm = 100/426 * a.

    crystal_x_width = args.x # Default: 36

    # --- Geometry ---
    if empty:
        print("Building EMPTY simulation (no geometry)")
        geometry = None

        simulation_domain = SlottedTriangleLattice(
            r, a, thickness, shift, sw,
            index=args.n,
            width=crystal_x_width
        )
    else:
        print("Building CRYSTAL simulation")
        if args.cavity:
            simulation_domain = SlottedTriangleLatticeCavity(
                r, a, thickness, shift, sw,
                index=args.n,
                width=crystal_x_width
            )
        else:
            simulation_domain = SlottedTriangleLattice(
                r, a, thickness, shift, sw,
                index=args.n,
                width=crystal_x_width
            )

        geometry = simulation_domain.geometry

    cell =  simulation_domain.cell + mp.Vector3(12, 12, 12) * thickness

    # --- resolution ---
    resolution = int(np.ceil(a_nm/18))      

    # --- Boundary conditions ---
    dpml = thickness    # PML thickness
    pml_layers = [mp.PML(thickness=dpml)]

    # --- Simulation ---
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=resolution
    )

    # --- Beam source ---
    # Approximate electron beam as a moving point source
    
    electron_v = E_to_speed(args.v * 1e3)
    path_length = cell.x - 2 * dpml
    start_pos = -0.5 * path_length

    def electron_path(t):
        return mp.Vector3(0, 0, start_pos + electron_v * t)

    src_width = 2 / resolution   
    src_size  = mp.Vector3(2*src_width, 2*src_width, 2*src_width)

    def src_amplitude(r):
        rsq = r.dot(r)
        sigma2 = src_width**2
        return -electron_v * np.exp(-rsq / (2*sigma2))
    

    def move_source(sim):
        sim.change_sources([
            mp.Source(
                mp.ContinuousSource(frequency=0.01),
                component=mp.Ez,
                center=electron_path(sim.meep_time()),
                size=src_size,
                amp_func=src_amplitude
            )
        ])


    # --- Field output ---
    filename = (
        f"{args.mode.upper()}"
        f"_a{args.x}"
        f"_r{int(args.r*1000)}"
        f"_x{args.x}"
        f"_W{args.W}"
        f"_cavity{int(args.cavity)}"
    )

    return sim, move_source, path_length, electron_v, filename

def run_sim(sim, move_source, path_length, electron_v, filename):
    sim.run(
        move_source,
        mp.at_every(
            5,
            mp.to_appended(filename, mp.output_efield_z)
        ),
        until=path_length / electron_v
    )


def main(args):

    is_empty = (args.mode == "empty")

    sim, move_source, L, v, fname = build_sim(args, empty=is_empty)

    run_sim(sim, move_source, L, v, fname)

    print("Done.")
    print(f"Mode: {args.mode}")
    print(f"Output file: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["empty", "crystal"],
        required=True,
        help="Run either EMPTY or CRYSTAL simulation"
    )

    parser.add_argument("-a", type=int, default=426)
    parser.add_argument("-d", type=int, default=220)
    parser.add_argument("-x", type=int, default=9)
    parser.add_argument("-W", type=float, default=1.2)
    parser.add_argument("-s", type=int, default=100)
    parser.add_argument("-r", type=float, default=0.245)
    parser.add_argument("-n", type=float, default=3.45)
    parser.add_argument("-v", type=float, default=100.0)
    parser.add_argument("-c", "--cavity", action="store_true")

    args = parser.parse_args()
    main(args)