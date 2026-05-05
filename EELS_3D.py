import argparse
import meep as mp
import numpy as np
from geometries import *
from helper_functions import E_to_speed

q_e =  1.60217646e-19

def main(args):
    # --- Parameters ---
    a_nm = args.a           # Default: 426
    a = 1                   # Lattice constant
    h = np.sqrt(3)*a        # Unit cell height

    thickness = args.d/a_nm # Slab thickness
    r = args.r              # Radius of holes, r = 0.245*a
    shift = (args.W-1)/2*h  # Amount by which the two halves are shifted up and down (0.1 creates a W1.2 wvg)
    sw = args.s/a_nm        # Slot width, sw = 100nm = 100/426 * a.

    crystal_x_width = args.x # Default: 36

    # Build geometry
    if args.cavity:
        print("Using cavity geometry")
        simulation_domain = SlottedTriangleLatticeCavity(
            r, a, thickness, shift, sw,
            index=args.n, 
            width=crystal_x_width
        )
    else:
        print("Using slab geometry")
        simulation_domain = SlottedTriangleLattice(
            r, a, thickness, shift, sw, 
            index=args.n, width=crystal_x_width
        )

    geometry, cell = simulation_domain.geometry, simulation_domain.cell

    # Add air on all sides
    air_offset = mp.Vector3(12*thickness,12*thickness,12*thickness)
    cell = cell + air_offset

    if args.test and not args.plot: 
        cell = mp.Vector3(2,2,2)

    # --- resolution ---
    resolution = int(np.ceil(a_nm/18))      # convert resolution in terms of nm to resolution in terms of a
    print(f"Resolution: {resolution} pixels per unit")

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

    if args.plot:
        sim.plot3D()
        return
    
    sim.use_output_directory()

    # --- Beam source ---
    # Approximate electron beam as a moving point source
    
    electron_v = E_to_speed(args.v * 1e3)

    beam_length = cell.z - 2 * dpml
    start_z = -0.5 * beam_length

    def electron_pos(t):
        return mp.Vector3(0, 0, start_z + electron_v * t)

    def update_source(sim):
        sim.change_sources([
            mp.Source(
                mp.GaussianSource(frequency=1e-3, fwidth=1e-3), 
                component=mp.Ez,
                center=electron_pos(sim.meep_time())
            )
        ])

    # --- Field output ---

    monitor_volume = mp.Volume(
        center=mp.Vector3(),
        size=cell
    )

    filename = (
        f"{'EMPTY' if args.empty else 'CRYSTAL'}"
        f"_Efield_a{crystal_x_width}_r{int(r*1000)}"
        f"_{'cavity' if args.cavity else 'no_cavity'}"
    )

    # --- Run simulation ---
    sim.run(
        update_source,

        # Recor E-Field
        mp.to_appended(
            "Efield",
            mp.in_volume(
                monitor_volume,
                mp.output_efield
            )
        ),

        until = beam_length / electron_v
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-e", "--empty", action="store_true")
    parser.add_argument("-c", "--cavity", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")

    parser.add_argument("-a", type=int, default=426)
    parser.add_argument("-d", type=int, default=220)
    parser.add_argument("-x", type=int, default=36)
    parser.add_argument("-W", type=float, default=1.2)
    parser.add_argument("-s", type=int, default=100)
    parser.add_argument("-r", type=float, default=0.245)
    parser.add_argument("-n", type=float, default=3.45)
    parser.add_argument("-v", type=float, default=100.0)

    args = parser.parse_args()
    main(args)