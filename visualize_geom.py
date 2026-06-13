#!/usr/bin/env python
"""
Standalone geometry visualization script.
Run with: python visualize_geom.py [--cavity] [--r 0.245] [--W 1.2] [--s 100] [--d 220] [--a 426] [--n 3.45] [--x 9]
"""

import argparse
import os
import numpy as np
from geometries import SlottedTriangleLattice, SlottedTriangleLatticeCavity
from helper_functions import visualize_geometry

def main():
    parser = argparse.ArgumentParser(description="Visualize photonic crystal geometry")
    parser.add_argument("-a", type=int, default=426, help="Lattice constant in nm")
    parser.add_argument("-d", type=int, default=220, help="Slab thickness in nm")
    parser.add_argument("-x", type=int, default=9, help="Crystal width in unit cells")
    parser.add_argument("-W", type=float, default=1.2, help="Waveguide width")
    parser.add_argument("-s", type=int, default=100, help="Slot width in nm")
    parser.add_argument("-r", type=float, default=0.245, help="Hole radius in units of a")
    parser.add_argument("-n", type=float, default=3.45, help="Refractive index")
    parser.add_argument("-c", "--cavity", action="store_true", help="Include cavity")
    
    args = parser.parse_args()
    
    # Convert to natural units
    a_nm = args.a
    a = 1
    h = np.sqrt(3) * a
    
    thickness = args.d / a_nm
    r = args.r
    shift = (args.W - 1) / 2 * h
    sw = args.s / a_nm
    crystal_x_width = args.x
    
    print(f"Visualizing geometry:")
    print(f"  Lattice constant: {a_nm} nm")
    print(f"  Hole radius: {r:.3f}a")
    print(f"  Waveguide width: {args.W}a")
    print(f"  Shift: {shift:.3f}a")
    print(f"  Slot width: {sw:.3f}a")
    print(f"  Crystal width: {crystal_x_width} unit cells")
    
    # Create geometry
    if args.cavity:
        print("  Type: SlottedTriangleLatticeCavity")
        simulation_domain = SlottedTriangleLatticeCavity(
            r, a, thickness, shift, sw,
            index=args.n,
            width=crystal_x_width
        )
    else:
        print("  Type: SlottedTriangleLattice")
        simulation_domain = SlottedTriangleLattice(
            r, a, thickness, shift, sw,
            index=args.n,
            width=crystal_x_width
        )
    
    geometry = simulation_domain.geometry
    
    # Visualize
    out_dir = "Geometries"
    os.makedirs(out_dir, exist_ok=True)
    mode = "cavity" if args.cavity else "nocavity"

    filename = (
        f"{mode}"
        f"_a{args.a}"
        f"_x{args.x}"
        f"_r{int(args.r*1000)}"
        f"_W{args.W:.2f}"
        f"_s{args.s}"
        f"_n{args.n:.2f}"
        f"_d{args.d}"
    )
    filepath = os.path.join(out_dir, filename + ".png")
    visualize_geometry(geometry, crystal_x_width, a, h, r, shift, mode, filepath)

if __name__ == "__main__":
    main()
