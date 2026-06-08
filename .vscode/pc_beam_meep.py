#!/usr/bin/env python3
"""
pc_beam_meep_moving.py

Relativistic electron beam interacting with a slotted photonic crystal slab
waveguide (with optional cavity) in 3D Meep (Python API), using a single
finite-size moving current source (no discrete packets).
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import meep as mp


# =========================
# 1. Utility and physics
# =========================

E_CHARGE = 1.602176634e-19  # Coulomb
M_E = 9.10938356e-31        # kg
C0 = 299792458.0            # m/s
M_E_C2_KEV = 511.0          # keV


@dataclass
class BeamParameters:
    kinetic_energy_kev: float
    gamma: float
    beta: float
    v_m_per_s: float
    v_meep: float


def compute_relativistic_beam(kinetic_energy_kev: float) -> BeamParameters:
    """Compute relativistic parameters for an electron given kinetic energy in keV."""
    gamma = 1.0 + kinetic_energy_kev / M_E_C2_KEV
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    v = beta * C0
    v_meep = beta  # c = 1 in Meep units
    return BeamParameters(
        kinetic_energy_kev=kinetic_energy_kev,
        gamma=gamma,
        beta=beta,
        v_m_per_s=v,
        v_meep=v_meep,
    )


# =========================
# 2. Geometry construction
# =========================

@dataclass
class GeometryParameters:
    a: float
    slab_thickness: float
    n_silicon: float
    r_hole: float
    slot_width: float
    slot_shift: float
    nx: int
    ny: int
    nz: int
    cavity: bool
    cavity_length: float
    cavity_taper: float


def build_pc_slab_geometry(params: GeometryParameters) -> List[mp.GeometricObject]:
    """
    Build a triangular-lattice photonic crystal slab with a central slot waveguide
    and optional cavity.

    Coordinate system:
    - z: electron beam direction and waveguide axis
    - y: slab thickness direction (vertical)
    - x: transverse direction in the slab plane
    """
    geom: List[mp.GeometricObject] = []

    si = mp.Medium(index=params.n_silicon)

    half_thickness = params.slab_thickness / 2.0
    half_slot = params.slot_width / 2.0

    upper_slab = mp.Block(
        size=mp.Vector3(mp.inf, half_thickness - half_slot, mp.inf),
        center=mp.Vector3(params.slot_shift, (half_thickness + half_slot) / 2.0, 0),
        material=si,
    )
    lower_slab = mp.Block(
        size=mp.Vector3(mp.inf, half_thickness - half_slot, mp.inf),
        center=mp.Vector3(-params.slot_shift, -(half_thickness + half_slot) / 2.0, 0),
        material=si,
    )

    geom.append(upper_slab)
    geom.append(lower_slab)

    for ix in range(-params.nx, params.nx + 1):
        for iz in range(-params.nz, params.nz + 1):
            x = ix * params.a + 0.5 * iz * params.a
            z = (math.sqrt(3) / 2.0) * iz * params.a

            if params.cavity:
                if abs(z) < params.cavity_length / 2.0:
                    continue
                edge = params.cavity_length / 2.0
                dz = abs(abs(z) - edge)
                taper_range = params.a * 2.0
                if dz < taper_range:
                    taper_factor = 1.0 - params.cavity_taper * (1.0 - dz / taper_range)
                else:
                    taper_factor = 1.0
                r_hole = params.r_hole * taper_factor
            else:
                r_hole = params.r_hole

            if abs(x) < params.slot_width / 2.0:
                continue

            cyl = mp.Cylinder(
                radius=r_hole,
                height=params.slab_thickness * 2.0,
                axis=mp.Vector3(0, 1, 0),
                center=mp.Vector3(x, 0, z),
                material=mp.air,
            )
            geom.append(cyl)

    return geom


# =========================
# 3. Moving electron source
# =========================

@dataclass
class ElectronSourceParameters:
    beam: BeamParameters
    z_start: float
    z_end: float
    spatial_sigma: float


def build_moving_electron_source(params: ElectronSourceParameters):
    """
    Build a single finite-size moving electron source using sim.change_sources,
    following the Meep 'amp_func + moving center' pattern.

    - component: Jz (current along z)
    - center: (0, 0, z(t)) with z(t) = z_start + v_meep * t
    - spatial profile: Gaussian with width spatial_sigma
    - time dependence: ContinuousSource (effectively DC), motion encoded in center
    """
    q = -1.0
    v_meep = params.beam.v_meep
    src_width = params.spatial_sigma

    srctime = mp.ContinuousSource(frequency=1e-10)
    src_size = mp.Vector3(src_width, src_width, src_width)
    z0 = params.z_start

    def src_amplitude(r):
        rsq = r.dot(r)
        ssq2 = 2 * src_width * src_width
        return q * v_meep * np.exp(-rsq / ssq2) / math.sqrt(math.pi * ssq2)

    initial_source = mp.Source(
        src=srctime,
        component=mp.Jz,
        center=mp.Vector3(0, 0, z0),
        size=src_size,
        amp_func=src_amplitude,
    )

    def move_source(sim: mp.Simulation):
        t = sim.meep_time()
        znew = z0 + v_meep * t
        sim.change_sources([
            mp.Source(
                src=srctime,
                component=mp.Jz,
                center=mp.Vector3(0, 0, znew),
                size=src_size,
                amp_func=src_amplitude,
            )
        ])

    return [initial_source], move_source


# =========================
# 4. Simulation setup
# =========================

@dataclass
class SimulationParameters:
    cell_size: mp.Vector3
    resolution: int
    dpml: float
    sim_time: float
    output_interval: float
    output_dir: str


@dataclass
class Metadata:
    geometry: GeometryParameters
    beam: BeamParameters
    electron_source: ElectronSourceParameters
    simulation: SimulationParameters


def setup_simulation(
    geom: List[mp.GeometricObject],
    sources: List[mp.Source],
    sim_params: SimulationParameters,
) -> mp.Simulation:
    pml_layers = [mp.PML(sim_params.dpml)]
    sim = mp.Simulation(
        cell_size=sim_params.cell_size,
        geometry=geom,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=sim_params.resolution,
        default_material=mp.air,
    )
    return sim


# =========================
# 5. Post-processing hooks
# =========================

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_metadata(metadata: Metadata, output_dir: str):
    ensure_dir(output_dir)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "geometry": asdict(metadata.geometry),
                "beam": asdict(metadata.beam),
                "electron_source": {
                    "beam": asdict(metadata.electron_source.beam),
                    "z_start": metadata.electron_source.z_start,
                    "z_end": metadata.electron_source.z_end,
                    "spatial_sigma": metadata.electron_source.spatial_sigma,
                },
                "simulation": {
                    "cell_size": {
                        "x": metadata.simulation.cell_size.x,
                        "y": metadata.simulation.cell_size.y,
                        "z": metadata.simulation.cell_size.z,
                    },
                    "resolution": metadata.simulation.resolution,
                    "dpml": metadata.simulation.dpml,
                    "sim_time": metadata.simulation.sim_time,
                    "output_interval": metadata.simulation.output_interval,
                    "output_dir": metadata.simulation.output_dir,
                },
            },
            f,
            indent=2,
        )


def sample_fields(sim: mp.Simulation, t: float, output_dir: str):
    ensure_dir(output_dir)
    ex = sim.get_array(component=mp.Ex)
    ey = sim.get_array(component=mp.Ey)
    ez = sim.get_array(component=mp.Ez)
    hx = sim.get_array(component=mp.Hx)
    hy = sim.get_array(component=mp.Hy)
    hz = sim.get_array(component=mp.Hz)

    fname = os.path.join(output_dir, f"fields_t_{t:.4f}.npz")
    np.savez_compressed(
        fname,
        t=t,
        ex=ex,
        ey=ey,
        ez=ez,
        hx=hx,
        hy=hy,
        hz=hz,
    )


# =========================
# 6. Main driver
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Relativistic electron in a slotted photonic crystal slab (Meep 3D) with moving source."
    )

    # Geometry
    parser.add_argument("--a", type=float, default=1.0, help="Lattice constant (Meep units).")
    parser.add_argument("--slab-thickness", type=float, default=0.5, help="Slab thickness (units of a).")
    parser.add_argument("--n-silicon", type=float, default=3.45, help="Silicon refractive index.")
    parser.add_argument("--r-hole", type=float, default=0.3, help="Hole radius (units of a).")
    parser.add_argument("--slot-width", type=float, default=0.2, help="Slot width (units of a).")
    parser.add_argument("--slot-shift", type=float, default=0.0, help="Lateral shift of slab halves (units of a).")
    parser.add_argument("--nx", type=int, default=5, help="Number of lattice periods in x (each side).")
    parser.add_argument("--nz", type=int, default=10, help="Number of lattice periods in z (each side).")
    parser.add_argument("--cavity", action="store_true", help="Enable cavity by removing holes.")
    parser.add_argument("--cavity-length", type=float, default=3.0, help="Cavity length along z (units of a).")
    parser.add_argument("--cavity-taper", type=float, default=0.2, help="Fractional taper of hole radius near cavity.")

    # Beam
    parser.add_argument("--kev", type=float, default=200.0, help="Electron kinetic energy in keV.")

    # Simulation domain
    parser.add_argument("--dpml", type=float, default=1.0, help="PML thickness (units of a).")
    parser.add_argument("--padding-x", type=float, default=2.0, help="Padding in x beyond holes (units of a).")
    parser.add_argument("--padding-y", type=float, default=2.0, help="Padding in y beyond slab (units of a).")
    parser.add_argument("--padding-z", type=float, default=2.0, help="Padding in z beyond holes (units of a).")

    # Resolution and time
    parser.add_argument("--resolution", type=int, default=60, help="Resolution (pixels per a).")
    parser.add_argument("--sim-time", type=float, default=None, help="Total simulation time (units of a/c).")
    parser.add_argument("--output-interval", type=float, default=0.5, help="Time interval between field snapshots.")

    # Electron source spatial width
    parser.add_argument("--spatial-sigma", type=float, default=0.1, help="Spatial width of moving source (units of a).")

    # Output
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for field data and metadata.")

    return parser.parse_args()


def main():
    args = parse_args()

    beam = compute_relativistic_beam(args.kev)
    print("=== Beam parameters ===")
    print(f"Kinetic energy: {beam.kinetic_energy_kev:.3f} keV")
    print(f"gamma: {beam.gamma:.6f}")
    print(f"beta = v/c: {beam.beta:.6f}")
    print(f"v (m/s): {beam.v_m_per_s:.3e}")
    print(f"v (Meep units, c=1): {beam.v_meep:.6f}")

    geom_params = GeometryParameters(
        a=args.a,
        slab_thickness=args.slab_thickness,
        n_silicon=args.n_silicon,
        r_hole=args.r_hole,
        slot_width=args.slot_width,
        slot_shift=args.slot_shift,
        nx=args.nx,
        ny=1,
        nz=args.nz,
        cavity=args.cavity,
        cavity_length=args.cavity_length,
        cavity_taper=args.cavity_taper,
    )

    geometry = build_pc_slab_geometry(geom_params)

    Lx = 2 * args.nx * args.a + 2 * args.padding_x + 2 * args.dpml
    Lz = 2 * args.nz * args.a + 2 * args.padding_z + 2 * args.dpml
    Ly = args.slab_thickness + 2 * args.padding_y + 2 * args.dpml

    cell_size = mp.Vector3(Lx, Ly, Lz)

    z_start = -Lz / 2.0 + args.dpml + 0.5
    z_end = Lz / 2.0 - args.dpml - 0.5

    e_src_params = ElectronSourceParameters(
        beam=beam,
        z_start=z_start,
        z_end=z_end,
        spatial_sigma=args.spatial_sigma,
    )

    sources, move_source = build_moving_electron_source(e_src_params)

    if args.sim_time is None:
        travel_time = (z_end - z_start) / beam.v_meep
        sim_time = travel_time + 10.0
    else:
        sim_time = args.sim_time

    sim_params = SimulationParameters(
        cell_size=cell_size,
        resolution=args.resolution,
        dpml=args.dpml,
        sim_time=sim_time,
        output_interval=args.output_interval,
        output_dir=args.output_dir,
    )

    metadata = Metadata(
        geometry=geom_params,
        beam=beam,
        electron_source=e_src_params,
        simulation=sim_params,
    )
    save_metadata(metadata, args.output_dir)

    sim = setup_simulation(geometry, sources, sim_params)

    print("=== Starting simulation ===")
    t = 0.0
    while t < sim_params.sim_time:
        t_next = min(t + sim_params.output_interval, sim_params.sim_time)
        sim.run(move_source, until=t_next - t)
        t = t_next
        print(f"Sampling fields at t = {t:.4f}")
        sample_fields(sim, t, os.path.join(args.output_dir, "fields"))

    print("=== Simulation complete ===")


if __name__ == "__main__":
    main()