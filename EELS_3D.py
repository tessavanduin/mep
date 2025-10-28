import meep as mp
import numpy as np
from geometries import *
from helper_functions import E_to_speed

# Some parameters to describe the geometry:
a = 1               # Lattice constant
h = np.sqrt(3)*a    # Unit cell height
thickness = 220/426 # Slab thickness 
r = 0.245           # Radius of holes, r = 0.245*a
shift = 0.1*h       # Amount by which the two halves are shifted up and down (0.1 creates a W1.2 wvg)
sw = 100/426        # Slot width, sw = 100nm = 100/426 * a.

simulation_domain = SlottedTriangleLatticeCavity(r, a, thickness, shift, sw, index=3.45)
geometry, cell = simulation_domain.geometry, simulation_domain.cell

# resolution of 18 nm
resolution=np.ceil(426/18) # convert resolution in terms of nm to resolution in terms of a
print(f"RESOLUTION: {resolution} = {426/resolution} nm")

dpml = thickness    # PML thickness
pml_layers = [mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.Z)]

sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    boundary_layers=pml_layers,
                    symmetries=None,
                    resolution=resolution)


# 100keV electron velocity
electron_v = E_to_speed(1e5)

def move_source(sim):
    sim.change_sources(
        [
            mp.Source(
                mp.ContinuousSource(frequency=1e-10),
                component=mp.Ex,
                center=mp.Vector3(-0.5 * cell.x + electron_v * sim.meep_time(), 0, 0),
            )
        ]
    )

# sim.plot3D()

sim.use_output_directory()

sim.run(mp.at_beginning(mp.in_volume(mp.Volume(mp.Vector3(), mp.Vector3(cell.x,cell.y)), mp.output_epsilon)),
        move_source,
        mp.to_appended("ex", mp.in_volume(mp.Volume(mp.Vector3(), mp.Vector3(cell.x,cell.y)), mp.output_efield_x)),
        until=cell.x / electron_v,)
