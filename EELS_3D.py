import meep as mp
import numpy as np
from geometries import SlottedTriangleLattice
from helper_functions import E_to_speed

# Some parameters to describe the geometry:
a = 1               # Lattice constant
h = np.sqrt(3)*a    # Unit cell height
thickness = 220/426 # Slab thickness 
r = 0.255           # Radius of holes, r = 0.255*a
shift = 0.1*h       # Amount by which the two halves are shifted up and down (0.1 creates a W1.2 wvg)
sw = 100/426        # Slot width, sw = 100nm = 100/426 * a.

simulation_domain = SlottedTriangleLattice(r, a, thickness, shift, sw, index=3.45)
geometry, cell = simulation_domain.geometry, simulation_domain.cell

# resolution of 18 nm
resolution=np.ceil(426/18) # convert resolution in terms of nm to resolution in terms of a
print(f"RESOLUTION: {resolution} = {426/resolution} nm")

dpml = thickness    # PML thickness (y direction only)
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

sim.plot3D()

sim.use_output_directory()

sim.run(mp.at_beginning(mp.output_epsilon),
    move_source,
    mp.output_png(mp.Hz, "-0 -z 0 -Zc dkbluered -C EELS_3D-out/EELS_3D-eps-000000.00.h5"),
    until=cell.x / electron_v,
)