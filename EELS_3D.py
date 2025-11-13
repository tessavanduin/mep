import meep as mp
import numpy as np
import h5py
from geometries import *
from helper_functions import E_to_speed

q_e =  1.60217646e-19

# Some parameters to describe the geometry:
a = 1               # Lattice constant
h = np.sqrt(3)*a    # Unit cell height
thickness = 220/426 # Slab thickness
r = 0.245           # Radius of holes, r = 0.245*a
shift = 0.1*h       # Amount by which the two halves are shifted up and down (0.1 creates a W1.2 wvg)
sw = 100/426        # Slot width, sw = 100nm = 100/426 * a.

crystal_x_width = 36
simulation_domain = SlottedTriangleLatticeCavity(r, a, thickness, shift, sw, index=3.45, width=crystal_x_width)
geometry, cell = simulation_domain.geometry, simulation_domain.cell

air_offset = mp.Vector3(12*thickness,12*thickness,12*thickness)
cell = cell + air_offset


# resolution of 18 nm
resolution=np.ceil(426/18) # convert resolution in terms of nm to resolution in terms of a
print(f"RESOLUTION: {resolution} = {426/resolution} nm")

dpml = thickness    # PML thickness
# pml_layers = [mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.Z)]
pml_layers = [mp.PML(thickness=dpml)]

sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    boundary_layers=pml_layers,
                    symmetries=None,
                    resolution=resolution)


# 100keV electron velocity
electron_v = E_to_speed(1e5)

# model the electron from the edge of the PML to the edge of the other PML
electron_path_length = cell.x - 2*dpml
start_pos = -0.5 * electron_path_length
def electron_path(t):
    return mp.Vector3( start_pos + electron_v * t, 0, 0)

charge_density = resolution**3 * -q_e

def move_source(sim: mp.Simulation):
    sim.change_sources(
        [
            mp.Source(
                mp.ContinuousSource(frequency=1e-20),
                component=mp.Ex,
                center=electron_path(sim.meep_time()),
                # amplitude=charge_density*electron_v
            )
        ]
    )

flux_total = []
ds = (a/resolution)**2 # surface element in units of a
b = 0.1 # cube size
g = 0.5*b
def get_flux(sim: mp.Simulation):
    flux = np.empty((3,2))
    b_center = electron_path(sim.meep_time()) # same position as electron
    flux[0,0] =  np.sum(sim.get_array(mp.Ex, center=b_center+mp.Vector3( g, 0, 0), size=mp.Vector3(0,b,b))) # x pos
    flux[0,1] = -np.sum(sim.get_array(mp.Ex, center=b_center+mp.Vector3(-g, 0, 0), size=mp.Vector3(0,b,b))) # x neg
    flux[1,0] =  np.sum(sim.get_array(mp.Ey, center=b_center+mp.Vector3( 0, g, 0), size=mp.Vector3(b,0,b))) # y pos
    flux[1,1] = -np.sum(sim.get_array(mp.Ey, center=b_center+mp.Vector3( 0,-g, 0), size=mp.Vector3(b,0,b))) # y neg
    flux[2,0] =  np.sum(sim.get_array(mp.Ez, center=b_center+mp.Vector3( 0, 0, g), size=mp.Vector3(b,b,0))) # z pos
    flux[2,1] = -np.sum(sim.get_array(mp.Ez, center=b_center+mp.Vector3( 0, 0,-g), size=mp.Vector3(b,b,0))) # z neg
    flux_total.append(np.sum(flux)*ds)

# sim.plot3D()

sim.use_output_directory()

monitor_width = crystal_x_width # monitor_width < electron_path_length
start_pos_till_monitor = (electron_path_length - monitor_width)/2
start_time = start_pos_till_monitor/electron_v
end_time   = (start_pos_till_monitor + monitor_width)/electron_v

sim.run(move_source,
    mp.after_time(
        start_time,
        mp.before_time(
            end_time,
            get_flux,
            mp.to_appended("ex", mp.in_volume(mp.Volume(mp.Vector3(), mp.Vector3(monitor_width, 0, 0)), mp.output_efield_x))
        )
    ),
    until=electron_path_length / electron_v
)

with h5py.File("EELS_3D-out/EELS_3D-ex.h5", "r+") as f:
    dset = f.require_dataset("flux", (len(flux_total)), dtype='<f8', data=flux_total)