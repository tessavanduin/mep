import argparse
import meep as mp
import numpy as np
import h5py
from geometries import *
from helper_functions import E_to_speed
from helper_functions import create_flux_box
from helper_functions import integrate_flux_box
from divergence import divE_at_point

q_e =  1.60217646e-19

def main(args):
    a_nm = args.a
    # Some parameters to describe the geometry:
    a = 1                   # Lattice constant
    h = np.sqrt(3)*a        # Unit cell height
    thickness = 220/a_nm    # Slab thickness
    r = args.r              # Radius of holes, r = 0.245*a
    shift = (args.W-1)/2*h  # Amount by which the two halves are shifted up and down (0.1 creates a W1.2 wvg)
    sw = 100/a_nm           # Slot width, sw = 100nm = 100/426 * a.

    crystal_x_width = 36
    simulation_domain = SlottedTriangleLatticeCavity(r, a, thickness, shift, sw, index=args.n, width=crystal_x_width)
    geometry, cell = simulation_domain.geometry, simulation_domain.cell

    air_offset = mp.Vector3(1,1,1)#*12*thickness
    cell = cell + air_offset


    # resolution of 18 nm
    resolution=np.ceil(a_nm/18) # convert resolution in terms of nm to resolution in terms of a
    print(f"RESOLUTION: {resolution} = {a_nm/resolution} nm")

    dpml = thickness    # PML thickness
    # pml_layers = [mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.Z)]
    pml_layers = [mp.PML(thickness=dpml)]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        symmetries=None,
                        resolution=resolution)

    if args.plot:
        if sim.dimensions == 3:
            sim.plot3D()
        elif sim.dimensions == 2:
            sim.plot2D()
        return
    filename = f"_{'EMPTY' if args.e else 'CRYSTAL'}_PML_a{crystal_x_width}-r" + str(int(r*1000)) + "-ex_air_flx3"
    sim.use_output_directory()


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

    # Find flux through a small box following the electron
    b = mp.Vector3(0.1,0.1,0.1) # cube size
    def get_flux(sim: mp.Simulation):
        b_center = electron_path(sim.meep_time()) # same position as electron
        flux_box = create_flux_box(b_center, b)
        flux = integrate_flux_box(sim, flux_box, ds)
        flux_total.append(flux)

    # Find flux through big stationary box close to the edge of the simulation domain
    big_box = cell - 2.1*mp.Vector3(dpml,dpml,dpml)
    flux_box = create_flux_box(mp.Vector3(), big_box)
    def get_flux_2(sim: mp.Simulation):
        flux = integrate_flux_box(sim, flux_box, ds)
        flux_total.append(flux)

    def get_divergence(sim: mp.Simulation):
        flux = divE_at_point(sim, electron_path(sim.meep_time()))
        flux_total.append(flux)


    monitor_width = crystal_x_width # monitor_width < electron_path_length
    start_pos_till_monitor = (electron_path_length - monitor_width)/2
    start_time = start_pos_till_monitor/electron_v
    end_time   = (start_pos_till_monitor + monitor_width)/electron_v

    sim.run(move_source,
        mp.after_time(
            start_time,
            mp.before_time(
                end_time,
                get_flux_2,
                mp.to_appended(filename, mp.in_volume(mp.Volume(mp.Vector3(), mp.Vector3(monitor_width, 0, 0)), mp.output_efield_x))
            )
        ),
        until=electron_path_length / electron_v
    )

    with h5py.File(f"EELS_3D-out/EELS_3D{filename}.h5", "r+") as f:
        dset = f.require_dataset("flux", (len(flux_total)), dtype='<f8', data=flux_total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action='store_false', help="Plot the defined geometry.")
    parser.add_argument("-e", "--empty", action='store_true', help="Run the simulation in an empty simulation " \
    "domain with size as if the specified geometry would be there.")
    parser.add_argument("-a", type=int, default=426, help="Allows for specifying parameters as fraction of a." \
    "The other non-ratio parameters would be divided by 'a' prior to being used in the simulation." \
    "This is the same as setting a=1 and specifying all parameters as ratios.")
    parser.add_argument("-d", type=int, default=220, help="Thickness of your structure.")
    parser.add_argument("-W", type=int, default=1.2, help="Width of the center waveguide.")
    parser.add_argument("-r", type=float, default=0.245, help="Ratio of hole radius to a. That is, radius/a.")
    parser.add_argument("-n", type=float, default=3.45, help="Square root of the material permittivity.")
    args = parser.parse_args()
    main(args)