import meep as mp
import numpy as np
import argparse

def crystal_unit_cell(r: float, a=1, T: mp.Vector3 = mp.Vector3(0,0,0)):
    """Create all objects associated with 1 unit cell of the following crystal lattice:
      1
     . .
    2   3
     . .
      4
    made out of four holes that form a diamond shape consisting of two equilateral triangles with each edge equal to a.
    Args:
        r (float): Hole radius
        a (int, optional): Primitive lattice vector. Defaults to 1.
        T (mp.Vector3, optional): Translation of the unit cell. Defaults to mp.Vector3().
    """
    # Note that the unit cell's width is equal to a
    h = np.sqrt(3)*a    # height of the unit cell
    c1 = mp.Cylinder(center=mp.Vector3(0,-0.5*h,0) + T, radius=r)
    c2 = mp.Cylinder(center=mp.Vector3(0.5*a,0,0) + T, radius=r)
    c3 = mp.Cylinder(center=mp.Vector3(-0.5*a,0,0) + T, radius=r)
    c4 = mp.Cylinder(center=mp.Vector3(0,0.5*h,0) + T, radius=r)
    return [c1, c2, c3, c4]

def main(args):
    # a = 426nm
    # r = 0.255*a
    # sw = 100nm = 100/426 * a. # slot width
    #
    # Some parameters to describe the geometry:
    eps = 11.9025       # Silicon relative permittivity
    a = 1               # lattice constant
    w = a               # unit cell width
    h = np.sqrt(3)*a    # unit cell height
    thickness = 220/426 # slab thickness
    r = 0.255           # radius of holes
    shift = 0.1*h       # Amount by which the two halves are shifted up and down (0.1 creates a W1.2 wvg)
    sw = 100/426        # slot width
    dpml = thickness    # PML thickness (y direction only)

    # The cell dimensions
    cell_h = 6*h + 2*shift
    cell = mp.Vector3(w, cell_h, 12*thickness)

    # Geometry of objects
    b = mp.Block(center=mp.Vector3(0,0,0), size=mp.Vector3(mp.inf,mp.inf, thickness), material=mp.Medium(epsilon=eps))
    geometry = [b]
    for i in np.arange(-3,4):
        T = i*h+np.sign(i)*shift
        if T != 0:
            geometry.extend(crystal_unit_cell(r, a, mp.Vector3(0,T,0)))

    # cover top and bottom holes
    geometry += [mp.Block(size=mp.Vector3(mp.inf,2*r,thickness), center=mp.Vector3(0,-0.5*cell_h,0), material=mp.Medium(epsilon=eps)),
                mp.Block(size=mp.Vector3(mp.inf,2*r,thickness), center=mp.Vector3(0,0.5*cell_h,0), material=mp.Medium(epsilon=eps))]

    # create the air slot
    geometry += [mp.Block(size=mp.Vector3(mp.inf,sw,mp.inf))]

    # create air above and below the slab
    geometry += [mp.Block(size=mp.Vector3(mp.inf,mp.inf,2), center=mp.Vector3(0,0,thickness+1)),
                mp.Block(size=mp.Vector3(mp.inf,mp.inf,2), center=mp.Vector3(0,0,-(thickness+1)))]


    # Let's choose the pulse so we don't look at vaccuum wavelengths smaller than a (426nm)
    # That is, fcen + df = 1. lambda = 1/(fcen+df)*a = 1/1*426 => fcen = 0.25 and df = 0.75
    fcen = 0.225 # pulse center frequency
    df = 1.525     # pulse freq. width: large df = short impulse

    # We need at least 8px per shortest wavelength in the simulation
    # In the high eps material we have
    # lambda = 1/(n*f) = 1/(3.45*1) = 0.289855...
    # resolution = 8px/lambda = 8/0.289855 = 27.6 = 30
    resolution=20


    pml_layers = [mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.Z)]

    odd = True
    if odd:
        # odd H source
        src = [
            mp.Source(src=mp.GaussianSource(fcen, fwidth=df), component=mp.Hz,center=mp.Vector3(0.1234,0))
        ]
        sym = [mp.Mirror(direction=mp.Y, phase=-1), mp.Mirror(direction=mp.Z, phase=-1)]
    else:
        # even Ex source
        src = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df), component=mp.Ex, center=mp.Vector3(0.1234, 0))]
        sym = [mp.Mirror(direction=mp.Y, phase=1), mp.Mirror(direction=mp.Z, phase=-1)]


    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        sources=src,
                        symmetries=None,
                        resolution=resolution)

    # sim.plot3D()

    # sim.use_output_directory()

    kmax = 0.5      # maximum overall k
    kmin = 0        # minimum overall k
    N = args.N      # Total number of k points
    p = args.p      # number of pieces we break the k points into (number of processor cores)
    kpp = int(N/p)  # number of k points per piece
    Dk = (kmax-kmin)/(N-1) if N != 1 else 0 # Frequency distance associated with one k point of separation
    n = args.n      # n will be the number for the piece we are simulating

    k_interp = kpp-2
    l = n*kpp*Dk            # lowest k value for this piece
    h = ((n+1)*kpp - 1)*Dk # highest k value for this piece

    if kpp == 1:
        sim.k_point = mp.Vector3(l)
        sim.run(mp.after_sources(mp.Harminv(mp.Hz, mp.Vector3(0.1234), fcen, df)), until_after_sources=300)
    else:
        sim.run_k_points(300, mp.interpolate(k_interp, [mp.Vector3(l), mp.Vector3(h)]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=4, help="Total number of k points")
    parser.add_argument("-p", type=int, default=1, help="Number of processor cores")
    parser.add_argument("-n", type=int, default=0, help="Number of the region of k points to simulate, must be an element of [0,p-1]")
    args = parser.parse_args()
    main(args)
