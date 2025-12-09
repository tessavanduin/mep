import meep as mp
import numpy as np
import argparse

def main(args):
    # a = 426nm
    # r = 0.255*a
    #
    #   1
    #  / \
    # 2 - 3
    #  \ /
    #   4
    #
    # Some parameters to describe the geometry:
    eps = 11.9025       # Silicon relative permittivity
    a = 1               # lattice constant
    w = a               # width
    h = np.sqrt(3)*a    # height
    thickness = 220/426 # thickness of the slab
    r = 0.255           # radius of holes
    dpml = thickness    # PML thickness (z direction only!)

    # The cell dimensions
    cell = mp.Vector3(w, h, 12*thickness)

    b = mp.Block(center=mp.Vector3(0,0,0), size=mp.Vector3(mp.inf,mp.inf, thickness), material=mp.Medium(epsilon=eps))
    c1 = mp.Cylinder(center=mp.Vector3(0,-0.5*h,0), radius=r, height=mp.inf)
    c2 = mp.Cylinder(center=mp.Vector3(0.5*a,0,0), radius=r, height=mp.inf)
    c3 = mp.Cylinder(center=mp.Vector3(-0.5*a,0,0), radius=r, height=mp.inf)
    c4 = mp.Cylinder(center=mp.Vector3(0,0.5*h,0), radius=r, height=mp.inf)
    geometry = [b,c1,c2,c3,c4]


    # Let's choose the pulse so we don't look at vaccuum wavelengths smaller than a (426nm)
    # That is, fcen + df = 1. lambda = 1/(fcen+df)*a = 1/1*426 => fcen = 0.225 and df = 0.775
    fcen = 0.225 # pulse center frequency
    df = 1.525   # pulse freq. width: large df = short impulse

    # We need at least 8px per shortest wavelength in the simulation
    # In the high eps material we have
    # lambda = 1/(n*f) = 1/(3.45*1) = 0.289855...
    # resolution = 8px/lambda = 8/0.289855 = 27.6 = 30
    resolution=20


    pml_layers = [mp.PML(dpml, direction=mp.Z)]

    odd = False
    if (odd):
        # odd H source
        src = [
            mp.Source(src=mp.GaussianSource(fcen, fwidth=df), component=mp.Hz,center=mp.Vector3(0.1234,0,0))
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



    kmax = 0.5      # maximum overall k
    kmin = 0        # minimum overall k
    N = args.N      # Total number of k points
    p = args.p      # number of pieces we break the k points into (number of processor cores)
    kpp = int(N/p)  # number of k points per piece
    Dk = (kmax-kmin)/(N-1) if N != 1 else 0 # Frequency distance associated with one k point of separation
    n = args.n      # n will be the number for the piece we are simulating

    k_interp = kpp-2
    lk = n*kpp*Dk            # lowest k value for this piece
    hk = ((n+1)*kpp - 1)*Dk # highest k value for this piece

    if kpp == 1:
        sim.k_point = mp.Vector3(lk)
        sim.run(mp.after_sources(mp.Harminv(mp.Hz, mp.Vector3(0.1234), fcen, df)), until_after_sources=300)
    else:
        sim.run_k_points(300, mp.interpolate(k_interp, [mp.Vector3(lk), mp.Vector3(hk)]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=4, help="Total number of k points")
    parser.add_argument("-p", type=int, default=1, help="Number of processor cores")
    parser.add_argument("-n", type=int, default=0, help="Number of the region of k points to simulate, must be an element of [0,p-1]")
    args = parser.parse_args()
    main(args)
