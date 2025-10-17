import meep as mp
import numpy as np

class SimulationDomain:
    def __init__(self):
        self.geometry = []
        self.cell = mp.Vector3()

class TriangleUnitCell:
    def __init__(self, r: float, a=1, coords: mp.Vector3=mp.Vector3(0,0,0)):
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
        c1 = mp.Cylinder(center=mp.Vector3(0,-0.5*h,0) + coords, radius=r)
        c2 = mp.Cylinder(center=mp.Vector3(0.5*a,0,0) + coords, radius=r)
        c3 = mp.Cylinder(center=mp.Vector3(-0.5*a,0,0) + coords, radius=r)
        c4 = mp.Cylinder(center=mp.Vector3(0,0.5*h,0) + coords, radius=r)

        self.geometry = [c1, c2, c3, c4]
        self.cell = mp.Vector3(a, h, 0)


def create_slab_holes(r, a, h, shift, coords, geometry):
    for i in np.append(np.arange(-3,0),np.arange(1,4)):
            T = i*h+np.sign(i)*shift
            geometry.extend(TriangleUnitCell(r, a, coords + mp.Vector3(0,T,0)).geometry)

class SlottedTriangleLattice:
    def __init__(self,  r: float, a: float=1, thickness: float=1, shift: float=0, sw: float=0, index: float=3.45, coords: mp.Vector3=mp.Vector3(0,0,0)):
        """Create a triangle lattice in a slab with air slot in the middle parallel to the x-direction.
        The two halves of the crystal can be shifted up and down to make the waveguide wider.

        Args:
            r (float): Hole radius
            a (int, optional): Primitive lattice vector. Defaults to 1.
            thickness (float, optional): thickness of the slab. Defaults to 1.
            shift (float, optional): Amount by which to shift the holes in the two halves of the crystal up and down. Defaults to 0.
            sw (float, optional): Width of the air slot. Defaults to 0.
            index (float, optional): Square root of permittivity. Defaults to 3.45 (Silicon).
        """
        h = np.sqrt(3)*a    # Height of a unit cell
        cell = mp.Vector3(a, 6*h + 2*shift, 12*thickness)

        # Create the dielectric slab
        b = mp.Block(center=mp.Vector3(0,0,0), size=mp.Vector3(mp.inf,mp.inf, thickness), material=mp.Medium(index=index))
        geometry = [b]

        # Create holes in the slab
        create_slab_holes(r, a, h, shift, coords, geometry)

        # Cover top and bottom holes
        geometry.extend([
            mp.Block(size=mp.Vector3(mp.inf,2*r,thickness), center=mp.Vector3(0,-0.5*cell.y,0), material=mp.Medium(index=index)),
            mp.Block(size=mp.Vector3(mp.inf,2*r,thickness), center=mp.Vector3(0,0.5*cell.y,0), material=mp.Medium(index=index))
        ])

        # Create the air slot
        geometry.append( mp.Block(size=mp.Vector3(mp.inf,sw,mp.inf)) )

        self.geometry = geometry
        self.cell = cell


class SlottedTriangleLatticeCavity:
    def __init__(self,  r: float, a: float=1, thickness: float=1, shift: float=0, sw: float=0, index: float=3.45):
        a_nm = 426
        h = np.sqrt(3)*a    # Height of a unit cell
        cell = mp.Vector3(28*a, 6*h, 12*thickness)


        # Create the dielectric slab
        b = mp.Block(center=mp.Vector3(0,0,0), size=mp.Vector3(mp.inf,mp.inf, thickness), material=mp.Medium(index=index))
        geometry = [b]

        # Create row of SlottedTriangleLattice "unit cells"
        for T in np.arange(1,29) - 14.5:
            create_slab_holes(r, a, h, shift, mp.Vector3(T,0,0), geometry)
        
        # Cover top and bottom holes
        geometry.extend([
            mp.Block(size=mp.Vector3(mp.inf,2*r,thickness), center=mp.Vector3(0,-0.5*cell.y,0), material=mp.Medium(index=index)),
            mp.Block(size=mp.Vector3(mp.inf,2*r,thickness), center=mp.Vector3(0,0.5*cell.y,0), material=mp.Medium(index=index))
        ])


        # Cover cavity holes, and create new shifted holes
        covers = [
            mp.Block(center=mp.Vector3(0,0.6*h), size=mp.Vector3(5*a+2*r,2*r,thickness), material=mp.Medium(index=index)),
            mp.Block(center=mp.Vector3(0,1.1*h), size=mp.Vector3(4*a+2*r,2*r,thickness), material=mp.Medium(index=index)),
            mp.Block(center=mp.Vector3(0,1.6*h), size=mp.Vector3(3*a+2*r,2*r,thickness), material=mp.Medium(index=index)),
            mp.Block(center=mp.Vector3(0,-0.6*h), size=mp.Vector3(5*a+2*r,2*r,thickness), material=mp.Medium(index=index)),
            mp.Block(center=mp.Vector3(0,-1.1*h), size=mp.Vector3(4*a+2*r,2*r,thickness), material=mp.Medium(index=index)),
            mp.Block(center=mp.Vector3(0,-1.6*h), size=mp.Vector3(3*a+2*r,2*r,thickness), material=mp.Medium(index=index))
        ]
        geometry.extend(covers)

        # Create shifted holes
        shift1 = 5/a_nm
        shift2 = 10/a_nm
        shift3 = 15/a_nm
        
        shift1_holes = [
            mp.Cylinder(center=mp.Vector3(0.5*a,1.6*h+shift1), radius=r),
            mp.Cylinder(center=mp.Vector3(1.5*a,1.6*h+shift1), radius=r),
            mp.Cylinder(center=mp.Vector3(2.0*a,1.1*h+shift1), radius=r),
            mp.Cylinder(center=mp.Vector3(2.5*a,0.6*h+shift1), radius=r),

            mp.Cylinder(center=mp.Vector3(0.5*a,-(1.6*h+shift1)), radius=r),
            mp.Cylinder(center=mp.Vector3(1.5*a,-(1.6*h+shift1)), radius=r),
            mp.Cylinder(center=mp.Vector3(2.0*a,-(1.1*h+shift1)), radius=r),
            mp.Cylinder(center=mp.Vector3(2.5*a,-(0.6*h+shift1)), radius=r),

            mp.Cylinder(center=mp.Vector3(-0.5*a,1.6*h+shift1), radius=r),
            mp.Cylinder(center=mp.Vector3(-1.5*a,1.6*h+shift1), radius=r),
            mp.Cylinder(center=mp.Vector3(-2.0*a,1.1*h+shift1), radius=r),
            mp.Cylinder(center=mp.Vector3(-2.5*a,0.6*h+shift1), radius=r),

            mp.Cylinder(center=mp.Vector3(-0.5*a,-(1.6*h+shift1)), radius=r),
            mp.Cylinder(center=mp.Vector3(-1.5*a,-(1.6*h+shift1)), radius=r),
            mp.Cylinder(center=mp.Vector3(-2.0*a,-(1.1*h+shift1)), radius=r),
            mp.Cylinder(center=mp.Vector3(-2.5*a,-(0.6*h+shift1)), radius=r),
        ]

        shift2_holes = [
            mp.Cylinder(center=mp.Vector3(1.0*a,1.1*h+shift2), radius=r),
            mp.Cylinder(center=mp.Vector3(1.5*a,0.6*h+shift2), radius=r),

            mp.Cylinder(center=mp.Vector3(1.0*a,-(1.1*h+shift2)), radius=r),
            mp.Cylinder(center=mp.Vector3(1.5*a,-(0.6*h+shift2)), radius=r),

            mp.Cylinder(center=mp.Vector3(-1.0*a,1.1*h+shift2), radius=r),
            mp.Cylinder(center=mp.Vector3(-1.5*a,0.6*h+shift2), radius=r),

            mp.Cylinder(center=mp.Vector3(-1.0*a,-(1.1*h+shift2)), radius=r),
            mp.Cylinder(center=mp.Vector3(-1.5*a,-(0.6*h+shift2)), radius=r),

            mp.Cylinder(center=mp.Vector3(0.0*a,1.1*h+shift2), radius=r),
            mp.Cylinder(center=mp.Vector3(0.0*a,-(1.1*h+shift2)), radius=r),
        ]

        shift3_holes = [
            mp.Cylinder(center=mp.Vector3(0.5*a,0.6*h+shift3), radius=r),
            mp.Cylinder(center=mp.Vector3(0.5*a,-(0.6*h+shift3)), radius=r),
            mp.Cylinder(center=mp.Vector3(-0.5*a,0.6*h+shift3), radius=r),
            mp.Cylinder(center=mp.Vector3(-0.5*a,-(0.6*h+shift3)), radius=r),
        ]

        geometry.extend(shift1_holes)
        geometry.extend(shift2_holes)
        geometry.extend(shift3_holes)

        # Create the air slot
        geometry.append( mp.Block(size=mp.Vector3(mp.inf,sw,mp.inf)) )


        self.geometry = geometry
        self.cell = cell