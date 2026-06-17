import meep as mp
import numpy as np


class SimulationDomain:
    def __init__(self):
        self.geometry = []
        self.cell = mp.Vector3()


class TriangleUnitCell:
    def __init__(self, r: float, a=1, coords: mp.Vector3 = mp.Vector3(), mask: list = None):
        """
        Unit cell of 4-hole diamond motif.
        """

        if mask is None:
            mask = [True, True, True, True]

        h = np.sqrt(3) * a

        centers = [
            mp.Vector3(0, 0.5 * h, 0),
            mp.Vector3(-0.5 * a, 0, 0),
            mp.Vector3(0.5 * a, 0, 0),
            mp.Vector3(0, -0.5 * h, 0),
        ]

        cylinders = [
            mp.Cylinder(center=c + coords, radius=r)
            for c in centers
        ]

        self.geometry = [c for c, m in zip(cylinders, mask) if m]
        self.cell = mp.Vector3(a, h, 0)


def create_slab_holes(r, a, h, shift, coords):
    geometry = []

    rows = list(np.r_[np.arange(-3, 0), np.arange(1, 4)])

    for i in rows:
        T = i * h + np.sign(i) * shift

        mask = [True, True, True, True]

        if i == -3:
            mask = [False, False, False, True]
        elif i == 3:
            mask = [True, False, False, False]

        geometry.extend(
            TriangleUnitCell(r, a, coords + mp.Vector3(0, T, 0), mask).geometry
        )

    return geometry


class SlottedTriangleLattice:
    def __init__(
        self,
        r: float,
        a: float = 1,
        thickness: float = 1,
        shift: float = 0,
        sw: float = 0,
        index: float = 3.45,
        width: int = 1,
        mask=None,
    ):
        h = np.sqrt(3) * a

        cell = mp.Vector3(width * a, 6 * h + 2 * shift, thickness)
        geometry = [mp.Block(center=mp.Vector3(), size=cell, material=mp.Medium(index=index))]

        if mask is None:
            mask = [True] * width

        x_positions = np.arange(width) - (width - 1) / 2

        for i, x in enumerate(x_positions):
            if not mask[i]:
                continue
            geometry.extend(create_slab_holes(r, a, h, shift, mp.Vector3(x, 0, 0)))

        # air slot
        geometry.append(mp.Block(size=mp.Vector3(mp.inf, sw, mp.inf)))

        self.geometry = geometry
        self.cell = cell


class SlottedTriangleLatticeCavity(SlottedTriangleLattice):
    def __init__(
        self,
        r: float,
        a: float = 1,
        thickness: float = 1,
        shift: float = 0,
        sw: float = 0,
        index: float = 3.45,
        width: int = 28,
    ):
        half_way = (width - 6) // 2
        mask = [True] * half_way + [False] * 6 + [True] * half_way

        super().__init__(r, a, thickness, shift, sw, index, width, mask=mask)

        h = np.sqrt(3) * a

        def C(x, y):
            return mp.Cylinder(center=mp.Vector3(x * a, y, 0), radius=r)

        def add(lst):
            self.geometry.extend(lst)

        # unshifted corrections near cavity
        base = [
            (0.5, 2 * h + shift, [1, 1, 1, 0]),
            (1.5, 2 * h + shift, [1, 1, 1, 0]),
            (2.5, 2 * h + shift, [1, 0, 0, 1]),
            (-0.5, 2 * h + shift, [1, 1, 1, 0]),
            (-1.5, 2 * h + shift, [1, 1, 1, 0]),
            (-2.5, 2 * h + shift, [1, 0, 0, 1]),

            (0.5, -2 * h - shift, [0, 1, 1, 1]),
            (1.5, -2 * h - shift, [0, 1, 1, 1]),
            (2.5, -2 * h - shift, [1, 0, 0, 1]),
            (-0.5, -2 * h - shift, [0, 1, 1, 1]),
            (-1.5, -2 * h - shift, [0, 1, 1, 1]),
            (-2.5, -2 * h - shift, [1, 0, 0, 1]),
        ]

        for x, y, m in base:
            add(TriangleUnitCell(r, a, mp.Vector3(x, y, 0), m).geometry)

        # shifted perturbation holes
        shift1 = 5 / 426
        shift2 = 10 / 426
        shift3 = 15 / 426

        shift1_holes = [
            C(0.5, 1.6 * h + shift1),
            C(1.5, 1.6 * h + shift1),
            C(2.0, 1.1 * h + shift1),
            C(2.5, 0.6 * h + shift1),

            C(0.5, -(1.6 * h + shift1)),
            C(1.5, -(1.6 * h + shift1)),
            C(2.0, -(1.1 * h + shift1)),
            C(2.5, -(0.6 * h + shift1)),

            C(-0.5, 1.6 * h + shift1),
            C(-1.5, 1.6 * h + shift1),
            C(-2.0, 1.1 * h + shift1),
            C(-2.5, 0.6 * h + shift1),

            C(-0.5, -(1.6 * h + shift1)),
            C(-1.5, -(1.6 * h + shift1)),
            C(-2.0, -(1.1 * h + shift1)),
            C(-2.5, -(0.6 * h + shift1)),
        ]

        shift2_holes = [
            C(1.0, 1.1 * h + shift2),
            C(1.5, 0.6 * h + shift2),
            C(1.0, -(1.1 * h + shift2)),
            C(1.5, -(0.6 * h + shift2)),

            C(-1.0, 1.1 * h + shift2),
            C(-1.5, 0.6 * h + shift2),
            C(-1.0, -(1.1 * h + shift2)),
            C(-1.5, -(0.6 * h + shift2)),

            C(0.0, 1.1 * h + shift2),
            C(0.0, -(1.1 * h + shift2)),
        ]

        shift3_holes = [
            C(0.5, 0.6 * h + shift3),
            C(0.5, -(0.6 * h + shift3)),
            C(-0.5, 0.6 * h + shift3),
            C(-0.5, -(0.6 * h + shift3)),
        ]

        self.geometry.extend(shift1_holes + shift2_holes + shift3_holes)