"""
geometries.py  --  slotted triangular-lattice photonic-crystal slab.

Rewrite of the original motif-tiling version.  Functionally identical (the unique
set of hole positions is provably the same -- see verify_geometry.py) but:

  * holes are generated directly on the triangular lattice, so NO duplicate
    cylinders are produced (the old diamond-motif tiling emitted ~30% redundant
    overlapping holes: 648 -> 436 for the lattice, 602 -> 424 for the cavity);
  * every hole goes through one deduplicating accumulator, so the geometry can
    never contain accidental coincident objects;
  * the cavity perturbation is kept as explicit, verbatim data (these holes are
    hand-tuned to create the high-Q mode and must not be "simplified").

Coordinate convention (unchanged):
    x  along the slot / electron path     y  across the slot     z  slab normal
"""

import numpy as np
import meep as mp

ROOT3 = np.sqrt(3.0)


# --------------------------------------------------------------------- helpers
class _HoleSet:
    """Accumulates hole centres, transparently discarding exact duplicates."""

    def __init__(self, tol=1e-6):
        self._tol = tol
        self._keys = set()
        self.centres = []          # list of (x, y)

    def add(self, x, y):
        key = (round(x / self._tol), round(y / self._tol))
        if key not in self._keys:
            self._keys.add(key)
            self.centres.append((x, y))

    def remove(self, x, y):
        key = (round(x / self._tol), round(y / self._tol))
        if key in self._keys:
            self._keys.discard(key)
            self.centres = [c for c in self.centres
                            if (round(c[0] / self._tol), round(c[1] / self._tol)) != key]

    def cylinders(self, r):
        return [mp.Cylinder(center=mp.Vector3(x, y, 0), radius=r) for x, y in self.centres]


def _base_lattice(holes, a, h, shift, columns, skip_columns=()):
    """Fill `holes` with the triangular lattice over the given column centres.

    Two interleaved sub-lattices reproduce the original motif exactly:
      * 'A' rows  (y/h = 0.5, 1.5, 2.5, 3.5)  carry holes at the column centres;
      * 'B' rows  (y/h = 1.0, 2.0)            carry holes at column centre +/- 0.5a.
    The y>0 half is displaced by +shift, the y<0 half by -shift.  The outermost
    A row (3.5h) and the missing 3.0h row reproduce the original boundary.
    """
    A_rows = (0.5, 1.5, 2.5, 3.5)      # holes at x = column centre
    B_rows = (1.0, 2.0)                # holes at x = column centre +/- 0.5a
    skip = set(np.round(np.asarray(skip_columns, float), 6))

    for c in columns:
        if round(float(c), 6) in skip:
            continue
        for ry in A_rows:
            holes.add(c, ry * h + shift)
            holes.add(c, -(ry * h + shift))
        for ry in B_rows:
            for dx in (-0.5 * a, 0.5 * a):
                holes.add(c + dx, ry * h + shift)
                holes.add(c + dx, -(ry * h + shift))


# --------------------------------------------------------------------- classes
class SlottedTriangleLattice:
    def __init__(self, r, a=1, thickness=1, shift=0, sw=0,
                 index=3.45, width=1, mask=None):
        h = ROOT3 * a
        cell = mp.Vector3(width * a, 6 * h + 2 * shift, thickness)

        columns = np.arange(width) - (width - 1) / 2          # column centres in x
        if mask is not None:
            columns = columns[np.asarray(mask, bool)]

        holes = _HoleSet()
        _base_lattice(holes, a, h, shift, columns)
        self._holes = holes

        self.geometry = (
            [mp.Block(center=mp.Vector3(), size=cell, material=mp.Medium(index=index))]
            + holes.cylinders(r)
            + [mp.Block(size=mp.Vector3(mp.inf, sw, mp.inf))]   # air slot (last = wins)
        )
        self.cell = cell


class SlottedTriangleLatticeCavity(SlottedTriangleLattice):
    """Width-modulated (Kuramochi-style) cavity: the six central columns are left
    out of the regular lattice and replaced by a hand-tuned set of unshifted
    corrections plus three groups of inward-shifted holes that form the high-Q
    even cavity mode.  These positions are reproduced verbatim from the original
    design and must not be altered."""

    # (x [units of a], y_over_h, sign)  unshifted corrections near the cavity
    # expressed as the holes the original TriangleUnitCell masks actually emit.
    def __init__(self, r, a=1, thickness=1, shift=0, sw=0, index=3.45, width=28):
        h = ROOT3 * a
        cell = mp.Vector3(width * a, 6 * h + 2 * shift, thickness)

        columns = np.arange(width) - (width - 1) / 2
        half = (width - 6) // 2
        skip = columns[half:half + 6]                          # six central columns

        holes = _HoleSet()
        _base_lattice(holes, a, h, shift, columns, skip_columns=skip)

        # ---- unshifted corrections (verbatim from the original masks) --------
        # original used TriangleUnitCell(r,a,(x,y),mask) at y = +-(2h+shift);
        # we expand each mask to the explicit holes it produces.
        def cell_holes(xc, yc, mask):
            pts = [(0.0, 0.5 * h), (-0.5 * a, 0.0), (0.5 * a, 0.0), (0.0, -0.5 * h)]
            for (dx, dy), m in zip(pts, mask):
                if m:
                    holes.add(xc + dx, yc + dy)

        for x in (0.5, 1.5, -0.5, -1.5):
            cell_holes(x, 2 * h + shift, (1, 1, 1, 0))
            cell_holes(x, -2 * h - shift, (0, 1, 1, 1))
        for x in (2.5, -2.5):
            cell_holes(x, 2 * h + shift, (1, 0, 0, 1))
            cell_holes(x, -2 * h - shift, (1, 0, 0, 1))

        # ---- inward-shifted perturbation holes (verbatim) --------------------
        s1, s2, s3 = 5 / 426, 10 / 426, 15 / 426

        shift1 = [(0.5, 1.6 * h + s1), (1.5, 1.6 * h + s1), (2.0, 1.1 * h + s1), (2.5, 0.6 * h + s1),
                  (-0.5, 1.6 * h + s1), (-1.5, 1.6 * h + s1), (-2.0, 1.1 * h + s1), (-2.5, 0.6 * h + s1)]
        shift2 = [(1.0, 1.1 * h + s2), (1.5, 0.6 * h + s2), (-1.0, 1.1 * h + s2), (-1.5, 0.6 * h + s2),
                  (0.0, 1.1 * h + s2)]
        shift3 = [(0.5, 0.6 * h + s3), (-0.5, 0.6 * h + s3)]

        for x, y in shift1 + shift2 + shift3:        # add both halves (y and -y)
            holes.add(x * a, y)
            holes.add(x * a, -y)

        self._holes = holes
        self.geometry = (
            [mp.Block(center=mp.Vector3(), size=cell, material=mp.Medium(index=index))]
            + holes.cylinders(r)
            + [mp.Block(size=mp.Vector3(mp.inf, sw, mp.inf))]
        )
        self.cell = cell