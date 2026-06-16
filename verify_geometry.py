"""
verify_geometry.py  --  self-contained sanity checks for geometries.py.

Run with:  python verify_geometry.py
Checks, for both the plain lattice and the cavity:
  * no two cylinders share a centre (dedup actually works);
  * the hole pattern is mirror-symmetric about y = 0 (slot symmetry);
  * the cavity keeps its inner shifted defect holes.
These checks need only MEEP and do not depend on any reference file.
"""

import numpy as np
import meep as mp
from geometries import SlottedTriangleLattice, SlottedTriangleLatticeCavity

A_NM = 426
PARAMS = dict(r=0.245, a=1.0, thickness=220 / A_NM, sw=100 / A_NM,
              index=3.45, width=36)
SHIFT = (1.2 - 1) / 2 * np.sqrt(3)


def centres(domain):
    return [(round(g.center.x, 6), round(g.center.y, 6))
            for g in domain.geometry if isinstance(g, mp.Cylinder)]


def check(name, domain):
    c = centres(domain)
    uniq = set(c)
    no_dups = len(c) == len(uniq)
    mirror = all((x, -y) in uniq for (x, y) in uniq)
    print(f"[{name}] cylinders={len(c)}  unique={len(uniq)}  "
          f"no_duplicates={no_dups}  mirror_symmetric={mirror}")
    assert no_dups, "duplicate cylinders found!"
    assert mirror, "geometry is not symmetric about the slot (y=0)!"
    return uniq


if __name__ == "__main__":
    lat = SlottedTriangleLattice(**PARAMS, shift=SHIFT)
    cav = SlottedTriangleLatticeCavity(**PARAMS, shift=SHIFT)

    check("lattice", lat)
    u_cav = check("cavity ", cav)

    central_cols = {-2.5, -1.5, -0.5, 0.5, 1.5, 2.5}
    defect = {(x, y) for (x, y) in u_cav if x in central_cols and abs(y) < 1.2}
    print(f"[cavity ] innermost shifted defect holes in central columns: "
          f"{len(defect)} (expected non-zero)")
    assert len(defect) > 0, "cavity defect holes missing!"

    print("\nAll geometry sanity checks passed.")