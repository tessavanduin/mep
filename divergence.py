# This code was adapted from chatGPT

import meep as mp

def divE_at_point(sim: mp.Simulation, pt):
    dx = 1 / sim.resolution
    # Offsets for central difference
    o = dx

    # ---- Sample along x-line ----
    Ex_p = sim.get_field_point(mp.Ex, pt + mp.Vector3(o, 0, 0))
    Ex_m = sim.get_field_point(mp.Ex, pt - mp.Vector3(o, 0, 0))
    dEx_dx = (Ex_p - Ex_m) / (2*dx)

    # ---- Sample along y-line ----
    Ey_p = sim.get_field_point(mp.Ey, pt + mp.Vector3(0, o, 0))
    Ey_m = sim.get_field_point(mp.Ey, pt - mp.Vector3(0, o, 0))
    dEy_dy = (Ey_p - Ey_m) / (2*dx)

    # ---- Sample along z-line (3D only) ----
    if sim.dimensions == 3:
        Ez_p = sim.get_field_point(mp.Ez, pt + mp.Vector3(0, 0, o))
        Ez_m = sim.get_field_point(mp.Ez, pt - mp.Vector3(0, 0, o))
        dEz_dz = (Ez_p - Ez_m) / (2*dx)
        return (dEx_dx + dEy_dy + dEz_dz).real

    return (dEx_dx + dEy_dy).real