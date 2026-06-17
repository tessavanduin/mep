# electron_source.py
import meep as mp
import numpy as np


def make_electron_path(x_start, beta, y0, z0):
    """Return a function t -> Vector3 giving the electron position at time t."""
    def electron_path(t):
        return mp.Vector3(x_start + beta * t, y0, z0)
    return electron_path


def pulse_chain(path_len, resolution, x_start, x_end, beta,
                fcen, df, amp0, y0, z0):
    """Return a LIST of fixed Gaussian-pulse sources along the path."""
    npix_src = int(round(path_len * resolution))
    xs_src = np.linspace(x_start, x_end, npix_src)
    sources = []
    for xi in xs_src:
        ti = (xi - x_start) / beta
        sources.append(mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=df,
                              start_time=ti, cutoff=4.0),
            component=mp.Ex,
            center=mp.Vector3(xi, y0, z0),
            amplitude=amp0,
        ))
    return sources


def moving_gaussian(electron_path, amp0, src_width=0.10):
    """Return a step-callback that moves a spatial-Gaussian source each step."""
    src_size = mp.Vector3(1.0, 1.0, 1.0)

    def src_amplitude(r):
        rsq = r.dot(r)
        ssq2 = 2 * src_width * src_width
        return amp0 * np.exp(-rsq / ssq2) / np.sqrt(np.pi * ssq2)

    def move_gauss(sim_obj):
        tnow = sim_obj.meep_time()
        sim_obj.change_sources([mp.Source(
            mp.ContinuousSource(frequency=1e-8),
            component=mp.Ex,
            center=electron_path(tnow),
            size=src_size,
            amp_func=src_amplitude,
        )])
    return move_gauss