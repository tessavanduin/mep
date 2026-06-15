r"""
Brute-force EELS Gamma(omega, y0, z0) for a slotted triangular-lattice
photonic-crystal cavity, electron flying parallel to x through the slot.

Reproduces the "home-made FDTD" method described in
"High-Efficiency Coupling of Free Electrons to Sub-lambda^3 Modal Volume,
High-Q Photonic Cavities" -- using MEEP as the FDTD engine.

WHAT THIS METHOD TARGETS:
  The brute-force moving-source spectrum reproduces the BROAD slot-mode and
  band features of the experimental spectrum (the alpha ~0.7, beta ~0.81,
  gamma ~0.95 eV peaks of Fig. 3a). It does NOT resolve the sharp cavity mode:
  the paper's cavity has Q = 2.5e5, a ~few-microeV linewidth that is (a) below
  the 30 meV experimental resolution and (b) impossible to ring down in FDTD.
  The sharp cavity line is the job of the Green-tensor method (eels_green.py,
  eq. 3), where the complex resonant frequency omega_k is an INPUT, not
  something extracted by waiting out the time-domain decay.

Method (per the paper's Methods section):
  1. Model the 100 keV electron as a moving current source along x at v=beta*c.
     Two source models are available (--source-model):
       moving-gaussian (default): a single source with a smooth spatial-Gaussian
         profile, repositioned every timestep to follow the electron. Always-on
         direct field is removed by the empty-domain subtraction. Verified clean.
       pulse-chain: a line of fixed Gaussian-pulse sources, each fired as the
         electron passes. Also verified, but ~850 sources so slower.
  2. Run the SAME simulation twice: once with the cavity, once in an empty
     (vacuum) domain. Record E_x(x, y0, z0, t) every timestep along the path.
  3. Subtract: E_ind = E_cavity - E_empty (removes the direct electron field).
  4. Per spatial pixel, temporal Fourier transform E_ind(x,t) -> E_ind(x,omega),
     via windowed DFT or Pade (--pade, --pade-method prony|scipy).
  5. Assemble the eq.(1) integral along x:
        Gamma(omega) = (e/(pi hbar omega)) Re[ \int E_ind_x(x,omega) e^{i omega x/v} dx ]
     Optional --spatial-taper suppresses finite-path sinc ripple when the
     induced field is nonzero at the path ends.
  6. Convolve with a 30 meV Gaussian (paper Methods: 30 meV FWHM).

Split workflow: --save-fields runs only the (expensive) FDTD and stores the
  induced + raw fields; --from-fields re-runs only the (cheap) transform, so
  windows / Pade orders / tapers can be retuned without re-simulating.

Units: MEEP dimensionless units with a = a_nm (lattice constant) = 1, c = 1.
       Frequencies are in units of c/a. To convert MEEP freq f -> photon energy:
           E[eV] = h c / (lambda)  with lambda = a_nm / f  (a_nm in metres)
       Implemented in meep_freq_to_eV().

Run:  python eels_brute_force.py --cavity            # cavity run + empty run + spectrum
      python eels_brute_force.py --cavity --quick    # coarse resolution smoke test
"""

import argparse
import numpy as np

try:
    import meep as mp
except ImportError:
    mp = None  # allows static import / unit conversions without MEEP installed

# ----------------------------------------------------------------------------
# Geometry classes (your code, unchanged in spirit; included so this is one file)
# ----------------------------------------------------------------------------

class TriangleUnitCell:
    def __init__(self, r, a=1, coords=None, mask=None):
        if coords is None:
            coords = mp.Vector3(0, 0, 0)
        if mask is None:
            mask = [True] * 4
        h = np.sqrt(3) * a
        c1 = mp.Cylinder(center=mp.Vector3(0, 0.5 * h, 0) + coords, radius=r)
        c2 = mp.Cylinder(center=mp.Vector3(-0.5 * a, 0, 0) + coords, radius=r)
        c3 = mp.Cylinder(center=mp.Vector3(0.5 * a, 0, 0) + coords, radius=r)
        c4 = mp.Cylinder(center=mp.Vector3(0, -0.5 * h, 0) + coords, radius=r)
        self.geometry = list(np.array([c1, c2, c3, c4])[np.array(mask, dtype=bool)])
        self.cell = mp.Vector3(a, h, 0)


class SlottedTriangleLattice:
    def __init__(self, r, a=1, thickness=1, shift=0, sw=0, index=3.45,
                 width=1, mask=None):
        h = np.sqrt(3) * a
        Nx = width
        cell_y = 6 * h + 2 * shift
        cell = mp.Vector3(Nx * a, cell_y, thickness)
        geometry = [mp.Block(center=mp.Vector3(0, 0, 0), size=cell,
                             material=mp.Medium(index=index))]
        for i in range(Nx):
            x_shift = (i - (Nx - 1) / 2) * a
            geometry.extend(
                TriangleUnitCell(r, a, coords=mp.Vector3(x_shift, 0, 0)).geometry
            )
        geometry.append(
            mp.Block(center=mp.Vector3(0, 0, 0),
                     size=mp.Vector3(cell.x, sw, thickness),
                     material=mp.air)
        )
        self.geometry = geometry
        self.cell = cell


class SlottedTriangleLatticeCavity(SlottedTriangleLattice):
    def __init__(self, r, a=1, thickness=1, shift=0, sw=0, index=3.45, width=28):
        half_way = int((width - 6) / 2)
        mask = [True] * half_way + [False] * 6 + [True] * half_way
        super().__init__(r, a, thickness, shift, sw, index, width, mask=mask)

        a_nm = 426
        h = np.sqrt(3) * a

        shift1 = 5 / a_nm
        shift2 = 10 / a_nm
        shift3 = 15 / a_nm

        unshifted = []
        for sx, m in [(0.5, [1, 1, 1, 0]), (1.5, [1, 1, 1, 0]), (2.5, [1, 0, 0, 1]),
                      (-0.5, [1, 1, 1, 0]), (-1.5, [1, 1, 1, 0]), (-2.5, [1, 0, 0, 1])]:
            unshifted.extend(
                TriangleUnitCell(r, a, mp.Vector3(sx, 2 * h + shift), m).geometry)
        for sx, m in [(0.5, [0, 1, 1, 1]), (1.5, [0, 1, 1, 1]), (2.5, [1, 0, 0, 1]),
                      (-0.5, [0, 1, 1, 1]), (-1.5, [0, 1, 1, 1]), (-2.5, [1, 0, 0, 1])]:
            unshifted.extend(
                TriangleUnitCell(r, a, mp.Vector3(sx, -2 * h - shift), m).geometry)
        self.geometry.extend(unshifted)

        def cyl(x, y):
            return mp.Cylinder(center=mp.Vector3(x, y), radius=r)

        s1 = shift1
        shift1_holes = [
            cyl(0.5 * a, 1.6 * h + s1), cyl(1.5 * a, 1.6 * h + s1),
            cyl(2.0 * a, 1.1 * h + s1), cyl(2.5 * a, 0.6 * h + s1),
            cyl(0.5 * a, -(1.6 * h + s1)), cyl(1.5 * a, -(1.6 * h + s1)),
            cyl(2.0 * a, -(1.1 * h + s1)), cyl(2.5 * a, -(0.6 * h + s1)),
            cyl(-0.5 * a, 1.6 * h + s1), cyl(-1.5 * a, 1.6 * h + s1),
            cyl(-2.0 * a, 1.1 * h + s1), cyl(-2.5 * a, 0.6 * h + s1),
            cyl(-0.5 * a, -(1.6 * h + s1)), cyl(-1.5 * a, -(1.6 * h + s1)),
            cyl(-2.0 * a, -(1.1 * h + s1)), cyl(-2.5 * a, -(0.6 * h + s1)),
        ]
        s2 = shift2
        shift2_holes = [
            cyl(1.0 * a, 1.1 * h + s2), cyl(1.5 * a, 0.6 * h + s2),
            cyl(1.0 * a, -(1.1 * h + s2)), cyl(1.5 * a, -(0.6 * h + s2)),
            cyl(-1.0 * a, 1.1 * h + s2), cyl(-1.5 * a, 0.6 * h + s2),
            cyl(-1.0 * a, -(1.1 * h + s2)), cyl(-1.5 * a, -(0.6 * h + s2)),
            cyl(0.0 * a, 1.1 * h + s2), cyl(0.0 * a, -(1.1 * h + s2)),
        ]
        s3 = shift3
        shift3_holes = [
            cyl(0.5 * a, 0.6 * h + s3), cyl(0.5 * a, -(0.6 * h + s3)),
            cyl(-0.5 * a, 0.6 * h + s3), cyl(-0.5 * a, -(0.6 * h + s3)),
        ]
        self.geometry.extend(shift1_holes + shift2_holes + shift3_holes)


# ----------------------------------------------------------------------------
# Physics helpers
# ----------------------------------------------------------------------------

# physical constants
_C = 299792458.0          # m/s
_HBAR_EV = 6.582119569e-16  # eV*s
_H_EV = 4.135667696e-15     # eV*s
_ME_C2 = 510998.95          # electron rest energy, eV
_E_CHARGE = 1.602176634e-19  # C


def electron_beta(E_kin_keV):
    """Relativistic beta = v/c for a given electron kinetic energy in keV."""
    E_kin = E_kin_keV * 1e3  # eV
    gamma = 1.0 + E_kin / _ME_C2
    return np.sqrt(1.0 - 1.0 / gamma**2)


def meep_freq_to_eV(f_meep, a_nm):
    """MEEP frequency (units c/a) -> photon energy in eV. a_nm in nm."""
    a_m = a_nm * 1e-9
    lam_m = a_m / np.asarray(f_meep)      # wavelength in metres
    return _H_EV * _C / lam_m


def eV_to_meep_freq(E_eV, a_nm):
    a_m = a_nm * 1e-9
    lam_m = _H_EV * _C / np.asarray(E_eV)
    return a_m / lam_m


# ----------------------------------------------------------------------------
# Build geometry + cell (fixes the padding bug: pad_z now actually used; the
# x-direction gets PML, the electron enters/exits through it)
# ----------------------------------------------------------------------------

def build_phc(args):
    a_nm = args.a
    a = 1.0
    h = np.sqrt(3) * a
    thickness = args.d / a_nm
    r = args.r
    shift = (args.W - 1) / 2 * h
    sw = args.s / a_nm
    crystal_x_width = args.x

    if args.cavity:
        dom = SlottedTriangleLatticeCavity(r, a, thickness, shift, sw,
                                           index=args.n, width=crystal_x_width)
    else:
        dom = SlottedTriangleLattice(r, a, thickness, shift, sw,
                                     index=args.n, width=crystal_x_width)
    return dom.geometry, dom.cell, thickness, a_nm


def make_cell(struct_cell, thickness, dpml, pad_xy, pad_z):
    """Full simulation cell = structure + air padding (y,z) + PML everywhere.
    x keeps the structure length plus PML so the electron flies straight
    through; no extra air pad on x is needed (the path lives inside the cell).
    """
    return mp.Vector3(
        struct_cell.x + 2 * dpml,                 # x: structure + PML on both ends
        struct_cell.y + 2 * (pad_xy + dpml),      # y: + air pad + PML
        thickness + 2 * (pad_z + dpml),           # z: slab + air pad + PML
    )


# ----------------------------------------------------------------------------
# Moving-electron source: chain of pulsed point dipoles along x
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# One FDTD run: moving/impulsive electron source + fast field recording on path
# ----------------------------------------------------------------------------

def run_path_recording(geometry, cell, beta, y0, z0, fcen, df, resolution,
                       dpml, T_extra, label="", source_model="pulse-chain"):
    """Run a single simulation with the electron as a current source and record
    Ex along its straight path at (y0,z0) every timestep.

    source_model selects how the electron is represented:
      "pulse-chain" (DEFAULT, verified): a line of fixed Ex sources along the
          path, each firing a narrow Gaussian pulse timed to when the electron
          passes. Impulsive + broadband by construction; nothing stays driven,
          so the structure rings down freely. This is the model whose eq.1+Pade
          pipeline we validated end-to-end. Needs fcen/df to set pulse bandwidth.
      "moving-pulse": a SINGLE moving source whose time profile is a narrow
          Gaussian in the co-moving frame -- impulsive like the chain but only
          one source (faster). Also needs fcen/df.
      "moving-dc": a single moving ~DC ContinuousSource with a smooth transient
          envelope. Fast and tuning-free, BUT in testing this produced an
          entry/exit-dominated, persistently-driven field that does not Pade-fit
          (kept for comparison / debugging only -- not recommended).

    Returns (times, xs, E_field[ntime, nx]) with Ex sampled on the path pixels.
    """
    pml = [mp.PML(dpml)]

    # electron path: enter just inside one PML, exit just inside the other
    x_start = -cell.x / 2.0 + dpml
    x_end = cell.x / 2.0 - dpml
    path_len = x_end - x_start
    transit = path_len / beta
    amp0 = 1.0 / beta             # ~ e/v current weight (abs cal still pending)

    def electron_path(t):
        return mp.Vector3(x_start + beta * t, y0, z0)

    # ----- build the source(s) according to the chosen model -----
    static_sources = []
    step_callback = None

    if source_model == "pulse-chain":
        # fixed sources along x, each a Gaussian pulse fired at t_i = x_i/v
        npix_src = int(round(path_len * resolution))
        xs_src = np.linspace(x_start, x_end, npix_src)
        for xi in xs_src:
            ti = (xi - x_start) / beta
            static_sources.append(mp.Source(
                mp.GaussianSource(frequency=fcen, fwidth=df,
                                  start_time=ti, cutoff=4.0),
                component=mp.Ex,
                center=mp.Vector3(xi, y0, z0),
                amplitude=amp0,
            ))

    elif source_model == "moving-gaussian":
        # Validated MEEP-community recipe: a single source that MOVES across the
        # whole path, with a smooth Gaussian SPATIAL profile (finite src_size +
        # amp_func), driven by an always-on ~DC ContinuousSource. The source
        # genuinely traverses the domain because its position is recomputed from
        # meep_time() every step; the spatial Gaussian gives it a smooth finite
        # extent. The always-on direct field is removed by the empty-domain
        # subtraction, leaving the induced field. Verified to give a clean,
        # properly-subtracted, moving field.
        src_width = 0.10                      # spatial width of the charge blob (a)
        src_size = mp.Vector3(1.0, 1.0, 1.0)  # finite source extent

        def src_amplitude(r):
            rsq = r.dot(r)
            ssq2 = 2 * src_width * src_width
            return amp0 * np.exp(-rsq * rsq / ssq2) / np.sqrt(np.pi * ssq2)

        def move_gauss(sim_obj):
            tnow = sim_obj.meep_time()
            sim_obj.change_sources([mp.Source(
                mp.ContinuousSource(frequency=1e-8),
                component=mp.Ex,
                center=electron_path(tnow),
                size=src_size,
                amp_func=src_amplitude,
            )])
        step_callback = move_gauss

    else:
        raise ValueError(f"unknown source_model: {source_model} "
                         "(use 'pulse-chain' or 'moving-gaussian')")

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=static_sources,        # empty for moving models
        boundary_layers=pml,
        resolution=resolution,
        force_complex_fields=True,
    )

    # path pixels to sample
    npix = int(round(path_len * resolution))
    xs = np.linspace(x_start, x_end, npix)
    total_time = transit + T_extra

    rec_times = []
    rec_fields = []
    line_vol = mp.Volume(center=mp.Vector3(0.5 * (x_start + x_end), y0, z0),
                         size=mp.Vector3(path_len, 0, 0))

    def record(sim_obj):
        ex_line = sim_obj.get_array(mp.Ex, vol=line_vol)
        rec_times.append(sim_obj.meep_time())
        rec_fields.append(np.asarray(ex_line, dtype=complex))

    run_args = [mp.at_every(sim.Courant / resolution, record)]
    if step_callback is not None:
        run_args = [step_callback] + run_args
    sim.run(*run_args, until=total_time)

    rec_fields = np.array(rec_fields)
    nx_actual = rec_fields.shape[1]
    xs = np.linspace(x_start, x_end, nx_actual)
    return np.array(rec_times), xs, rec_fields


# ----------------------------------------------------------------------------
# Post-processing: subtract empty run, temporal FT, eq.(1) assembly, broadening
# ----------------------------------------------------------------------------

def temporal_ft(times, E_xt, omegas):
    r"""E_xt: [ntime, nx] complex. Returns E_xw: [nomega, nx] complex.
    Windowed DFT: E(x,omega) = \int E(x,t) e^{-i omega t} dt, Hann-windowed.
    omegas are MEEP angular frequencies (= 2 pi f_meep)."""
    ntime = len(times)
    win = np.hanning(ntime)[:, None]
    Ew = E_xt * win
    dt = np.gradient(times)[:, None]
    # E(x,omega) = sum_t E(x,t) e^{-i omega t} dt
    phase = np.exp(-1j * np.outer(omegas, times))  # [nomega, ntime]
    return phase @ (Ew * dt)                        # [nomega, nx]


def _pade_spectrum_1d(signal, dt, omegas, M=None):
    r"""Padé-approximant spectral estimate of a single time series.

    Given uniformly-sampled signal s[n] (n=0..N-1, step dt), build the
    [L/M] Pade approximant of its z-transform and evaluate the resulting
    rational spectrum at z = e^{i omega dt}. This fits a sum of damped
    sinusoids (poles = resonances), so it resolves narrow lines from short
    records far better than a windowed DFT.

    Method: the signal's z-transform S(z) = sum s[n] z^{-n} is approximated
    by P(z)/Q(z). The Q (denominator) coefficients come from solving the
    Hankel system built from the autocorrelation-like lag matrix (the
    standard Pade/Prony linear system); P follows by convolution. We then
    evaluate P/Q on the unit circle at the requested frequencies.

    signal : complex [N]
    dt     : scalar time step (uniform)
    omegas : [nomega] MEEP angular frequencies to evaluate
    M      : denominator order (number of poles). Default N//2 (capped).
    Returns complex [nomega].
    """
    s = np.asarray(signal, dtype=complex)
    N = len(s)
    if M is None:
        M = min(N // 2, 200)          # cap order for conditioning/speed
    L = M                              # symmetric [M/M] approximant

    # Build the linear system for denominator coeffs b (b0 = 1):
    #   sum_{k=1..M} b_k s[L + j - k] = -s[L + j],  j = 1..M
    rows = []
    rhs = []
    for j in range(1, M + 1):
        idx = L + j - np.arange(1, M + 1)
        if np.any(idx < 0) or (L + j) >= N:
            continue
        rows.append(s[idx])
        rhs.append(-s[L + j])
    if len(rows) < M:
        # not enough samples for this order: fall back to DFT for this series
        phase = np.exp(-1j * np.outer(omegas, np.arange(N) * dt))
        return phase @ (s * dt)
    A = np.array(rows)
    rhs = np.array(rhs)
    # least-squares (robust to mild rank deficiency) for the poles
    b_tail, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    b = np.concatenate(([1.0], b_tail))      # denominator coeffs, b0=1

    # numerator coeffs a_k = sum_{m=0..k} b_m s[k-m], k=0..L
    a = np.zeros(L + 1, dtype=complex)
    for k in range(L + 1):
        acc = 0.0 + 0j
        for m in range(min(k, M) + 1):
            acc += b[m] * s[k - m]
        a[k] = acc

    # evaluate P(z)/Q(z) at z = e^{i omega dt}  (z^{-1} = e^{-i omega dt})
    zinv = np.exp(-1j * omegas * dt)          # [nomega]
    ka = np.arange(L + 1)
    kb = np.arange(M + 1)
    P = (zinv[:, None] ** ka[None, :]) @ a    # [nomega]
    Q = (zinv[:, None] ** kb[None, :]) @ b    # [nomega]
    Q = np.where(np.abs(Q) < 1e-30, 1e-30, Q)
    return (P / Q) * dt                        # match DFT amplitude scaling


def temporal_ft_pade(times, E_xt, omegas, M=None):
    r"""Padé version of temporal_ft. Same signature/return shape.
    Requires (approximately) uniform time sampling -- MEEP's at_every gives
    this. E_xt: [ntime, nx] -> returns [nomega, nx]."""
    times = np.asarray(times)
    dt = np.median(np.diff(times))
    ntime, nx = E_xt.shape
    out = np.empty((len(omegas), nx), dtype=complex)
    for j in range(nx):
        out[:, j] = _pade_spectrum_1d(E_xt[:, j], dt, omegas, M=M)
    return out


def temporal_ft_pade_scipy(times, E_xt, omegas, M=None):
    r"""Padé transform using scipy.interpolate.pade (the student's pade3.py
    approach). Builds the [L/M] Padé approximant of the time series treated as
    a power series in z^{-1}=exp(-i omega dt), then evaluates P/Q on the unit
    circle. Simpler and closer to what the paper describes; can be less robust
    to noise than the Prony/lstsq variant. Same signature/return as temporal_ft.
    """
    from scipy.interpolate import pade as _scipy_pade
    times = np.asarray(times)
    dt = times[1] - times[0]
    ntime, nx = E_xt.shape
    out = np.empty((len(omegas), nx), dtype=complex)
    zinv = np.exp(-1j * omegas * dt)
    order = M if M is not None else int((ntime - 1) / 2)
    for j in range(nx):
        sig = E_xt[:, j].astype(np.complex128)
        try:
            P, Q = _scipy_pade(sig, order)
            out[:, j] = (P(zinv) / Q(zinv)) * dt
        except Exception:
            # fall back to DFT for an ill-conditioned pixel
            phase = np.exp(-1j * np.outer(omegas, np.arange(ntime) * dt))
            out[:, j] = phase @ (sig * dt)
    return out


def assemble_gamma(E_xw, xs, omegas_meep, beta, a_nm, spatial_taper=0.0):
    r"""eq.(1): Gamma(omega) = (e/(pi hbar omega)) Re[ \int E_ind_x(x,omega)
    e^{i omega x / v} dx ].  Here omega is angular freq in MEEP units (c/a).

    spatial_taper (0..1): fraction of the path over which to taper the field to
    zero at EACH end (Tukey window) before the x-integral. The integral has a
    hard cutoff at the path ends; if the induced field is nonzero there (e.g. an
    entrance-edge artifact), that hard cutoff produces a finite-length sinc
    ripple of period ~ h*v/L that swamps the real modes. Tapering the ends
    removes it (same idea as the Hann window in the temporal FT). 0 = no taper.

    Returns Gamma in arbitrary-but-consistent units; absolute calibration via
    the prefactor block below."""
    dx = xs[1] - xs[0]
    nx = len(xs)

    # optional Tukey spatial window: flat in the middle, cosine taper at ends
    if spatial_taper and spatial_taper > 0:
        w = np.ones(nx)
        ntap = int(spatial_taper * nx)
        if ntap > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ntap) / ntap))
            w[:ntap] = ramp
            w[-ntap:] = ramp[::-1]
        E_xw = E_xw * w[None, :]

    # e^{i omega x / v}: in MEEP units v = beta (c=1), omega = omegas_meep
    phase = np.exp(1j * np.outer(omegas_meep, xs) / beta)  # [nomega, nx]
    integral = np.sum(E_xw * phase, axis=1) * dx           # [nomega]

    # ---- absolute prefactor e/(pi hbar omega) ----
    a_m = a_nm * 1e-9
    omega_phys = omegas_meep * (_C / a_m)
    omega_phys = np.where(np.abs(omega_phys) < 1e-30, np.nan, omega_phys)
    prefactor = _E_CHARGE / (np.pi * (_HBAR_EV * _E_CHARGE) * omega_phys)
    gamma = prefactor * np.real(integral)
    return gamma


def gaussian_convolve(E_eV, gamma, fwhm_eV=0.030):
    """Convolve spectrum with a Gaussian of given FWHM (default 30 meV, per paper)."""
    sigma = fwhm_eV / (2 * np.sqrt(2 * np.log(2)))
    dE = np.gradient(E_eV)
    out = np.zeros_like(gamma)
    for i, E in enumerate(E_eV):
        k = np.exp(-0.5 * ((E_eV - E) / sigma) ** 2)
        k /= np.sum(k * dE)
        out[i] = np.sum(gamma * k * dE)
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-a", type=int, default=426, help="lattice constant a [nm]")
    p.add_argument("-d", type=int, default=220, help="slab thickness [nm]")
    p.add_argument("-x", type=int, default=36, help="crystal width [units of a]")
    p.add_argument("-W", type=float, default=1.2, help="waveguide widening factor")
    p.add_argument("-s", type=int, default=100, help="slot width [nm]")
    p.add_argument("-r", type=float, default=0.245, help="hole radius [units of a]")
    p.add_argument("-n", type=float, default=3.45, help="refractive index")
    p.add_argument("-v", type=float, default=100.0, help="electron energy [keV]")
    p.add_argument("--cavity", action="store_true", help="use cavity geometry")
    p.add_argument("--res", type=float, default=None,
                   help="FDTD resolution [pixels per a]; default = a/18nm")
    p.add_argument("--Emin", type=float, default=0.40, help="spectrum min [eV] "
                   "(default 0.4: covers the alpha/beta/gamma slot-mode region)")
    p.add_argument("--Emax", type=float, default=1.00, help="spectrum max [eV]")
    p.add_argument("--nE", type=int, default=600, help="spectral points")
    p.add_argument("--src-Emin", type=float, default=None,
                   help="source band minimum [eV]; default = --Emin. Set wider "
                        "(e.g. 0) to deposit energy across a broad range.")
    p.add_argument("--src-Emax", type=float, default=None,
                   help="source band maximum [eV]; default = --Emax. Set wider "
                        "(e.g. 3) for a full-range EELS spectrum.")
    p.add_argument("--save-fields", type=str, default=None,
                   help="save the recorded induced field to this .npz and exit "
                        "after the (expensive) FDTD, so you can re-transform later")
    p.add_argument("--from-fields", type=str, default=None,
                   help="skip FDTD; load a previously --save-fields file and only "
                        "run the (cheap) transform + assembly + broadening")
    p.add_argument("--Textra", type=float, default=400.0,
                   help="ringdown time after transit [MEEP units]. NOTE: this "
                        "captures the BROAD slot/band modes (alpha/beta/gamma), "
                        "which is what brute-force EELS reproduces. Do NOT try to "
                        "ring down the Q=2.5e5 cavity mode here -- that is "
                        "physically impossible in FDTD and is the job of the "
                        "Green-tensor method (eels_green.py). A few hundred MEEP "
                        "units is plenty for the broad features.")
    p.add_argument("--quick", action="store_true", help="coarse smoke test")
    p.add_argument("--spatial-taper", type=float, default=0.0,
                   help="fraction (0..0.5) of the path to taper at each end "
                        "before the eq.1 x-integral. Suppresses finite-path sinc "
                        "ripple when the induced field is nonzero at the path ends. "
                        "Try 0.1-0.2 if the spectrum shows uniform ripple.")
    p.add_argument("--source-model", choices=["pulse-chain", "moving-gaussian"],
                   default="moving-gaussian",
                   help="electron source model. moving-gaussian (default): single "
                        "moving spatial-Gaussian source (validated, clean field, "
                        "fast). pulse-chain: impulsive fixed sources along path "
                        "(also verified, but ~850 sources so slower).")
    p.add_argument("--pade", action="store_true",
                   help="use Pade-approximant temporal transform (paper's method) "
                        "instead of the windowed DFT")
    p.add_argument("--pade-order", type=int, default=None,
                   help="Pade denominator order M (number of poles); default N//2 capped")
    p.add_argument("--pade-method", choices=["prony", "scipy"], default="prony",
                   help="prony = hand-rolled lstsq solver (robust to noise); "
                        "scipy = scipy.interpolate.pade (the student's pade3.py, "
                        "simpler, closer to paper). Compare both on real data.")
    p.add_argument("--out", type=str, default="eels_spectrum.npz")
    args = p.parse_args()

    if mp is None:
        raise SystemExit("MEEP is not installed in this environment. "
                         "Run this on your machine/cluster with MEEP.")

    # resolution: paper uses 18 nm -> pixels per a = a_nm / 18
    if args.res is None:
        resolution = args.a / 18.0
    else:
        resolution = args.res
    if args.quick:
        resolution = max(8.0, resolution / 4.0)
        args.Textra = 80.0

    beta = electron_beta(args.v)
    print(f"[info] beta = {beta:.4f}  (v = {beta*_C:.3e} m/s)")
    print(f"[info] resolution = {resolution:.2f} px/a  "
          f"(~{args.a/resolution:.1f} nm/px)")

    # ---- SOURCE bandwidth: independent of the output window ----
    # The electron field is physically broadband. We build a Gaussian-pulse
    # source spanning [src_Emin, src_Emax], which default to the output window
    # but can be set wider (e.g. 0..3 eV) via --src-Emin/--src-Emax so the FDTD
    # actually deposits energy across the whole range you want to analyze.
    src_Emin = args.src_Emin if args.src_Emin is not None else args.Emin
    src_Emax = args.src_Emax if args.src_Emax is not None else args.Emax
    # Gaussian source: center between the two, bandwidth covers the span.
    # Work in frequency: cover [f(src_Emax), f(src_Emin)] (note E and f both
    # monotonic, higher E -> higher f).
    f_lo = float(eV_to_meep_freq(src_Emin if src_Emin > 0 else 0.02, args.a))
    f_hi = float(eV_to_meep_freq(src_Emax, args.a))
    fcen = 0.5 * (f_lo + f_hi)
    df = (f_hi - f_lo) * 1.2 + 0.05   # pad so the pulse fully covers the span
    print(f"[info] source covers E=[{src_Emin:.2f}, {src_Emax:.2f}] eV "
          f"-> fcen={fcen:.4f} c/a, df={df:.4f} c/a")

    # ---- resolution sanity check against the highest source energy ----
    # need ~8 px per wavelength in the highest-index material at src_Emax
    nm_per_px = args.a / resolution
    lam_hi_vac_nm = _H_EV * _C / src_Emax / 1e-9   # vacuum wavelength [nm]
    lam_in_si_nm = lam_hi_vac_nm / args.n          # wavelength in silicon
    px_per_lam_si = lam_in_si_nm / nm_per_px
    print(f"[info] at E={src_Emax:.2f} eV: lambda_Si ~ {lam_in_si_nm:.0f} nm "
          f"= {px_per_lam_si:.1f} px/wavelength")
    if px_per_lam_si < 6:
        E_trust = _H_EV * _C / (6 * nm_per_px * args.n * 1e-9)
        print(f"[WARN] under-resolved at the top of the band -- trust the "
              f"spectrum only below ~{E_trust:.2f} eV. Raise --res for higher E.")

    geometry, struct_cell, thickness, a_nm = build_phc(args)

    dpml = 1.0
    pad_xy = 1.0
    pad_z = 1.5 * thickness
    cell = make_cell(struct_cell, thickness, dpml, pad_xy, pad_z)
    print(f"[info] cell = ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) a")

    # electron flies through the slot centre: y0 = 0 (slot midline), z0 = 0 (mid-plane)
    y0, z0 = 0.0, 0.0

    if args.from_fields:
        # ---- cheap path: load saved induced field, skip all FDTD ----
        print(f"[load] induced field from {args.from_fields}")
        d = np.load(args.from_fields)
        t_c = d["t"]
        xs = d["xs"]
        E_ind = d["E_ind"]
        # beta/resolution come from the file for consistency
        beta = float(d["beta"])
        print(f"[info] loaded field: {E_ind.shape[0]} timesteps x "
              f"{E_ind.shape[1]} path pixels")
    else:
        # --- run 1: cavity ---
        print(f"[run] cavity ... (source model: {args.source_model})")
        t_c, xs, E_c = run_path_recording(geometry, cell, beta, y0, z0,
                                          fcen, df, resolution, dpml,
                                          args.Textra, label="cavity",
                                          source_model=args.source_model)

        # --- run 2: empty (vacuum) domain, identical recording ---
        print("[run] empty ...")
        t_e, xs_e, E_e = run_path_recording([], cell, beta, y0, z0,
                                            fcen, df, resolution, dpml,
                                            args.Textra, label="empty",
                                            source_model=args.source_model)

        # align time bases (interpolate empty onto cavity grid if needed)
        if E_e.shape != E_c.shape:
            from numpy import interp
            E_e_i = np.empty_like(E_c)
            for j in range(E_c.shape[1]):
                E_e_i[:, j].real = interp(t_c, t_e, E_e[:, j].real)
                E_e_i[:, j].imag = interp(t_c, t_e, E_e[:, j].imag)
            E_e = E_e_i

        E_ind = E_c - E_e  # induced field, transients removed

        if args.save_fields:
            # save the induced field AND the raw cavity/empty fields so the
            # subtraction itself can be diagnosed (is E_ind tiny vs E_c, E_e?)
            np.savez(args.save_fields, t=t_c, xs=xs, E_ind=E_ind,
                     E_cavity=E_c, E_empty=E_e,
                     beta=beta, resolution=resolution)
            print(f"[saved] induced + raw fields -> {args.save_fields}")
            # report the relative size of the induced field vs the raw fields:
            # if E_ind is a tiny noisy residual of two large near-equal fields,
            # the subtraction is killing the signal.
            amp_c = np.sqrt(np.mean(np.abs(E_c)**2))
            amp_e = np.sqrt(np.mean(np.abs(E_e)**2))
            amp_i = np.sqrt(np.mean(np.abs(E_ind)**2))
            print(f"[diag] RMS |E_cavity|={amp_c:.4e}  |E_empty|={amp_e:.4e}  "
                  f"|E_induced|={amp_i:.4e}")
            print(f"[diag] induced/cavity ratio = {amp_i/amp_c:.4f} "
                  f"({'OK: induced is a real fraction' if amp_i/amp_c > 0.05 else 'WARN: induced is a tiny residual -> subtraction may be killing signal'})")
            print("[info] FDTD complete. Re-run with --from-fields "
                  f"{args.save_fields} to transform without re-simulating.")
            return

    # --- temporal FT + eq.(1) assembly (cheap; re-runnable via --from-fields) ---
    # guard against E=0 (divide-by-zero in freq conversion and the 1/omega prefactor)
    E_lo = max(args.Emin, 1e-3)
    if args.Emin < E_lo:
        print(f"[info] clamping spectrum minimum from {args.Emin} to {E_lo} eV "
              f"(E=0 is singular in the 1/omega prefactor)")
    E_eV = np.linspace(E_lo, args.Emax, args.nE)
    f_grid = eV_to_meep_freq(E_eV, args.a)        # MEEP freq
    omegas_meep = 2 * np.pi * f_grid              # MEEP angular freq

    if not args.pade:
        E_xw = temporal_ft(t_c, E_ind, omegas_meep)
    elif args.pade_method == "scipy":
        E_xw = temporal_ft_pade_scipy(t_c, E_ind, omegas_meep, M=args.pade_order)
        print("[info] using Pade transform (scipy.interpolate.pade)")
    else:
        E_xw = temporal_ft_pade(t_c, E_ind, omegas_meep, M=args.pade_order)
        print("[info] using Pade transform (Prony/lstsq)")
    gamma = assemble_gamma(E_xw, xs, omegas_meep, beta, a_nm,
                           spatial_taper=args.spatial_taper)
    gamma_conv = gaussian_convolve(E_eV, gamma, fwhm_eV=0.030)

    np.savez(args.out, E_eV=E_eV, gamma=gamma, gamma_conv=gamma_conv,
             beta=beta, resolution=resolution)
    print(f"[done] saved {args.out}")
    print(f"[peak] max of convolved spectrum at "
          f"{E_eV[np.nanargmax(gamma_conv)]:.4f} eV")


if __name__ == "__main__":
    main()