r"""
Fast EELS via Green-tensor / mode-decomposition (eq.3 of the paper).

Gamma(omega, y0, z0) = (1/E0) * (e^2 / (2 pi hbar omega)) *
        sum_k Re[ i omega / (omega_k^2 - omega^2)
                  * | FT[ u_xk(x', y0, z0, omega_k) ]_(omega/v) |^2 ]

with the mode normalized so  (1/2) \int eps0 eps_r |u_k|^2 d^3r = E0 = 1 J,
and FT[...]_(omega/v) the SPATIAL Fourier transform of the mode's x-component
along x, evaluated at wavevector k = omega/v.

Unlike the brute-force method (one FDTD run per (y,z) pixel), this needs only:
  Stage 1  -- Harminv: excite the cavity, find complex resonance(s) omega_k, Q.
  Stage 2  -- capture the steady mode field u_k(r) on a volume (or midplane).
  Stage 3  -- pure post-processing: for ANY trajectory (y0,z0), pull the line
              u_xk(x,y0,z0), spatial-FT along x, eval at omega/v, square, and
              sum the Lorentzians. Spatial mapping is then nearly free.

Workflow:
  python eels_green.py --cavity --stage harminv      # find modes, print table
  python eels_green.py --cavity --stage capture       # capture mode fields -> .npz
  python eels_green.py --cavity --stage spectrum      # eq.3 spectrum at (y0,z0)
  python eels_green.py --cavity --stage all           # do all three in sequence

Mode selection:
  default            -> sum ALL modes Harminv found in the window
  --mode-index 0     -> only the strongest mode
  --mode-index 0,2   -> a hand-picked subset (indices into the printed table)

Normalization:
  --norm3d           -> full 3D energy integral (exact, slower capture)
  (default)          -> z=0 midplane slice, scaled by thickness (approximate)
"""

import argparse
import numpy as np

try:
    import meep as mp
except ImportError:
    mp = None

# reuse geometry + helpers from the brute-force module (same directory)
from eels_brute_force import (
    SlottedTriangleLattice, SlottedTriangleLatticeCavity,
    electron_beta, meep_freq_to_eV, eV_to_meep_freq, build_phc, make_cell,
    gaussian_convolve, _C, _HBAR_EV, _E_CHARGE,
)

# physical constants not imported above
_EPS0 = 8.8541878128e-12  # F/m


# ----------------------------------------------------------------------------
# Stage 1: Harminv -- find the complex resonant frequencies in the window
# ----------------------------------------------------------------------------

def stage_harminv(args, geometry, cell, resolution, dpml, fcen, df):
    """Excite the cavity with a broadband pulse, run Harminv at the slot centre,
    return a list of dicts: {freq, decay, Q, amp, omega_k(complex, MEEP)}."""
    pml = [mp.PML(dpml)]
    # a couple of point sources slightly off-centre to excite both even/odd modes
    src = [
        mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey,
                  center=mp.Vector3(0.05, 0.0, 0.0)),
        mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey,
                  center=mp.Vector3(0.37, 0.11, 0.0)),
    ]
    sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=src,
                        boundary_layers=pml, resolution=resolution)

    h = mp.Harminv(mp.Ey, mp.Vector3(0.0, 0.0, 0.0), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=args.harminv_time)

    modes = []
    for m in h.modes:
        # MEEP complex frequency: f_complex = f * (1 + i/(2Q)) sign convention;
        # store omega_k = 2 pi f_complex (we use Re for the Lorentzian centre,
        # Im encodes the linewidth/Q).
        omega_k = 2 * np.pi * m.freq * (1 + 1j / (2 * m.Q)) if m.Q != 0 else 2 * np.pi * m.freq
        modes.append(dict(freq=m.freq, decay=m.decay, Q=m.Q,
                          amp=abs(m.amp), omega_k=complex(omega_k),
                          E_eV=float(meep_freq_to_eV(m.freq, args.a))))
    # sort by amplitude descending so index 0 = strongest
    modes.sort(key=lambda d: d["amp"], reverse=True)
    return modes


def print_mode_table(modes, a_nm):
    print("\n  idx |   f (c/a) |    E (eV) |       Q |   |amp|")
    print("  ----+-----------+-----------+---------+--------")
    for i, m in enumerate(modes):
        print(f"  {i:3d} | {m['freq']:9.5f} | {m['E_eV']:9.4f} | "
              f"{m['Q']:7.0f} | {m['amp']:.3e}")
    print()


# ----------------------------------------------------------------------------
# Stage 2: capture the mode field u_k(r) and its normalization constant
# ----------------------------------------------------------------------------

def stage_capture(args, geometry, cell, resolution, dpml, mode_freq, norm3d):
    r"""Narrow-band excite at mode_freq, let transients die, then snapshot the
    field. Returns dict with Ex line(s) and the normalization integral
    N = (1/2) \int eps0 eps_r |u|^2 d^3r used to enforce E0 = 1 J.

    We capture Ex over the volume (norm3d) or the z=0 plane (default).
    """
    pml = [mp.PML(dpml)]
    df = mode_freq / 50.0  # narrow band around the mode
    src = [mp.Source(mp.GaussianSource(mode_freq, fwidth=df), component=mp.Ey,
                     center=mp.Vector3(0.05, 0.0, 0.0))]
    sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=src,
                        boundary_layers=pml, resolution=resolution,
                        force_complex_fields=True)

    # run well past the source so only the (highest-Q) mode rings
    sim.run(until_after_sources=args.capture_time)

    # define capture volume
    if norm3d:
        vol = mp.Volume(center=mp.Vector3(), size=cell)
    else:
        vol = mp.Volume(center=mp.Vector3(),
                        size=mp.Vector3(cell.x, cell.y, 0))

    ex = sim.get_array(vector3=None, center=vol.center, size=vol.size,
                       component=mp.Ex)
    ey = sim.get_array(center=vol.center, size=vol.size, component=mp.Ey)
    ez = sim.get_array(center=vol.center, size=vol.size, component=mp.Ez)
    eps = sim.get_array(center=vol.center, size=vol.size, component=mp.Dielectric)

    # energy normalization N = 1/2 sum eps0 * eps_r * |u|^2 * dV
    dV = (1.0 / resolution) ** (3 if norm3d else 2)
    a_m = args.a * 1e-9
    dV_phys = dV * (a_m ** (3 if norm3d else 2))
    if not norm3d:
        dV_phys *= (args.d * 1e-9)  # multiply slab thickness for the z extent
    u2 = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    N = 0.5 * _EPS0 * np.sum(eps * u2) * dV_phys  # joules, before rescale

    # coordinate axes for the captured array
    xs = np.linspace(-cell.x / 2, cell.x / 2, ex.shape[0])
    ys = np.linspace(-cell.y / 2, cell.y / 2, ex.shape[1])
    if norm3d:
        zs = np.linspace(-cell.z / 2, cell.z / 2, ex.shape[2])
    else:
        zs = np.array([0.0])

    return dict(Ex=ex, eps=eps, xs=xs, ys=ys, zs=zs, N=N,
                norm3d=norm3d, mode_freq=mode_freq)


# ----------------------------------------------------------------------------
# Stage 3: eq.3 post-processing -- spectrum at a trajectory (y0, z0)
# ----------------------------------------------------------------------------

def spatial_ft_at_kv(ux_line, xs, k_eval):
    r"""FT[u_x(x)]_k = \int u_x(x) e^{-i k x} dx, evaluated at each k in k_eval.
    ux_line: complex [nx]; xs: [nx]; k_eval: [nomega]. Returns complex [nomega]."""
    dx = xs[1] - xs[0]
    phase = np.exp(-1j * np.outer(k_eval, xs))   # [nomega, nx]
    return (phase @ ux_line) * dx                 # [nomega]


def gamma_eq3(modes_used, capture_by_mode, omegas_meep, beta, a_nm,
              y0, z0):
    r"""Assemble eq.3 summed over the selected modes.
    omegas_meep: angular MEEP freq grid [nomega] (=2 pi f).
    capture_by_mode: list of capture dicts aligned with modes_used.
    Returns Gamma(omega) [nomega] in absolute units (prob per (MEEP-omega))."""
    a_m = a_nm * 1e-9
    omega_phys = omegas_meep * (_C / a_m)          # rad/s
    # k = omega / v ; v = beta c ; in MEEP units k_meep = omega_meep / beta
    k_meep = omegas_meep / beta

    gamma = np.zeros_like(omegas_meep, dtype=float)
    for mode, cap in zip(modes_used, capture_by_mode):
        # pull u_x along x at (y0,z0) from the captured field
        ux_line = _extract_line(cap, y0, z0)
        # enforce E0 = 1 J: u_normalized = u / sqrt(N)  so that the energy
        # integral becomes 1 J; |FT|^2 then scales by 1/N.
        ux_line = ux_line / np.sqrt(cap["N"])
        FT = spatial_ft_at_kv(ux_line, cap["xs"], k_meep)   # [nomega]
        FT2 = np.abs(FT) ** 2

        omega_k = mode["omega_k"]                  # complex MEEP angular freq
        wk_phys = omega_k * (_C / a_m)             # rad/s (complex)
        lorentz = 1j * omega_phys / (wk_phys**2 - omega_phys**2)
        gamma += np.real(lorentz * FT2)

    # prefactor e^2 / (2 pi hbar omega) ; E0 = 1 J already folded via /sqrt(N)
    hbar_J = _HBAR_EV * _E_CHARGE
    prefactor = (_E_CHARGE ** 2) / (2 * np.pi * hbar_J * omega_phys)
    return prefactor * gamma


def _extract_line(cap, y0, z0):
    """Nearest-pixel u_x(x) along the line (y0, z0) from a capture dict."""
    iy = int(np.argmin(np.abs(cap["ys"] - y0)))
    if cap["norm3d"]:
        iz = int(np.argmin(np.abs(cap["zs"] - z0)))
        return cap["Ex"][:, iy, iz]
    else:
        return cap["Ex"][:, iy]


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # structural params (same defaults as brute-force)
    p.add_argument("-a", type=int, default=426)
    p.add_argument("-d", type=int, default=220)
    p.add_argument("-x", type=int, default=36)
    p.add_argument("-W", type=float, default=1.2)
    p.add_argument("-s", type=int, default=100)
    p.add_argument("-r", type=float, default=0.245)
    p.add_argument("-n", type=float, default=3.45)
    p.add_argument("-v", type=float, default=100.0)
    p.add_argument("--cavity", action="store_true")
    p.add_argument("--res", type=float, default=None)
    # spectral window
    p.add_argument("--Emin", type=float, default=0.70)
    p.add_argument("--Emax", type=float, default=0.90)
    p.add_argument("--nE", type=int, default=400)
    # method controls
    p.add_argument("--stage", choices=["harminv", "capture", "spectrum", "all"],
                   default="all")
    p.add_argument("--harminv-time", type=float, default=600.0)
    p.add_argument("--capture-time", type=float, default=800.0)
    p.add_argument("--mode-index", type=str, default=None,
                   help="comma-sep indices into the mode table; default = all")
    p.add_argument("--norm3d", action="store_true",
                   help="full 3D energy normalization (else z=0 midplane)")
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--z0", type=float, default=0.0)
    p.add_argument("--modes-file", type=str, default="modes.npz")
    p.add_argument("--out", type=str, default="eels_green_spectrum.npz")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if mp is None:
        raise SystemExit("MEEP not installed here. Run on your machine/cluster.")

    resolution = (args.a / 18.0) if args.res is None else args.res
    if args.quick:
        resolution = max(8.0, resolution / 4.0)
        args.harminv_time = 200.0
        args.capture_time = 250.0

    beta = electron_beta(args.v)
    geometry, struct_cell, thickness, a_nm = build_phc(args)
    dpml = 1.0
    cell = make_cell(struct_cell, thickness, dpml, pad_xy=1.0,
                     pad_z=1.5 * thickness)

    f_win = eV_to_meep_freq(np.array([args.Emin, args.Emax]), args.a)
    fcen = float(np.mean(f_win))
    df = float(abs(f_win[1] - f_win[0]) * 1.4 + 0.05)

    # ---- Stage 1: harminv ----
    if args.stage in ("harminv", "all"):
        print("[stage] harminv ...")
        modes = stage_harminv(args, geometry, cell, resolution, dpml, fcen, df)
        print_mode_table(modes, a_nm)
        np.savez(args.modes_file,
                 freqs=[m["freq"] for m in modes],
                 Qs=[m["Q"] for m in modes],
                 amps=[m["amp"] for m in modes],
                 omega_k=[m["omega_k"] for m in modes],
                 E_eV=[m["E_eV"] for m in modes])
        if args.stage == "harminv":
            return
    else:
        d = np.load(args.modes_file, allow_pickle=True)
        modes = [dict(freq=f, Q=q, amp=a_, omega_k=complex(w),
                      E_eV=float(meep_freq_to_eV(f, args.a)))
                 for f, q, a_, w in zip(d["freqs"], d["Qs"], d["amps"],
                                        d["omega_k"])]

    # which modes to use
    if args.mode_index is None:
        idx = list(range(len(modes)))
    else:
        idx = [int(i) for i in args.mode_index.split(",")]
    modes_used = [modes[i] for i in idx]
    print(f"[info] using modes {idx}: "
          f"{[round(modes[i]['E_eV'],4) for i in idx]} eV")

    # ---- Stage 2: capture each used mode ----
    if args.stage in ("capture", "all"):
        print(f"[stage] capture ({'3D' if args.norm3d else 'midplane'}) ...")
        caps = []
        for i, m in zip(idx, modes_used):
            print(f"  capturing mode {i} @ {m['E_eV']:.4f} eV")
            cap = stage_capture(args, geometry, cell, resolution, dpml,
                                m["freq"], args.norm3d)
            caps.append(cap)
            print(f"    N = {cap['N']:.3e} J  (pre-rescale energy integral)")
        # cache the lines only (volumes are big) keyed by chosen y0,z0 later
        np.savez(args.out + ".caps.npz",
                 **{f"Ex_{j}": c["Ex"] for j, c in enumerate(caps)},
                 **{f"xs_{j}": c["xs"] for j, c in enumerate(caps)},
                 **{f"ys_{j}": c["ys"] for j, c in enumerate(caps)},
                 **{f"zs_{j}": c["zs"] for j, c in enumerate(caps)},
                 **{f"N_{j}": c["N"] for j, c in enumerate(caps)},
                 norm3d=args.norm3d)
        if args.stage == "capture":
            return
    else:
        z = np.load(args.out + ".caps.npz", allow_pickle=True)
        caps = []
        for j in range(len(modes_used)):
            caps.append(dict(Ex=z[f"Ex_{j}"], xs=z[f"xs_{j}"],
                             ys=z[f"ys_{j}"], zs=z[f"zs_{j}"],
                             N=float(z[f"N_{j}"]), norm3d=bool(z["norm3d"])))

    # ---- Stage 3: eq.3 spectrum ----
    print(f"[stage] spectrum at (y0,z0)=({args.y0},{args.z0}) ...")
    E_eV = np.linspace(args.Emin, args.Emax, args.nE)
    omegas_meep = 2 * np.pi * eV_to_meep_freq(E_eV, args.a)
    gamma = gamma_eq3(modes_used, caps, omegas_meep, beta, a_nm,
                      args.y0, args.z0)
    gamma_conv = gaussian_convolve(E_eV, gamma, fwhm_eV=0.040)

    np.savez(args.out, E_eV=E_eV, gamma=gamma, gamma_conv=gamma_conv,
             modes_used=idx, y0=args.y0, z0=args.z0, beta=beta)
    print(f"[done] saved {args.out}")
    print(f"[peak] convolved max at "
          f"{E_eV[np.nanargmax(gamma_conv)]:.4f} eV")


if __name__ == "__main__":
    main()
