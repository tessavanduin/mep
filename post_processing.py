import numpy as np

# physical constants
_C = 299792458.0          # m/s
_HBAR_EV = 6.582119569e-16  # eV*s
_E_CHARGE = 1.602176634e-19  # C


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