"""
From ONE saved field file, test how much ringdown is actually needed by
truncating the time series to several lengths and transforming each.

If the spectrum stops changing past some truncation length, that length was
enough ringdown -- you don't need to simulate longer next time.

Usage:
  python ringdown_test.py fields_<jobid>.npz

Produces ringdown_test.png overlaying spectra from truncated records, and
prints how much the spectrum changes as you add more ringdown.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# import the transform + assembly from the main module
sys.path.insert(0, ".")
from eels import (
    eV_to_meep_freq, temporal_ft_pade, assemble_gamma, gaussian_convolve,
)

f = np.load(sys.argv[1] if len(sys.argv) > 1 else "fields.npz")
t = f["t"]
xs = f["xs"]
E_ind = f["E_ind"]
beta = float(f["beta"])
a_nm = 426  # adjust if you changed -a

transit = (xs.max() - xs.min()) / beta
print(f"transit = {transit:.1f} MEEP units, total record = {t.max():.1f}")

# energy grid for the broad alpha/beta/gamma window
E_eV = np.linspace(0.40, 1.00, 600)
omegas = 2 * np.pi * eV_to_meep_freq(E_eV, a_nm)

# truncation fractions of the post-transit record
post = t.max() - transit
fracs = [0.4, 0.6, 0.8, 1.0]

fig, ax = plt.subplots(figsize=(9, 5))
prev = None
for fr in fracs:
    t_cut = transit + fr * post
    mask = t <= t_cut
    tt = t[mask]
    EE = E_ind[mask, :]
    Exw = temporal_ft_pade(tt, EE, omegas)
    g = assemble_gamma(Exw, xs, omegas, beta, a_nm)
    gc = gaussian_convolve(E_eV, g, fwhm_eV=0.030)
    gc_n = gc / (np.max(np.abs(gc)) or 1)
    ax.plot(E_eV, gc_n, lw=1.4,
            label=f"ringdown {fr*post:.0f} (t<{t_cut:.0f})")
    if prev is not None:
        change = np.mean(np.abs(gc_n - prev))
        print(f"  frac {fr:.1f}: mean change from previous = {change:.4f}")
    prev = gc_n

for e, n in [(0.70, "a"), (0.81, "b"), (0.95, "g")]:
    ax.axvline(e, color="gray", ls=":", lw=1)
    ax.text(e, 0.95, n, color="gray", fontsize=9, ha="right")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Energy loss (eV)")
ax.set_ylabel("Gamma (broadened, norm.)")
ax.set_title("Spectrum vs ringdown length (converged = enough ringdown)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ringdown_test.png", dpi=130)
print("saved ringdown_test.png")
print("\nIf the curves CONVERGE as ringdown grows, that length was enough.")
print("If they keep CHANGING, you need a longer simulation next time.")
plt.show()