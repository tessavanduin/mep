"""
Test whether a spatial taper on the eq.1 path-integral removes the ripple.
Runs the transform on ONE saved field file at several taper fractions.

Usage:  python taper_test.py fields_<jobid>.npz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from eels_brute_force import (
    eV_to_meep_freq, temporal_ft_pade, assemble_gamma, gaussian_convolve,
)

f = np.load(sys.argv[1] if len(sys.argv) > 1 else "fields.npz")
t = f["t"]; xs = f["xs"]; E_ind = f["E_ind"]; beta = float(f["beta"])
a_nm = 426

E_eV = np.linspace(0.40, 1.00, 600)
omegas = 2 * np.pi * eV_to_meep_freq(E_eV, a_nm)

# transform once (temporal), then apply different spatial tapers in assembly
Exw = temporal_ft_pade(t, E_ind, omegas)

fig, ax = plt.subplots(figsize=(9, 5))
for tap in [0.0, 0.1, 0.2, 0.35]:
    g = assemble_gamma(Exw, xs, omegas, beta, a_nm, spatial_taper=tap)
    gc = gaussian_convolve(E_eV, g, fwhm_eV=0.030)
    gc_n = gc / (np.max(np.abs(gc)) or 1)
    ax.plot(E_eV, gc_n, lw=1.4, label=f"taper {tap}")

for e, n in [(0.70, "a"), (0.81, "b"), (0.95, "g")]:
    ax.axvline(e, color="gray", ls=":", lw=1)
    ax.text(e, 0.95, n, color="gray", fontsize=9, ha="right")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Energy loss (eV)"); ax.set_ylabel("Gamma (broadened, norm.)")
ax.set_title("Spectrum vs spatial taper (does tapering kill the ripple?)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("taper_test.png", dpi=130)
print("saved taper_test.png")

# also show the spatial profile of |E_ind(x)| averaged over late time, to SEE
# whether the field is nonzero at the path ends (the ripple cause)
late = np.mean(np.abs(E_ind[len(t)//2:, :]), axis=0)
fig2, ax2 = plt.subplots(figsize=(9, 3))
ax2.plot(xs, late, lw=1)
ax2.set_xlabel("x (a)"); ax2.set_ylabel("mean |E_ind| (late time)")
ax2.set_title("Spatial profile -- is the field nonzero at the path ENDS?")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("spatial_profile.png", dpi=130)
print("saved spatial_profile.png")
plt.show()