"""
Re-examine the EELS spectrum against the paper's SUPPLEMENTARY figures.

Key insight from SI Fig S4: the RAW simulated spectrum is SPIKY (log scale,
spans orders of magnitude). Only AFTER 30 meV convolution does it become the
smooth alpha/beta/gamma envelope. So we must judge the CONVOLVED curve, not the
raw one.

SI Fig S3 gives exact TE slot-mode energies: 0.775, 0.794, 0.87, 0.887 eV.

Usage: python reexamine_SI.py fields_<jobid>.npz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from eels_brute_force import (
    eV_to_meep_freq, temporal_ft_pade, temporal_ft, assemble_gamma,
    gaussian_convolve,
)

f = np.load(sys.argv[1] if len(sys.argv) > 1 else "fields.npz")
t = f["t"]; xs = f["xs"]; E_ind = f["E_ind"]; beta = float(f["beta"])
a_nm = 426

# SI Fig S3 exact mode energies, and the SI uses the 0.5-1.3 eV window of S4
SI_MODES = [0.775, 0.794, 0.87, 0.887]
E_eV = np.linspace(0.50, 1.30, 800)
omegas = 2*np.pi*eV_to_meep_freq(E_eV, a_nm)

# transform (Pade) and assemble
Exw = temporal_ft_pade(t, E_ind, omegas)
gamma = assemble_gamma(Exw, xs, omegas, beta, a_nm)
gamma_conv = gaussian_convolve(E_eV, gamma, fwhm_eV=0.030)

# Plot in the SAME style as SI Fig S4: raw on log scale, convolved on linear
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# left: raw, log scale (like S4 left). Use |gamma| since log needs positive.
ax[0].semilogy(E_eV, np.abs(gamma), lw=0.7)
ax[0].set_title("RAW |Gamma| (log scale) — SHOULD be spiky, cf. SI Fig S4 left")
ax[0].set_xlabel("Energy loss (eV)"); ax[0].set_ylabel("|Gamma| (arb, log)")
for m in SI_MODES:
    ax[0].axvline(m, color="r", ls=":", lw=1)

# right: convolved, linear (like S4 right) — THIS is what to compare to Fig 3a
ax[1].plot(E_eV, gamma_conv, lw=1.6)
ax[1].set_title("CONVOLVED Gamma (30 meV) — compare to Fig 3a / SI S4 right")
ax[1].set_xlabel("Energy loss (eV)"); ax[1].set_ylabel("Gamma (arb)")
for m, lab in zip(SI_MODES, ["0.775", "0.794", "0.87", "0.887"]):
    ax[1].axvline(m, color="r", ls=":", lw=1)
    ax[1].text(m, ax[1].get_ylim()[1]*0.9, lab, rotation=90, fontsize=7,
               color="r", ha="right", va="top")

plt.tight_layout()
plt.savefig("reexamine_SI.png", dpi=130)
print("saved reexamine_SI.png")

# Report: does the convolved spectrum have local maxima near the SI modes /
# the alpha-beta-gamma cluster (0.7-0.95)?
from scipy.signal import find_peaks
pk, props = find_peaks(gamma_conv, height=0)
pk_E = E_eV[pk]
print("\nconvolved-spectrum peak energies (eV):")
for e in pk_E:
    near = min(SI_MODES + [0.70, 0.81, 0.95], key=lambda m: abs(m-e))
    flag = "  <- near SI/paper feature %.3f" % near if abs(near-e) < 0.04 else ""
    print(f"  {e:.3f}{flag}")
plt.show()