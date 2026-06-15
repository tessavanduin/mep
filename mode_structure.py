"""
Check whether the induced field has SPATIAL MODE STRUCTURE at the slot-mode
frequencies. For EELS to show alpha/beta/gamma, E_x(x, omega) along the path
must carry the mode's spatial profile at those energies. If it's flat, the
electron isn't imprinting the mode -> no peaks possible.

Usage:  python mode_structure.py fields_<jobid>.npz
        python mode_structure.py fields_A.npz fields_B.npz   # compare two sources
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from eels_brute_force import eV_to_meep_freq, temporal_ft_pade

files = sys.argv[1:]
if not files:
    files = ["fields.npz"]

# energies to inspect: the three peaks + one off-peak control
E_check = {"alpha 0.70": 0.70, "beta 0.81": 0.81, "gamma 0.95": 0.95,
           "off-peak 0.55": 0.55}
a_nm = 426

fig, axes = plt.subplots(len(E_check), 1, figsize=(10, 2.4*len(E_check)),
                         sharex=True)

for fname in files:
    f = np.load(fname)
    t = f["t"]; xs = f["xs"]; E_ind = f["E_ind"]
    omegas = 2*np.pi*eV_to_meep_freq(np.array(list(E_check.values())), a_nm)
    # temporal FT at exactly these frequencies -> E_x(x, omega) [nE, nx]
    Exw = temporal_ft_pade(t, E_ind, omegas)
    for ax, (label, Eval), row in zip(axes, E_check.items(), Exw):
        prof = np.abs(row)
        ax.plot(xs, prof / (prof.max() or 1), lw=1,
                label=f"{fname.split('/')[-1]}")
        ax.set_title(f"|E_x(x)| at {label} eV   "
                     f"(flat = no mode; oscillating = mode present)")
        ax.set_ylabel("|E_x| norm")
        ax.grid(alpha=0.3)

axes[-1].set_xlabel("x along path (a)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("mode_structure.png", dpi=130)
print("saved mode_structure.png")

# quantify: a structured mode has high spatial variance; flat has ~0.
# print a 'structure score' = std/mean of |E_x(x)| at each energy
print("\nstructure score (std/mean of |E_x(x)|; higher = more mode structure):")
for fname in files:
    f = np.load(fname)
    t = f["t"]; xs = f["xs"]; E_ind = f["E_ind"]
    omegas = 2*np.pi*eV_to_meep_freq(np.array(list(E_check.values())), a_nm)
    Exw = temporal_ft_pade(t, E_ind, omegas)
    print(f"  {fname}:")
    for (label, _), row in zip(E_check.items(), Exw):
        p = np.abs(row)
        score = np.std(p) / (np.mean(p) + 1e-30)
        print(f"     {label}: {score:.3f}")
plt.show()