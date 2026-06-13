import sys
import numpy as np
import matplotlib.pyplot as plt

f = np.load(sys.argv[1] if len(sys.argv) > 1 else "eels_spectrum.npz")
E = f["E_eV"]
raw = f["gamma"]
conv = f["gamma_conv"]

fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

ax[0].plot(E, raw, lw=0.8, alpha=0.5, label="Gamma (raw)")
ax[0].plot(E, conv, lw=1.8, label="Gamma (40 meV broadened)")
ax[0].set_ylabel("Loss prob. (arb.)")
ax[0].legend()
ax[0].set_title("EELS spectrum")

# also show |gamma| so a peak that dips negative still shows its location
ax[1].plot(E, np.abs(conv), lw=1.8, color="C3")
ax[1].set_ylabel("|Gamma| broadened")
ax[1].set_xlabel("Energy loss (eV)")

for a in ax:
    a.grid(alpha=0.3)
    a.axhline(0, color="k", lw=0.5)

plt.tight_layout()
plt.savefig("eels_plot.png", dpi=130)
print("saved eels_plot.png")

# diagnostics
print(f"E range: [{E.min():.3f}, {E.max():.3f}] eV")
print(f"gamma:      min={raw.min():.3e}  max={raw.max():.3e}")
print(f"gamma_conv: min={conv.min():.3e}  max={conv.max():.3e}")
imax = np.nanargmax(np.abs(conv))
print(f"largest |gamma_conv| at {E[imax]:.4f} eV "
      f"({'edge!' if imax in (0, len(E)-1) else 'interior'})")
plt.show()