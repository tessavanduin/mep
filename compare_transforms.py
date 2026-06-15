"""
Overlay the three transform outputs (DFT, Prony-Pade, scipy-Pade) on one axis
so we can see whether any reproduces the paper's alpha/beta/gamma structure.

Usage:
  python compare_transforms.py t_dft.npz t_prony.npz t_scipy.npz

Paper reference peaks (parallel geometry, rho=0.245, Fig 3a):
  alpha ~ 0.70 eV (dielectric band)
  beta  ~ 0.81 eV (dielectric slot / cavity peak)  <- the important one
  gamma ~ 0.95 eV (air band / air slot)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

labels = ["DFT", "Prony-Pade", "scipy-Pade"]
files = sys.argv[1:4]
PAPER = {"alpha": 0.70, "beta": 0.81, "gamma": 0.95}

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

for fname, lab in zip(files, labels):
    try:
        d = np.load(fname)
    except Exception as e:
        print(f"skip {fname}: {e}")
        continue
    E = d["E_eV"]
    raw = d["gamma"]
    conv = d["gamma_conv"]
    # normalize each to its own max |.| so shapes are comparable despite scale
    norm = np.max(np.abs(conv)) or 1.0
    axes[0].plot(E, conv / norm, lw=1.6, label=lab)
    axes[1].plot(E, np.abs(conv) / norm, lw=1.6, label=lab)
    pk = E[np.nanargmax(np.abs(conv))]
    print(f"{lab:12s}: peak |gamma_conv| at {pk:.4f} eV"
          f"{'  (window edge!)' if np.argmax(np.abs(conv)) in (0, len(E)-1) else ''}")

for ax in axes:
    for name, e in PAPER.items():
        ax.axvline(e, color="gray", ls=":", lw=1)
        ax.text(e, ax.get_ylim()[1]*0.92, name, rotation=90,
                va="top", ha="right", fontsize=8, color="gray")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Gamma (broadened, norm.)")
axes[0].set_title("EELS transforms vs paper alpha/beta/gamma positions")
axes[1].set_ylabel("|Gamma| (broadened, norm.)")
axes[1].set_xlabel("Energy loss (eV)")

plt.tight_layout()
plt.savefig("compare_transforms.png", dpi=130)
print("saved compare_transforms.png")
plt.show()