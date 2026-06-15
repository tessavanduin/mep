"""
Why is the induced field nonzero only in the leftmost part of the path?
Inspect the RAW cavity and empty arrays directly.

Usage: python inspect_raw.py fields_<jobid>.npz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

f = np.load(sys.argv[1] if len(sys.argv) > 1 else "fields.npz")
t = f["t"]; xs = f["xs"]; E_ind = f["E_ind"]
print("E_ind shape:", E_ind.shape, " xs shape:", xs.shape)
print("xs range:", xs.min(), "to", xs.max())

has_raw = "E_cavity" in f and "E_empty" in f
if has_raw:
    E_c = f["E_cavity"]; E_e = f["E_empty"]
    print("E_cavity shape:", E_c.shape, " E_empty shape:", E_e.shape)

# where is each field nonzero along x? (max over time at each pixel)
def nz_profile(E):
    return np.max(np.abs(E), axis=0)

fig, ax = plt.subplots(2 if has_raw else 1, 1, figsize=(10, 7), squeeze=False)

p_ind = nz_profile(E_ind)
ax[0,0].plot(xs, p_ind, lw=1, label="|E_induced| max-over-t")
ax[0,0].set_title("Induced field: max |E| over time at each x pixel")
ax[0,0].set_xlabel("x (a)"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)
# find the last nonzero pixel
nz = np.where(p_ind > 0.01*p_ind.max())[0]
if len(nz):
    print(f"induced field nonzero from x={xs[nz[0]]:.1f} to x={xs[nz[-1]]:.1f} "
          f"(pixels {nz[0]}..{nz[-1]} of {len(xs)})")

if has_raw:
    ax[1,0].plot(xs[:E_c.shape[1]], nz_profile(E_c), lw=1, label="|E_cavity|")
    ax[1,0].plot(xs[:E_e.shape[1]], nz_profile(E_e), lw=1, label="|E_empty|")
    ax[1,0].set_title("Raw cavity & empty: max |E| over time at each x pixel")
    ax[1,0].set_xlabel("x (a)"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)
    # are cavity and empty nonzero over the FULL path or also truncated?
    for nm, E in [("cavity", E_c), ("empty", E_e)]:
        p = nz_profile(E)
        nz = np.where(p > 0.01*p.max())[0]
        if len(nz):
            print(f"{nm} nonzero from pixel {nz[0]} to {nz[-1]} of {E.shape[1]}")

plt.tight_layout()
plt.savefig("inspect_raw.png", dpi=130)
print("saved inspect_raw.png")
plt.show()