"""
Diagnose the EELS ripple by looking at the saved induced field directly.
Usage:  python diagnose_fields.py fields_<jobid>.npz
Reads the --save-fields output (t, xs, E_ind, beta) and shows:
  1. E_ind at the central path pixel vs time   -> transients / reflections
  2. |E_ind| summed over the path vs time        -> when energy is present
  3. the spatial profile at a few snapshots      -> moving spot vs standing pattern
It prints the transit time and the time delays that would produce a ripple of
a given energy period, so we can match the spectral ripple to a physical delay.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

f = np.load(sys.argv[1] if len(sys.argv) > 1 else "fields.npz")
t = f["t"]
xs = f["xs"]
E = f["E_ind"]          # [ntime, nx] complex
beta = float(f["beta"])

# if raw cavity/empty fields are present, show them too (subtraction check)
has_raw = "E_cavity" in f and "E_empty" in f
if has_raw:
    E_cav = f["E_cavity"]
    E_emp = f["E_empty"]
    ic0 = E.shape[1] // 2
    figR, axR = plt.subplots(2, 1, figsize=(9, 7))
    axR[0].plot(t, E_cav[:, ic0].real, lw=0.8, label="cavity")
    axR[0].plot(t, E_emp[:, ic0].real, lw=0.8, label="empty")
    axR[0].plot(t, E[:, ic0].real, lw=1.0, label="induced (cav-empty)")
    axR[0].set_title("Raw cavity vs empty vs induced (center pixel, Re)")
    axR[0].set_xlabel("time (MEEP units)"); axR[0].legend(); axR[0].grid(alpha=0.3)
    # if cavity and empty nearly overlap, induced is their small difference;
    # if they look totally different, either strong scattering or misalignment
    diff_metric = np.mean(np.abs(E_cav - E_emp)) / (np.mean(np.abs(E_cav)) + 1e-30)
    axR[1].plot(t, np.abs(E_cav[:, ic0]), lw=0.8, label="|cavity|")
    axR[1].plot(t, np.abs(E_emp[:, ic0]), lw=0.8, label="|empty|")
    axR[1].set_title(f"Magnitudes (mean|cav-empty|/mean|cav| = {diff_metric:.2f})")
    axR[1].set_xlabel("time (MEEP units)"); axR[1].legend(); axR[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("subtraction_check.png", dpi=130)
    print("saved subtraction_check.png")

_C = 299792458.0
_H_EV = 4.135667696e-15

ntime, nx = E.shape
print(f"field shape: {ntime} timesteps x {nx} path pixels")
print(f"t range: [{t.min():.1f}, {t.max():.1f}] MEEP units, dt~{np.median(np.diff(t)):.4f}")
print(f"beta = {beta:.4f}")

# transit time across the recorded path (MEEP units, c=1, a=1)
path_len = xs.max() - xs.min()
transit = path_len / beta
print(f"path length = {path_len:.2f} a, transit time = {transit:.1f} MEEP units")

# a ripple of energy-period dE corresponds to a time delay dt = h/dE.
# convert: in MEEP units, time is in a/c. dE[eV] <-> dt[MEEP] via the a-scaling
# done in the main code; here we just report transit & total record for matching.
print(f"total record after transit = {t.max() - transit:.1f} MEEP units (ringdown)")

ic = nx // 2  # central pixel
fig, ax = plt.subplots(3, 1, figsize=(8, 9))

ax[0].plot(t, E[:, ic].real, lw=0.8, label="Re E_ind (center pixel)")
ax[0].plot(t, np.abs(E[:, ic]), lw=0.8, alpha=0.7, label="|E_ind|")
ax[0].axvline(transit, color="r", ls="--", lw=1, label="transit time")
ax[0].set_xlabel("time (MEEP units)")
ax[0].set_ylabel("E_ind at center")
ax[0].legend(); ax[0].set_title("Time signal at central path pixel")

power_t = np.sum(np.abs(E)**2, axis=1)
ax[1].semilogy(t, power_t + 1e-30, lw=0.9)
ax[1].axvline(transit, color="r", ls="--", lw=1)
ax[1].set_xlabel("time (MEEP units)")
ax[1].set_ylabel("sum_x |E_ind|^2")
ax[1].set_title("Energy along path vs time (look for echoes after transit)")

# spatial snapshots
for frac in [0.25, 0.5, 0.75]:
    it = int(frac * ntime)
    ax[2].plot(xs, np.abs(E[it, :]), lw=0.9, label=f"t={t[it]:.0f}")
ax[2].set_xlabel("x (a)")
ax[2].set_ylabel("|E_ind|")
ax[2].set_title("Spatial profile snapshots")
ax[2].legend()

plt.tight_layout()
plt.savefig("field_diagnosis.png", dpi=130)
print("saved field_diagnosis.png")

# quick automated checks
late = power_t[t > transit]
if len(late) and late.max() > 0.1 * power_t.max():
    print("[FLAG] significant energy AFTER transit -> reflection/echo or high-Q ringing")
peak_t = t[np.argmax(power_t)]
print(f"peak energy at t={peak_t:.1f} (transit={transit:.1f})")
if peak_t > 1.5 * transit:
    print("[FLAG] energy peaks well after transit -> likely a reflection")
plt.show()