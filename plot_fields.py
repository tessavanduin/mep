import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "EELS_3D_perp_a36_r245_c0.h5"

with h5py.File(filename, "r") as f:
    Ez_t = np.array(f["Ez"])   # (Nt, Nx, Ny, Nz)
    Ey_t = np.array(f["Ey"])
    Ex_t = np.array(f["Ex"])

# pick a z-plane (middle of slab)
mid_z = Ez_t.shape[3] // 2
mid_y = Ey_t.shape[2] // 2

# pick a time index
it = -1   # last snapshot
Ez_xy = Ez_t[it, :, :, mid_z]   # (Nx, Ny)
Ez_xz = Ez_t[it, :, mid_y, :]   # (Nx, Nz)
Ex_xy = Ex_t[-1, :, :, mid_z]
Ey_xz = Ey_t[-1, :, mid_y, :]



"""Plotting field snapshots from EELS_3D.py."""
### Efield slices at mid z-plane and mid y-plane, at the last time snapshot.
# plt.imshow(Ex_xy.T, cmap="RdBu", origin="lower", aspect="auto")
# plt.colorbar()
# plt.title("Ex field slice")
# plt.show()

### Efield slices at mid z-plane and mid y-plane, at the last time snapshot.
# plt.imshow(Ey_xz.T, cmap="RdBu", origin="lower", aspect="auto")
# plt.colorbar()
# plt.title("Ey field slice")
# plt.show()

### Efield slices at mid z-plane and mid y-plane, at the last time snapshot.
plt.figure(figsize=(7,6))
plt.imshow(Ez_xz.T, cmap="RdBu", origin="lower", aspect="auto")
plt.colorbar(label="Ez")
plt.title(f"Ez(x,z) at z = mid, t index = {it}")
plt.xlabel("x index")
plt.ylabel("z index")
plt.tight_layout()
plt.show()

### Efield slices at mid z-plane and mid y-plane, at the last time snapshot.
# plt.imshow(Ez_xz.T, cmap="RdBu", origin="lower", aspect="auto")
# plt.colorbar()
# plt.title("Ez(x,z) slice")
# plt.xlabel("x index")
# plt.ylabel("z index")
# plt.show()

### Efield slices at mid z-plane and mid y-plane, video over time.
# for it in range(Ez_t.shape[0]):
#     Ez_xy = Ez_t[it, :, :, mid_z]
#     plt.imshow(Ez_xy.T, cmap="RdBu", origin="lower", aspect="auto")
#     plt.title(f"t index = {it}")
#     plt.pause(0.1)
#     plt.clf()

### Efield slices at mid z-plane and mid y-plane, video over time.
# for it in range(Ez_t.shape[0]):
#     Ez_xz = Ez_t[it, :, mid_y, :]
#     plt.imshow(Ez_xz.T, cmap="RdBu", origin="lower", aspect="auto")
#     plt.title(f"t index = {it}")
#     plt.pause(0.1)
#     plt.clf()
