"""
EELS.py

Computes EELS spectrum from vacuum and crystal runs.
Now adapted for 4-D field data: Ez(t, x, y, z).
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def load_data(filename):
    """Load time array and 4-D Ez field."""
    with h5py.File(filename, "r") as f:
        t = f["time"][:]              # shape (Nt,)
        Ez = f["Ez"][:]               # shape (Nt, Nx, Ny, Nz)
        cell_x = f.attrs["cell_x"]
        cell_y = f.attrs["cell_y"]
        cell_z = f.attrs["cell_z"]
        resolution = f.attrs["resolution"]
        v = f.attrs["v_electron"]

    return t, Ez, cell_x, cell_y, cell_z, resolution, v


def extract_beam_line(Ez_txyz, cell_x, cell_y, cell_z, resolution, v, t):
    """
    Extract Ez(t) along the electron trajectory:
    electron moves along z at x=0, y=0.
    """

    Nt, Nx, Ny, Nz = Ez_txyz.shape

    # grid coordinates
    x_coords = np.linspace(-cell_x/2, cell_x/2, Nx)
    y_coords = np.linspace(-cell_y/2, cell_y/2, Ny)
    z_coords = np.linspace(-cell_z/2, cell_z/2, Nz)

    # beam is at x=0, y=0
    ix = np.argmin(np.abs(x_coords - 0))
    iy = np.argmin(np.abs(y_coords - 0))

    # electron path: z(t) = -half_span + v*t
    half_span = (cell_z - 2*(cell_z/6)) / 2   # same dpml logic as sim
    z_path = -half_span + v * t               # physical z positions

    Ez_line = np.zeros(Nt)

    for it in range(Nt):
        iz = np.argmin(np.abs(z_coords - z_path[it]))
        Ez_line[it] = Ez_txyz[it, ix, iy, iz]

    return Ez_line


def pade_spectrum(signal, dt):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.fft.rfft(signal)
    return freqs, spec


def loss_probability(freqs, spectrum):
    return np.abs(spectrum)**2


def gaussian_broadening(signal, sigma):
    x = np.arange(-200, 201)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return fftconvolve(signal, kernel, mode="same")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vacuum", required=True)
    parser.add_argument("--crystal", required=True)
    parser.add_argument("--sigma", type=float, default=3)
    args = parser.parse_args()

    # --- Load vacuum ---
    t_vac, Ez_vac_4d, cx, cy, cz, res, v = load_data(args.vacuum)
    dt = t_vac[1] - t_vac[0]

    Ez_vac = extract_beam_line(Ez_vac_4d, cx, cy, cz, res, v, t_vac)

    # --- Load crystal ---
    t_crys, Ez_crys_4d, cx2, cy2, cz2, res2, v2 = load_data(args.crystal)
    Ez_crys = extract_beam_line(Ez_crys_4d, cx2, cy2, cz2, res2, v2, t_crys)

    print("Ez_vac shape:", Ez_vac.shape)
    print("Ez_crys shape:", Ez_crys.shape)

    # --- Induced field ---
    induced = Ez_crys - Ez_vac

    # --- Spectrum ---
    freqs, spec = pade_spectrum(induced, dt)
    loss = loss_probability(freqs, spec)
    loss = gaussian_broadening(loss, args.sigma)

    # --- Convert frequency to energy (eV) ---
    hbar = 6.582119569e-16   # eV*s
    c = 299792458            # m/s
    a = 1e-6                 # Meep unit length (adjust to your simulation)

    omega = 2 * np.pi * c / a * freqs
    energy_eV = hbar * omega

    # --- Convert loss to (% / e- / meV) ---
    dE = np.mean(np.diff(energy_eV))     # eV
    loss_per_eV = loss / dE
    loss_per_meV = loss_per_eV / 1000.0
    loss_percent = 100 * loss_per_meV


    # --- Plot ---
    plt.figure(figsize=(7, 4))
    plt.plot(energy_eV, loss_percent)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Loss Probability (% / e- / meV)")
    plt.title("Electron Energy Loss Spectrum")
    plt.tight_layout()
    plt.savefig("EELS_spectrum.png", dpi=300)


if __name__ == "__main__":
    main()
