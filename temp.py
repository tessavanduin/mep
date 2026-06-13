import numpy as np
import matplotlib.pyplot as plt

data = np.load("/home/tessa/PhC-EELS/eels_spectrum.npz")

E = data["E_eV"]          # x-axis
gamma = data["gamma"]     # raw spectrum
gamma_c = data["gamma_conv"]  # broadened spectrum

plt.figure(figsize=(6,4))
plt.plot(E, gamma_c, label="Gamma (broadened)")
plt.plot(E, gamma, '--', alpha=0.5, label="Gamma (raw)")

plt.xlabel("Energy loss (eV)")
plt.ylabel("Loss probability (arb. units)")
plt.title("EELS Spectrum")
plt.xlim(-0.25, 2)        # your usual range
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()