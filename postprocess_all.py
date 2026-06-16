"""
postprocess_all.py  --  headless EELS post-processing for batch / cluster runs.

Pairs every  EELS_3D-out/EELS_3D-CRYSTAL_*.h5  with the (single) EMPTY run,
computes Gamma(omega), and writes a CSV + PNG per spectrum into  eels-spectra/.
No interactive display is used (matplotlib Agg backend), so it is safe under
SLURM / batch.

Usage:
    python postprocess_all.py [--method dft|pade] [--emax 2.1] [--fwhm 0.030]
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless: no X server needed
import matplotlib.pyplot as plt

from eels_postprocess import spectrum

OUTDIR = "eels-spectra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dft", choices=["dft", "pade"])
    ap.add_argument("--emax", type=float, default=2.1, help="max energy loss [eV]")
    ap.add_argument("--npts", type=int, default=1500)
    ap.add_argument("--fwhm", type=float, default=30e-3, help="Gaussian FWHM [eV]")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    energies = np.linspace(0.0, args.emax, args.npts)

    empties = sorted(glob.glob("EELS_3D-out/EELS_3D-EMPTY_*.h5"))
    crystals = sorted(glob.glob("EELS_3D-out/EELS_3D-CRYSTAL_*.h5"))
    if not empties:
        raise SystemExit("No EMPTY run found in EELS_3D-out/ -- run the --empty job first.")
    if not crystals:
        raise SystemExit("No CRYSTAL/cavity runs found in EELS_3D-out/.")
    empty = empties[0]
    print(f"Empty reference: {empty}")

    for cr in crystals:
        tag = os.path.basename(cr).replace("EELS_3D-", "").replace(".h5", "")
        try:
            E, G = spectrum(cr, empty, energies_eV=energies,
                            method=args.method, fwhm_eV=args.fwhm)
        except Exception as exc:                              # keep going on one bad file
            print(f"  [skip] {tag}: {exc}")
            continue

        np.savetxt(os.path.join(OUTDIR, f"{tag}.csv"),
                   np.column_stack([E, G]),
                   delimiter=",", header="energy_eV,probability_percent_per_eV",
                   comments="")

        plt.figure(figsize=(7, 4))
        plt.plot(E, G, lw=1.2)
        plt.axhline(0, ls="--", lw=0.5, color="grey")
        plt.xlabel("Energy loss (eV)")
        plt.ylabel("Probability (% / eV)")
        plt.title(tag)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{tag}.png"), dpi=150)
        plt.close()
        print(f"  wrote {OUTDIR}/{tag}.csv and .png")

    print("Post-processing done.")


if __name__ == "__main__":
    main()