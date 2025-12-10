# Photonic Crystal Simulations
This repository is dedicated to simulating certain photonic crystal (PhC) properties such as their band diagrams and electron energy loss spectra (EELS) using the [MEEP](https://meep.readthedocs.io/) python package for Finite Difference Time Domain Methods (FDTD). It is part of my applied physics bachelor thesis at TU Delft.

This project uses Meep, which is licensed under GPL-2.0+.
Meep is not distributed as part of this repository.

## Instalation
1) First, I recommend installing `pymeep` using the conda package manager as explained in the MEEP documentation [here](https://meep.readthedocs.io/en/master/Installation/#conda-packages). There is a sequential (mp) and parallel (pmp) version. Both should work in principle but the code might take longer when running on multiple cores due to communication overhead so I recommend running it sequentially for the default geometries.

2) Second, clone this repository. In principle, the files that run the simulation should now work. EELS_3D.py includes a CLI option `-p` to plot your PhC geometry. For this, additional packages may need to be installed. Always **install aditional packages using**:
```
conda install -c conda-forge <some-package>
```

3. Third, for the EELS simulation data processing, a pade Fourier Transform is used. The code for this is available [here](https://github.com/jjgoings/pade). This repository needs to be cloned into PhC-EELS so that its contents can be used by `eels.ipynb`. The folder structure will look like:
```bash
PhC-EELS/
├── pade/
│   ├── pade.py
│   ⋮
├── eels.ipynb
⋮
```

## Usage
The files that perform simulations will output HDF5 files containing electromagnetic time series data. Subsequently, this data can be analyzed using the accompanying notebooks.
### Band diagrams
The repository includes two scripts to generate band diagram data for two geometries. One of that of a "pure" PhC and one for a PhC with slot defect. The python scripts can be used on their own but a script `run_parallel.sh` is included to perform multiple simulations for different k-values simultaneously. The shell script accepts two parameters, the number of k values in the irriducible Brillouin zone and the number of processes you want to split the task into. Example usage of the script could look like
```bash
(time ./run_parallel.sh 64 8) > output.log 2>&1 &
```
This will try to find eigen frequencies for 64 k points using 8 parallel processes, this will take the same time as computing 64/8 = 8 k points sequentially. Though, with this setup, always use less processes than that your CPU has cores to avoid context switching. Using more processes than cores will result in a massive slow down that may take even longer than computing everything sequentially.
Furthermore, two notebooks are included that can be used to plot band diagram data. One can be used to plot two sets of data in one plot, useful for when you want to combine even and odd mode simulations in the same figure.

### EELS
EELS data can be generated using `EELS_3D.py`. It can be run using various CLI options that can be viewed by running the file with `-h`. The time series data is stored in the folder `EELS_3D-out/`. To compute the EELS, an additional simulation needs to be run that uses the same parameters but in an empty simulaiton domain, (this data is subtracted from the simulations with dielectric to find the induced electric field). Doing this is easy, one can specify the same simulation but run it with the option `-e` for `--empty`, this will create an empty simulation domain of the same size as if the specified crystal geometry would be there. Once the simulation has finished, the eels spectrum can be plotted by running the cells in `eels.ipynb`. You may have to change the name of the files that are loaded, but it will generally attempt to load 1 file for an empty simulation domain for subtraction and all files that have CRYSTAL in their name.

