# ESIpy
The ESIpy program is aimed to the calculation of population analysis and aromaticity indicators from different Hilbert space partitions using the PySCF. At the moment only single-determinant wavefunctions for restricted and unrestricted calculations can be computed. The atomic partitions supported from the program are Mulliken, Löwdin, meta-Löwdin, Natural Atomic Orbitals (NAO) and Intrinsic Atomic Orbitals (IAO). The ESIpy respository contains the esi.py main code as well as several example scripts.

# Features
- make_aoms(mol, mf, calc): From PySCF's 'mol' and 'mf' objects and 'calc' as a string containing the desired partition ('mul', 'lowdin', 'meta_lowdin', 'nao', 'iao'), generate a list of matrices containing the Atomic Overlap Matrices (AOMs).
- aromaticity(mol, mf, Smo, ring, calc, mci, av1245, num_threads): Compute population analyses, delocalization analyses and aromaticity indicators from the AOMs (variable Smo). The variable 'ring' is either a list or a list of lists containing the indices of the atoms for the aromaticity calculations. 'mci' and 'av1245' are boolean variables to compute the MCI and AV1425 indices, respectively. Multi-core processing is supported, albeit the speed-up is non-linear.

# Utilities
- write_int(mol, mf, molname, Smo, ring, None): Writes the AOMs as an input for Dr. Matito's ESI-3D code. The atomic files are stored in a directory, as well as a general input for the program ('molname'.bad).

# Installation
To install PySCF it is recommended to create a conda environment as follows:
```
conda create --name pyscf_env python=3.9
```
and install PySCF as:
```
conda activate pyscf_env
pip install pyscf
```
To install ESIpy in your local working stations:
```
mkdir ~/ESIpy
cd ~/ESIpy
git clone git@github.com:jgrebol/ESIpy.git
```
Add to your ```.bashrc``` file:
```
export PYTHONPATH=~/ESIpy/ESIpy:$PYTHONPATH (or the directory where it is located)
```
