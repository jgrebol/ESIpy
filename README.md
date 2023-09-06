# ESIpy
The ESIpy program is aimed to the calculation of population analysis and aromaticity indicators from different Hilbert space partitions using the PySCF. At the moment only single-determinant wavefunctions for restricted and unrestricted calculations can be computed. The atomic partitions supported from the program are Mulliken, Löwdin, meta-Löwdin, Natural Atomic Orbitals (NAO) and Intrinsic Atomic Orbitals (IAO).

# Features
- make_aoms(mol, mf, calc): From PySCF's 'mol' and 'mf' objects and 'calc' as a string containing the desired partition ('mul', 'lowdin', 'meta_lowdin', 'nao', 'iao'), generate a list of matrices containing the Atomic Overlap Matrices (AOMs).
- aromaticity(mol, mf, Smo, ring, calc, mci, av1245, num_threads): Compute population analyses, delocalization analyses and aromaticity indicators from the AOMs (variable Smo). The variable 'ring' is either a list or a list of lists containing the indices of the atoms for the aromaticity calculations. 'mci' and 'av1245' are boolean variables to compute the MCI and AV1425 indices, respectively. Multi-core processing is supported, albeit the speed-up is non-linear.

# Utilities
- write_int(mol, mf, molname, Smo, ring, None): Write the AOMs as an input for Dr. Matito's ESI-3D code.

# Installation
To install PySCF it is recommended to create a conda environment as follows:
```
conda create --name pyscf_env python=3.9
```

