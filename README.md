<p align="center"><img width="40.0%" src="https://github.com/jgrebol/ESIpy/blob/main/logoesipy.png"></p>

The ESIpy program is aimed at the calculation of population analysis and aromaticity indicators from different
Hilbert-space partitions using the PySCF module. The program supports both restricted and unrestricted calculations for
single-determinant wavefunctions, and correlated wavefunctions from a restricted object (RHF). The atomic partitions
supported by the program are Mulliken, Löwdin, meta-Löwdin, Natural Atomic Orbitals (NAO), and Intrinsic Atomic Orbita (
IAO).

The on-line documentation can be found [here](https://esipython.readthedocs.io/en/latest/).

## Citation

All the calculations performed for the creation and implementation of this program have been conducted in the following
scientific paper:

**Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador, Chem. Eur. J. 2024, 30, e202401282.**

Also, find it on-line [here](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.202401282?af=R). If you
are publishing the results obtained from ESIpy, remember to cite the program. The code is licensed under the GNU GPLv3.
See the [LICENSE](LICENSE) file for details. See the on-line documentation for details on how to
use the program. If you encounter any bugs, please feel free to report them on
the [Issues page](https://github.com/jgrebol/ESIpy/issues), or send an email
to [joan.grebol@dipc.org](mailto:joan.grebol@dipc.org).

## Installation

ESIpy can be installed through:

```
pip install esipython
```

The latest stable version can be obtained through:

```
pip upgrade esipython
```

The latest non-stable version available on Github can be obtained through:

```
pip install --upgrade git+https://github.com/jgrebol/ESIpy.git
```

For a detailed explanation on how to run the code and how to customize it, please see the [documentation](esipython.readthedocs.io).

## Getting started

ESIpy works on the object `ESI`, which will contain all the information required for the calculation. It is recommended
to initialize the object with all the data, rather than adding it once the initialization process is finished.

The simplest form of input follows a usual PySCF calculation

```python
    from pyscf import gto, dft
    import esipy

    mol = gto.Mole()
    mol.atom = '''
    6        0.000000000      0.000000000      1.393096000
    6        0.000000000      1.206457000      0.696548000
    6        0.000000000      1.206457000     -0.696548000
    6        0.000000000      0.000000000     -1.393096000
    6        0.000000000     -1.206457000     -0.696548000
    6        0.000000000     -1.206457000      0.696548000
    1        0.000000000      0.000000000      2.483127000
    1        0.000000000      2.150450000      1.241569000
    1        0.000000000      2.150450000     -1.241569000
    1        0.000000000      0.000000000     -2.483127000
    1        0.000000000     -2.150450000     -1.241569000
    1        0.000000000     -2.150450000      1.241569000
    '''
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = True
    mol.verbose = 0
    mol.build()

    mf = dft.KS(mol)
    mf.kernel()

    ring = [1, 2, 3, 4, 5, 6]
    arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition="nao")
    arom.print()
```

To avoid the single-point calculation, the attributes `saveaoms` and `savemolinfo` will save the AOMs and a dictionary
containing information about the molecule and calculation into a binary file in disk. Hereafter, these will be accessible
at any time. It is also recommended to use a for-loop scheme for all the partitions, as the computational time to generate
the matrices is minimal and independent of the chosen scheme.

```python
    ring = [1, 2, 3, 4, 5, 6]
    name = "benzene"
    for part in ["mulliken", "lowdin", "meta_lowdin", "nao", "iao"]:
        aoms_name = name + '_' + part + '.aoms'
        molinfo_name = name + '_' + part + '.molinfo'
        arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition=part, saveaoms=aoms_name, savemolinfo=molinfo_name)
        arom.print()
```

Additionally, one can generate a directory containing the AOMs in AIMAll format. These files are readable from ESIpy,
but also from Eduard Matito's ESI-3D code. These are written through the method `writeaoms()`:

```python
    arom = esipy.ESI(mol=mol, mf=mf, rings=[1,2,3,4,5,6], partition="nao")
    arom.writeaoms("benzene_nao")
```

## Further work

- Approximations for the MCI calculation in large systems.
- Read the AOMs (or the data required for their calculation) from other source programs and store them as ESIpy `Smo`.
- Calculation of aromaticity indicators from defined fragments.
- Split the calculation into orbital contributions.
- Algorithm to automatically find the rings inside a system.
- Adaptation of some indicators to non-closed circuits (e.g., linear chains).

