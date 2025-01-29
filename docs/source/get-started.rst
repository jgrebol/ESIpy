Getting started
========

In this section we will go through the process of performing all the available calculations using ESIpy's utilities
based on the provided examples. To run the code from the terminal, generate the Python script or adapt those of this
webpage and run it as ``python3 code.py``. To save the output as a file, use ``python3 code.py > code.esi``.
Even though the ``.esi`` extension is not mandatory, we recommend using it. The test files have been generated from benzene and naphthalene at
singlet (and triplet) level of theory, and are just set to showcase the capabilities of the code.

Core ESIpy
----------

ESIpy works on the object `ESI`, which will contain all the information required for the calculation. It is recommended
to initialize the object with all the data, rather than adding it once the initialization process is finished.

The simplest form of input follows a usual PySCF calculation:

.. code-block::

    from pyscf import gto, dft
    import esipy

    mol = gto.Mole()
    mol.atom = '''
    C       -2.989895238      0.000000000      0.822443952
    C       -2.989895238      1.206457000      0.125895952
    C       -2.989895238      1.206457000     -1.267200048
    C       -2.989895238      0.000000000     -1.963748048
    C       -2.989895238     -1.206457000     -1.267200048
    C       -2.989895238     -1.206457000      0.125895952
    H       -2.989895238      0.000000000      1.912474952
    H       -2.989895238      2.150450000      0.670916952
    H       -2.989895238      2.150450000     -1.812221048
    H       -2.989895238      0.000000000     -3.053779048
    H       -2.989895238     -2.150450000     -1.812221048
    H       -2.989895238     -2.150450000      0.670916952
    '''
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "B3LYP"
    mf.kernel()

    ring = [1, 2, 3, 4, 5, 6]
    arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition="nao")
    arom.print()

By providing the `mol` and `mf` objects, ESIpy generates the AOMs in the desired partition and computes the indices following
the ring connectivity in the list `ring`. The program works similarly through unrestricted wavefunctions,
the output of which provides the indices split into orbital contributions.

.. note::
    In the following, we will only consider the ESIpy part of the code.

Dealing with AOMs
------------

In order to avoid the single-point calculation, the attributes `saveaoms` and `savemolinfo` will save the AOMs and a dictionary
containing information about the molecule and calculation into a binary file in disk. Hereafter, these will be accessible
at any time. It is also recommended to use a for-loop scheme for all the partitions, as the computational time to generate
the matrices is minimal and independent to the chosen scheme.

.. code-block::

    ring = [1, 2, 3, 4, 5, 6]
    name = "benzene"
    for part in ["mulliken", "lowdin", "meta_lowdin", "nao", "iao"]:
        aoms_name = name + '_' + part + '.aoms'
        molinfo_name = name + '_' + part + '.molinfo'
        arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition=part, saveaoms=aoms_name, savemolinfo=molinfo_name)
        arom.print()

Additionally, one can generate a directory containing the AOMs in AIMAll format. These files are readable from ESIpy,
but also from Eduard Matito's ESI-3D code. These are written through the method `writeaoms()`

.. code-block::

    arom = esipy.ESI(mol=mol, mf=mf, rings=[1,2,3,4,5,6], partition="nao")
    arom.writeaoms("benzene_nao.aoms")

and read through the method `readaoms()`

.. code-block::

    arom = esipy.ESI(rings=[1,2,3,4,5,6], partition="nao")
    arom.readaoms()

.. warning::
    By using the `readaoms()` method, the output will be limited as it will not get information about the molecule

Correlated wavefunctions
------------

For natural orbitals wavefunctions, an additional diagonalization
of the first-order reduced density matrix (1-RDM) is carried out, the computational time of which is also very low.
The single-determinant (RHF) object has to be provided through the `myhf` attribute. Both Fulton's and Mayer's
approximations are used for the population analysis, but only Fulton's approximation is used for the aromaticity
calculations.

.. code-block::

    from pyscf import gto, scf, ci, cc, mp, mcscf
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
    mol.max_memory = 4000
    mol.build()

    mf = scf.RHF(mol).run()

    print("Running CCSD calculation...")
    mf1 = cc.CCSD(mf).run()
    print("Running CISD calculation...")
    mf2 = ci.CISD(mf).run()
    print("Running CASSCF calculation...")
    mf3 = mcscf.CASSCF(mf, 6, 6).run()
    print("Running MP2 calculation...")
    mf4 = mp.MP2(mf).run()
    ring = [1, 2, 3, 4, 5, 6]

    for part in ["mulliken", "lowdin", "meta_lowdin", "nao", "iao"]:
        for method in [mf1, mf2, mf3, mf4]:
            arom = esipy.ESI(mol=mol, mf=method, myhf=mf, rings=ring, partition=part)
            arom.print()

.. note::
    The IAOs expand the occupied orbitals in the same rank as the minimal basis. However, the role of valence orbitals
    is important for the calculation. Therefore, the transformation matrix is computed through the RHF object,
    thus making the `myhf` attribute needed for these calculations.
