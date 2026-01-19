# Getting started

In this section we will go through the process of performing all the available calculations using ESIpy's utilities
based on the provided examples. There are two main ways to run ESIpy: by creating a PySCF input, running the single-point
calculation and running ESIpy afterwards, or by creating an input from a FCHK file. The second option may not contain
all the features, they will be added in future releases.

## ESIpy from PySCF

ESIpy works on the object `ESI`, which will contain all the information required for the calculation. It is recommended
to initialize the object with all the data, rather than adding it once the initialization process is finished.
To run the code from the terminal, generate the Python script or adapt those of this
webpage and run it as ``python3 code.py``. To save the output as a file, use ``python3 code.py > code.esi``.
Even though the ``.esi`` extension is not mandatory, we recommend using it. The test files have been generated from benzene and naphthalene at
singlet (and triplet) level of theory, and are just set to showcase the capabilities of the code.

The simplest form of input following a usual PySCF calculation

.. code-block:: python

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
    mol.verbose = 9
    mol.build()

    mf = dft.KS(mol)
    mf.kernel()

    ring = [1, 2, 3, 4, 5, 6]
    arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition="nao")
    arom.print()

By providing the `mol` and `mf` objects, ESIpy generates the AOMs in the desired partition and computes the indices following
the ring connectivity in the list `rings`. The program works similarly through unrestricted wavefunctions,
the output of which provides the indices split into orbital contributions.

.. note::
    In the following, we will only consider the ESIpy part of the code.

### Dealing with AOMs

In order to avoid the single-point calculation, the `save` attribute will save the AOMs and a dictionary
containing information about the molecule and calculation into a binary file in disk. It should contain only the title of
the calculation, and ESIpy will add the used partition scheme and the extension (`.aoms` and `.molinfo`). Hereafter, these will be accessible
at any time. It is also recommended to use a for-loop scheme for all the partitions, as the computational time to generate
the matrices is minimal and independent to the chosen scheme.

.. code-block:: python

    ring = [1, 2, 3, 4, 5, 6]
    name = "benzene"
    for part in ["mulliken", "lowdin", "meta_lowdin", "nao", "iao"]:
        arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition=part, save=name)
        arom.print()

Additionally, one can generate a directory containing the AOMs in AIMAll format. These files are readable from ESIpy,
but also from Eduard Matito's ESI-3D code. These are written through the method `writeaoms()`

.. code-block:: python

    arom = esipy.ESI(mol=mol, mf=mf, rings=[1,2,3,4,5,6], partition="nao")
    arom.writeaoms("benzene_nao")

and read through the method `readaoms()` by previously specifying the `read=True` attribute and using another generated
"molinfo" dictionary.

.. code-block:: python

    arom = esipy.ESI(rings=[1,2,3,4,5,6], partition="nao", read=True, molinfo="benzene_nao.molinfo")
    arom.readaoms()
    arom.print()

.. warning::
    By using the `readaoms()` method, the output will be limited as it will not get information about the molecule.

### Correlated wavefunctions

For natural orbitals wavefunctions, an additional diagonalization
of the first-order reduced density matrix (1-RDM) is carried out, the computational time of which is also very low.
The single-determinant (RHF or UHF) object has to be provided through the `myhf` attribute. Population analyses use both Fulton's approach
and the 2-RDM approximation in terms of natural occupations, but only Fulton's approach is used for the aromaticity
calculations.

.. code-block:: python

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
    The IAOs expand the occupied orbitals in the same rank as the minimal basis, but the role of valence orbitals
    is important for the calculation. Therefore, the transformation matrix is computed through the RHF object,
    thus making the `myhf` attribute needed for these calculations. However, it is recommended to use other robust schemes
    for multi-determinant wave functions.

## ESIpy from a FCHK file

ESIpy can also read wavefunctions from a FCHK file generated by Gaussian. This is done by generating a simple input.
As an example, the simplest form of input would be:

.. code-block::

    $READFCHK
    benzene.fchk
    $RING
    1 2 3 4 5 6
    $PARTITION
    META_LOWDIN
    NAO
    IAO

All the keywords available are explained in the `ESIpy FCHK input reference <fchk.html>`_ section.