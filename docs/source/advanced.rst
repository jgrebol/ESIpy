Advanced customizations
=======================

Custom references
-----------------

Some aromaticity indicators, namely FLU, HOMA and HOMER, rely on reference systems. FLU requires the delocalization index
of some aromatic compounds: "CC" from benzene, "CN" from pyridine, "BN" from borazine, "NN" from pyridazine and
"CS" from thiophene, all optimized at HF/6-31G* level of theory. In the following example, a reference DI for a "CC" bond
will update the existing one, and the value is just to showcase this feature.

.. code-block::

    fluref = {'CC': 1.500}
    connectivity = ['C', 'C', 'C', 'C', 'C', 'C']
    aom = 'benzene_nao.aoms'
    molinfo = 'benzene_nao.molinfo'

    arom = esipy.ESI(aom=aom, molinfo=molinfo, rings=ring, partition=partition, flurefs=fluref, connectivity=connectivity)
    arom.print()

HOMA requires the **n_opt**, **c** and **r1** parameters, which
are obtained from :cite:`kruszewski:72tl`. However, we recommend using
HOMER's custom references, which only require the optimal distance between the two atoms (**r_opt**) and the polarizability of the bond (**alpha**).
The results, therefore, will be displayed in the HOMER section of the output. The following example showcases the use of
HOMA's default references for benzene into HOMER, which will lead to the same value.

.. code-block::

    homerref = {'CC': {'r_opt': 1.388, 'alpha': 257.7}}
    connectivity = ['C', 'C', 'C', 'C', 'C', 'C']
    ring = [1, 2, 3, 4, 5, 6]
    partition = 'nao'

    # The geometry can be directly extracted from the mol.atom_coords() method
    geom = [[-2.58138047, -1.26965218, -5.69564833],
        [-4.98179941, -2.68686741, -5.69564833],
        [-7.38220701, -1.26965218, -5.69564833],
        [-7.38220701, 1.26965218, -5.69564833],
        [-4.98178807, 2.68686741, -5.69564833],
        [-2.58138047, 1.26965218, -5.69564833],
        [-0.80908759, -2.32169862, -5.69564833],
        [-4.98179941, -4.74141922, -5.69564833],
        [-9.15450366, -2.32169484, -5.69564833],
        [-9.15449988, 2.32169862, -5.69564833],
        [-4.98178807, 4.74141922, -5.69564833],
        [-0.80908381, 2.32169484, -5.69564833]]
    molinfo = 'benzene_nao.molinfo'
    aom = 'benzene_nao.aoms'

    esipy.ESI(aom=aom, molinfo=molinfo, rings=ring, partition=partition, homerrefs=homerref, connectivity=connectivity,
          geom=geom).print()

Partition format
----------------

The partition string can be input in different names, as ESIpy will recognize the following:

- **Natural Atomic Orbitals:** "nao", "n" and "natural"
- **Intrinsic Atomic Orbitals:** "iao", "i" and "intrinsic"
- **Lowdin:** "lowdin", "l" and "low"
- **Mulliken:** "mulliken", "m" and "mul"
- **meta-Lowdin:** "meta_lowdin", "ml", "mlow", "m-low", "meta-low", "metalow", "mlowdin", "m-lowdin", "metalowdin", "meta-lowdin"

AV1245 and MCI calculation
--------------------------

The calculation of the AV1245, AVmin and MCI are set as default depending on the size of the ring. That is, if not specified otherwise by the user, the MCI
will not be computed for rings with less than 12 atoms, and the AV1245 and AVmin will be computed for rings larger than
10. However, it can be specified whether to compute these indices or not through by setting the `av1245` and `mci`
attributes to True.

The calculation of the MCI can be performed in parallel, as its computational time is the highest among the indices.
By default, the program will only use one core, but the user can specify the number of cores to be used in the calculation
through the `ncores` attribute. It is highly recommended increasing the number of cores for large rings. Please see the
`MCI TIMINGS <mci-timings.html>`_ for more information.

Multiple rings calculation
--------------------------

ESIpy can take into account different rings within the same molecule. The variable rings would, hence, become a list of
lists. The following example showcases the rings of a given Naphthalene molecule, which contains two 6-membered rings
and a 10-membered ring, all of which are included in the same calculation.

.. code-block::

    ring = [[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 10, 9, 8, 7, 6]]
    arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition="nao")
    arom.print()

Individual indices
------------------

Even though we strongly suggest using the keyword `ESI.print()` to compute all the indices at once, the user can also
access the individual indices through the `ESI.indicators` attribute. If there is only one ring, the indicators still
need to be accessible through the `ESI.indicators[0]` attribute. In more than one rings, the indices will be stored in
`ESI.indicators[0]`, `ESI.indicators[1]`, and so on. For instance, the AV1245 for the first ring can be accessed through:

.. code-block::

    arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition="nao")
    arom.print()
    print(arom.indicators[0].av1245)

