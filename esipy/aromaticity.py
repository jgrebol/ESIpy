import numpy as np
from esipy.tools import wf_type, mol_info, load_file, format_partition
from esipy.rest import deloc_rest, arom_rest, arom_rest_from_aoms
from esipy.unrest import deloc_unrest, arom_unrest, arom_unrest_from_aoms
from esipy.no import deloc_no, arom_no, arom_no_from_aoms


def aromaticity(Smo, rings, mol=None, mf=None, partition=None, mci=False, av1245=False, flurefs=None, homarefs=None,
                homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1):
    """Calculate population analysis, localization, delocalization indices, and aromaticity indicators.

    Arguments:
        Smo (list of matrices or str):
            Specifies the Atomic Overlap Matrices (AOMs) in the MO basis. This can either be a list of matrices generated from the `make_aoms()` function or a string with the filename/path where the AOMs are saved.

        rings (list or list of lists of int):
            Contains the indices defining the ring connectivity of a system. Can contain several rings as a list of lists.

        mol (SCF instance, optional, default: None):
            PySCF's Mole class and helper functions to handle parameters and attributes for GTO integrals.

        mf (SCF instance, optional, default: None):
            PySCF's object holds all parameters to control SCF.

        partition (str, optional, default: None):
            Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.

        mci (bool, optional, default: False):
            If `True`, the function computes the MCI index.

        av1245 (bool, optional, default: False):
            If `True`, the function computes the AV1245 (and AVmin) indices.

        flurefs (dict, optional, default: None):
            User-provided references for Delocalization Indices used in the FLU index calculation.

        homarefs (dict, optional, default: None):
            User-provided references for the HOMA index. Required data as in [Krygowski, et al. Chem. Rev. 114 6383-6422 (2014)].

        homerrefs (dict, optional, default: None):
            User-provided references for optimal distance and polarizability in HOMA or HOMER indices.

        connectivity (list of int, optional, default: None):
            List of atomic symbols in the order they appear in `mol`, representing ring connectivity.

        geom (list of floats, optional, default: None):
            Molecular coordinates, typically obtained from `mol.atom_coords()`.

        molinfo (dict or str, optional, default: None):
            Contains molecule and calculation details from the 'molinfo()' method inside ESI. Can also be a filename/path where this information is saved.

        ncores (int, optional, default: 1):
            Specifies the number of cores for multi-processing MCI calculation.

    """

    partition = format_partition(partition)
    fromaoms = False
    if molinfo and len(molinfo) != 1:
        if isinstance(molinfo, str):
            with open(molinfo, "rb") as f:
                molinfo = np.load(f, allow_pickle=True)
    else:
        if mol is None:
            if mf is None:
                print(" | Could not find mol nor mf. molinfo only contains the partition. ")
                fromaoms = True
        molinfo = mol_info(mol=mol, mf=mf)

    if fromaoms:
        basisset = "Not specified"
        calctype = "Not specified"
        xc = "Not specified"
        energy = "Not specified"
        method = "Not specified"
    else:
        basisset = molinfo["basisset"]
        calctype = molinfo["calctype"]
        xc = molinfo["xc"]
        energy = molinfo["energy"]
        method = molinfo["method"]

    # Case: Basis set is a dictionary. Will show "Not available"
    if isinstance(basisset, dict):
        basisset = "Different basis sets"

    if isinstance(Smo, str):
        print(f" | Loading the AOMs from file {Smo}")
        Smo = load_file(Smo)
        if Smo is None:
            raise NameError(" | Please provide a valid name to read the AOMs")

    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    if fromaoms is True:
        aromaticity_from_aoms(Smo=Smo, rings=rings, partition=partition, mol=mol, mci=mci, av1245=av1245,
                              flurefs=flurefs, homarefs=homarefs, homerrefs=homerrefs, connectivity=connectivity,
                              geom=geom, ncores=ncores)
        return

    wf = wf_type(Smo)

    ########### PRINTING THE OUTPUT ###########

    # Information from the calculation

    # UNRESTRICTED
    if wf == "unrest":
        print(" | Number of Atoms:          {}".format(len(Smo[0])))
        print(" | Occ. Mol. Orbitals:       {}({})".format(
            int(len(Smo[0][0])), int(len(Smo[1][0]))))
        print(" | Wavefunction type:        Unrestricted")

    # RESTRICTED
    elif wf == "rest":
        print(" | Number of Atoms:          {}".format(len(Smo)))
        print(" | Occ. Mol. Orbitals:       {}({})".format(
            int(len(Smo[0])), int(len(Smo[0]))))
        print(" | Wavefunction type:        Restricted")
    else:
        print(" | Number of Atoms:          {}".format(len(Smo[0])))
        print(" | Wavefunction type:        Natural Orbitals")


    if partition is not None:
        print(" | Atomic partition:         {}".format(partition.upper()))
    else:
        print(" | Atomic partition:         Not specified")
    print(" ------------------------------------------- ")
    print(" | Method:                  ", calctype)

    if "dft" in method and xc is not None:
        print(" | Functional:              ", xc)

    print(" | Basis set:               ", basisset.upper())
    if isinstance(energy, str):
        print(" | Total energy:          {}".format(energy))
    else:
        print(" | Total energy:          {:>13f}".format(energy))
    print(" ------------------------------------------- ")

    if wf == "unrest":
        trace_alpha = np.sum([np.trace(matrix) for matrix in Smo[0]])
        trace_beta = np.sum([np.trace(matrix) for matrix in Smo[1]])
        print(" | Tr(alpha):    {:>13f}".format(trace_alpha))
        print(" | Tr(beta):     {:>13f}".format(trace_beta))
        print(" | Tr(total):    {:>13f}".format(trace_alpha + trace_beta))

    elif wf == "rest":
        trace = np.sum([np.trace(matrix) for matrix in Smo])
        print(" | Tr(Enter):    {:.13f}".format(trace))

    else:
        trace = np.sum([np.trace(matrix) for matrix in Smo[0]])
        sum_nos = np.sum(sum(Smo[1]))
        print(" | Tr(Enter):    {:.13f}".format(trace))
        print(" | Sum(occ):     {:.13f}".format(sum_nos))


    print(" ------------------------------------------- ")

    # Checking the type of calculation and calling the function for each case

    # UNRESTRICTED
    if wf == "unrest":
        deloc_unrest(Smo, mol, molinfo)
        arom_unrest(Smo, rings, partition, mol, mci=mci, av1245=av1245, flurefs=flurefs,
                    homarefs=homarefs, homerrefs=homerrefs, connectivity=connectivity, geom=geom, ncores=ncores,
                    molinfo=molinfo)
    # RESTRICTED
    elif wf == "rest":
        deloc_rest(Smo, mol, molinfo)
        arom_rest(Smo, rings, partition, mol, mci=mci, av1245=av1245, flurefs=flurefs,
                  homarefs=homarefs, homerrefs=homerrefs, connectivity=connectivity, geom=geom, ncores=ncores,
                  molinfo=molinfo)
    # NATURAL ORBITALS
    else:
        deloc_no(Smo, mol, molinfo)
        arom_no(Smo, rings, partition, mol, mci=mci, av1245=av1245, flurefs=flurefs,
                  homarefs=homarefs, homerrefs=homerrefs, connectivity=connectivity, geom=geom, ncores=ncores,
                  molinfo=molinfo)



# AROMATICITY FROM LOADED AOMS
def aromaticity_from_aoms(Smo, rings, mol=None, partition=None, mci=False, av1245=False, flurefs=None, homarefs=None,
                          homerrefs=None, connectivity=None, geom=None, ncores=1):
    """Calculate population analysis, localization, delocalization indices, and aromaticity indicators from previously saved AOMs.

    Arguments:
        Smo (list of matrices or str):
            Specifies the Atomic Overlap Matrices (AOMs) in the MO basis. This can either be a list of matrices generated from the `make_aoms()` function or a string with the filename/path where the AOMs are saved.

        rings (list or list of lists of int):
            Contains the indices defining the ring connectivity of a system. Can contain several rings as a list of lists.

        mol (SCF instance, optional, default: None):
            PySCF's Mole class and helper functions to handle parameters and attributes for GTO integrals.

        partition (str, optional, default: None):
            Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.

        mci (bool, optional, default: False):
            If `True`, the function computes the MCI index.

        av1245 (bool, optional, default: False):
            If `True`, the function computes the AV1245 (and AVmin) indices.

        flurefs (dict, optional, default: None):
            User-provided references for Delocalization Indices used in the FLU index calculation.

        homarefs (dict, optional, default: None):
            User-provided references for the HOMA index. Required data as in [Krygowski, et al. Chem. Rev. 114 6383-6422 (2014)].

        homerrefs (dict, optional, default: None):
            User-provided references for optimal distance and polarizability in HOMA or HOMER indices.

        connectivity (list of int, optional, default: None):
            List of atomic symbols in the order they appear in `mol`, representing ring connectivity.

        geom (list of floats, optional, default: None):
            Molecular coordinates, typically obtained from `mol.atom_coords()`.

        molinfo (dict or str, optional, default: None):
            Contains molecule and calculation details from the 'molinfo()' method inside ESI. Can also be a filename/path where this information is saved.

        ncores (int, optional, default: 1):
            Specifies the number of cores for multi-processing MCI calculation.
    """

    wf = wf_type(Smo)

    ########### PRINTING THE OUTPUT ###########

    # Information from the calculation

    # UNRESTRICTED
    if wf == "unrest":
        print(" | Number of Atoms:          {}".format(len(Smo[0])))
        print(" | Occ. Mol. Orbitals:       {}({})".format(
            int(len(Smo[0][0])), int(len(Smo[1][0]))))
        print(" | Wavefunction type:        Unrestricted")

    # RESTRICTED
    elif wf == "rest":
        print(" | Number of Atoms:          {}".format(len(Smo)))
        print(" | Occ. Mol. Orbitals:       {}({})".format(
            int(len(Smo[0])), int(len(Smo[0]))))
        print(" | Wavefunction type:        Restricted")

    if partition is not None:
        print(" | Atomic partition:         {}".format(partition.upper()))
    else:
        print(" | Atomic partition:         Not specified")

    print(" ------------------------------------------- ")
    print(" | Method:            Not specified")
    print(" | Basis set:         Not specified")
    print(" | Total energy:      Not specified")
    print(" ------------------------------------------- ")

    if wf == "unrest":
        trace_alpha = np.sum([np.trace(matrix) for matrix in Smo[0]])
        trace_beta = np.sum([np.trace(matrix) for matrix in Smo[1]])
        print(" | Tr(alpha):    {:>13f}".format(trace_alpha))
        print(" | Tr(beta):     {:>13f}".format(trace_beta))
        print(" | Tr(total):    {:>13f}".format(trace_alpha + trace_beta))

    elif wf == "rest":
        print(" | Tr(Enter):    {:.13f}".format(
            np.sum([np.trace(matrix) for matrix in Smo])))

    print(" ------------------------------------------- ")

    # Checking the type of calculation and calling the function for each case

    # UNRESTRICTED
    if wf == "unrest":
        arom_unrest_from_aoms(Smo, rings, partition, mol, mci=mci, av1245=av1245,
                              flurefs=flurefs, homarefs=homarefs, homerrefs=homerrefs, connectivity=connectivity,
                              geom=geom, ncores=ncores)

    # RESTRICTED
    elif wf == "rest":
        arom_rest_from_aoms(Smo, rings, partition, mol, mci=mci, av1245=av1245,
                            flurefs=flurefs, homarefs=homarefs, homerrefs=homerrefs, connectivity=connectivity,
                            geom=geom, ncores=ncores)
    else:
        arom_no_from_aoms(Smo, rings, partition, mol, mci=mci, av1245=av1245,
                            flurefs=flurefs, homarefs=homarefs, homerrefs=homerrefs, connectivity=connectivity,
                            geom=geom, ncores=ncores)
