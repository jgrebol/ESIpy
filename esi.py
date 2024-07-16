import numpy as np
from os import environ

environ["NUMEXPR_NUM_THREADS"] = "1" 
environ["OMP_NUM_THREADS"] = "1" 
environ["MKL_NUM_THREADS"] = "1" 

##################################
########### CORE ESIpy ###########
##################################


def aromaticity(Smo, rings, mol=None, mf=None, partition=None, mci=False, av1245=False, flurefs=None, homarefs=None, connectivity=None, geom=None, molinfo=None, num_threads=None):
    """Population analysis, localization and delocalization indices and aromaticity indicators.

    Arguments:

       Smo: list of matrices / string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       rings: list
          Contains a list of the indices of the atoms in the ring connectivity for the aromaticity calculations.

       mol: an instance of SCF class. Default: None
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mf: an instance of SCF class. Default: None
          mf object holds all parameters to control SCF.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

       mci: boolean. Default: None
          Whether to compute the MCI index.

       av1245: boolean. Default: None
          Whether to compute the AV1245 (and AVmin) indices.

       flurefs: dictionary. Default: None
          User-provided references for the Delocalization Indices for the FLU index.

       homarefs: dictionary. Default: None
          User-provided references for the distance and polarizability for the HOMA or HOMER indices.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       geom: list. Default: None
          The molecular coordinates as given by the mol.atom_coords() function.

       molinfo: list / string. Default: None
          Contains information from the molecule and calculation to complete the output from the mol_info() function.
          Can also be a string with the name of the file where this information has been saved.

       num_threads: integer. Default: None
          Number of threads required for the calculation.

    """

    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    print(" ** Localization & Delocalization Indices **  ")
    print(" ** For Hilbert-Space Atomic Partitioning **  ")
    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    print( "   Application to Aromaticity Calculations   ")
    print("  Joan Grebol, Eduard Matito, Pedro Salvador  ")
    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    global atom_numbers, symbols
    global basisset, calctype, functional, energy, method

    fromaoms = False
    if molinfo is not None:
        if isinstance(molinfo, str):
            with open(molinfo, "rb") as f:
                molinfo = np.load(f, allow_pickle=True)

    if mol is None:
        if molinfo is None:
            print(" | 'mol' object not found")
            symbols = None
            fromaoms = True
        else:
            print(" | Obtaining molecular information from .molinfo file")
            symbols = molinfo[0]
            atom_numbers = molinfo[1]
            basisset = molinfo[2]
            geom = molinfo[3]
    else:
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        connectivity = symbols
        atom_numbers = [i + 1 for i in range(mol.natm)]
        basisset = mol.basis.upper()
        geom = mol.atom_coords()

    if mf is None:
        if molinfo is None:
            print(" | 'mf' object not found")
            calctype = "Not specified"
            functional = "Not specified"
            energy = "Not specified"
            method = "Not specified"
        else:
            print(" | Obtaining calculation information from .molinfo file")
            calctype = molinfo[4]
            functional = molinfo[-1]
            energy = molinfo[5]
            method = molinfo[6]
    else:
        calctype = mf.__class__.__name__
        if "dft" in mf.__module__ and mf.xc is not None:
            functional = mf.xc
        else:
            functional = "Not specified"
        energy = mf.e_tot
        method = mf.__module__

    # Set fromaoms to False only if molinfo is provided and both mol and mf are None
    if mol is None and mf is None and molinfo is not None:
        fromaoms = False

    if isinstance(Smo, str):
        print(f" | Loading the AOMs from file {Smo}")
        Smo = load_aoms(Smo)

    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    if fromaoms is True:
        aromaticity_from_aoms(Smo=Smo, rings=rings, partition=partition, mol=mol, mci=mci, av1245=av1245,
            flurefs=flurefs, homarefs=homarefs, connectivity=connectivity, geom=geom, num_threads=num_threads)
        return

    wf = rest_or_unrest(Smo)

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
    print(" | Method:                  ", calctype)

    if "dft" in method and functional is not None:
        print(" | Functional:              ", functional)

    print(" | Basis set:               ", basisset)
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

    else:
        trace = np.sum([np.trace(matrix) for matrix in Smo])
        print(" | Tr(Enter):    {:.13f}".format(trace))
    print(" ------------------------------------------- ")

    # Checking the type of calculation and calling the function for each case

    # UNRESTRICTED
    if wf == "unrest":
        deloc_unrest(mol, Smo, molinfo=molinfo)
        arom_unrest(Smo, rings, partition, mol, mci=mci, av1245=av1245, flurefs=flurefs,
            homarefs=homarefs, connectivity=connectivity, geom=geom, num_threads=num_threads,)

    # RESTRICTED
    elif wf == "rest":
        deloc_rest(mol, Smo, molinfo=molinfo)
        arom_rest( Smo, rings, partition, mol, mci=mci, av1245=av1245, flurefs=flurefs,
            homarefs=homarefs, connectivity=connectivity, geom=geom, num_threads=num_threads,)


# AROMATICITY FROM LOADED AOMS
def aromaticity_from_aoms(Smo, rings, mol=None, partition=None, mci=False, av1245=False, flurefs=None, homarefs=None, connectivity=None, geom=None, num_threads=None):
    """Population analysis, localization and delocalization indices and aromaticity 
    indicators from previously saved AOMs from the make_aoms() function.

    Arguments:

       Smo: list of matrices / string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       rings: list
          Contains a list of the indices of the atoms in the ring connectivity for the aromaticity calculations.

       mol: an instance of SCF class. Default: None
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

       mci: boolean. Default: None
          Whether to compute the MCI index.

       av1245: boolean. Default: None
          Whether to compute the AV1245 (and AVmin) indices.

       flurefs: dictionary. Default: None
          User-provided references for the Delocalization Indices for the FLU index.

       homarefs: dictionary. Default: None
          User-provided references for the distance and polarizability for the HOMA or HOMER indices.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       geom: list. Default: None
          The molecular coordinates as given by the mol.atom_coords() function.

       molinfo: list / string. Default: None
          Contains information from the molecule and calculation to complete the output from the mol_info() function.
          Can also be a string with the name of the file where this information has been saved.

       num_threads: integer. Default: None
          Number of threads required for the calculation.

    """

    wf = rest_or_unrest(Smo)

    ########### PRINTING THE OUTPUT ###########

    # Information from the calculation

    # UNRESTRICTED
    if wf == "unrest":
        print(" | Number of Atoms:          {}".format(len(Smo[0])))
        print(" | Occ. Mol. Orbitals:       {}({})".format(
                int(len(Smo[0][0])), int(len(Smo[1][0]))))
        print(" | Wavefunction type:        Unrestricted")

    # RESTRICTED
    if wf == "rest":
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

    else:
        print(" | Tr(Enter):    {:.13f}".format(
                np.sum([np.trace(matrix) for matrix in Smo])))
    print(" ------------------------------------------- ")

    # Checking the type of calculation and calling the function for each case

    # UNRESTRICTED
    if wf == "unrest":
        arom_unrest_from_aoms(Smo, rings, partition, mol, mci=mci, av1245=av1245,
            flurefs=flurefs, homarefs=homarefs, connectivity=connectivity, geom=geom, num_threads=num_threads)

    # RESTRICTED
    elif wf == "rest":
        arom_rest_from_aoms(Smo, rings, partition, mol, mci=mci, av1245=av1245,
            flurefs=flurefs, homarefs=homarefs, connectivity=connectivity, geom=geom, num_threads=num_threads)


########### POPULATION STUDIES ###########

# POPULATION AND DELOCALIZAION UNRESTRICTED


def deloc_unrest(mol, Smo, molinfo=None):
    """Population analysis, localization and delocalization indices for unrestriced AOMs.

    Arguments:

       mol: an instance of SCF class
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       molinfo: list / string. Default: None
          Contains information from the molecule and calculation to complete the output from the mol_info() function.
          Can also be a string with the name of the file where this information has been saved.
    """

    if mol is not None:
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    elif molinfo is not None:
        if isinstance(molinfo, str):
            with open(molinfo, "rb") as f:
                molinfo = np.load(f, allow_pickle=True)
        symbols = molinfo[0]
    else:
        raise NameError(" Could not find 'mol' nor 'molinfo' file")

    # Getting the LIs and DIs
    dis_alpha, dis_beta = [], []
    lis_alpha, lis_beta = [], []
    Nij_alpha, Nij_beta = [], []

    for i in range(len(Smo[0])):
        li_alpha = np.trace(np.dot(Smo[0][i], Smo[0][i]))
        li_beta = np.trace(np.dot(Smo[1][i], Smo[1][i]))
        lis_alpha.append(li_alpha)
        lis_beta.append(li_beta)
        Nij_alpha.append(np.trace(Smo[0][i]))
        Nij_beta.append(np.trace(Smo[1][i]))

        for j in range(i + 1, len(Smo[0])):
            if i != j:
                di_alpha = 2 * np.trace(np.dot(Smo[0][i], Smo[0][j]))
                di_beta = 2 * np.trace(np.dot(Smo[1][i], Smo[1][j]))
                dis_alpha.append(di_alpha)
                dis_beta.append(di_beta)

    print(" ----------------------------------------------------------------------------- ")
    print(" |  Atom     N(Sij)     Na(Sij)     Nb(Sij)     Lapl.      dloc_a     dloc_b  ")
    print(" ----------------------------------------------------------------------------- ")

    for i in range(len(Smo[0])):
        print(" | {} {:>2d}   {:10.6f}  {:10.6f}  {:10.6f}   *******   {:8.4f}   {:8.4f} ".format(
                symbols[i], i + 1, Nij_alpha[i] + Nij_beta[i], Nij_alpha[i], Nij_beta[i], lis_alpha[i], lis_beta[i]))
    print(" ----------------------------------------------------------------------------- ")
    print(" | TOT:   {:10.6f}  {:10.6f}  {:10.6f}   *******   {:8.4f}   {:8.4f}".format(
            sum(Nij_alpha) + sum(Nij_beta), sum(Nij_alpha), sum(Nij_beta), sum(lis_alpha), sum(lis_beta)))
    print(" ----------------------------------------------------------------------------- ")
    print(" ------------------------------------------- ")
    print(" |    Pair         DI       DIaa      DIbb ")
    print(" ------------------------------------------- ")

    for i in range(len(Smo[0])):
        for j in range(i + 1, len(Smo[0])):
            print(" | {} {}-{} {}  {:>9.4f} {:>9.4f} {:>9.4f}".format(
                    symbols[i], str(i + 1).rjust(2), symbols[j],
                    str(j + 1).rjust(2), 2 * (np.trace(np.dot(Smo[0][i], Smo[0][j])) + np.trace(np.dot(Smo[1][i], Smo[1][j]))),
                    2 * (np.trace(np.dot(Smo[0][i], Smo[0][j]))), 2 * (np.trace(np.dot(Smo[1][i], Smo[1][j])))))
    print(" ------------------------------------------- ")
    print(" |    TOT:    {:>9.4f} {:>9.4f} {:>9.4f} ".format(
            sum(dis_alpha) + sum(dis_beta) + sum(lis_alpha) + sum(lis_beta), sum(dis_alpha) + sum(lis_alpha), sum(dis_beta) + sum(lis_beta)))
    print(" |    LOC:    {:>9.4f} {:>9.4f} {:>9.4f} ".format(
            sum(lis_alpha) + sum(lis_beta), sum(lis_alpha), sum(lis_beta)))
    print(" |  DELOC:    {:>9.4f} {:>9.4f} {:>9.4f} ".format(
            sum(dis_alpha) + sum(dis_beta), sum(dis_alpha), sum(dis_beta)))
    print(" ------------------------------------------- ")


# POPULATION AND DELOCALIZATION RESTRICTED


def deloc_rest(mol, Smo, molinfo=None):
    """Population analysis, localization and delocalization indices for restricted AOMs.

    Arguments:

       mol: an instance of SCF class
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

    """
    if mol is not None:
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    elif molinfo is not None:
        if isinstance(molinfo, str):
            with open(molinfo, "rb") as f:
                molinfo = np.load(f, allow_pickle=True)
        symbols = molinfo[0]
    else:
        raise NameError(" Could not find 'mol' nor 'molinfo' file")

    # Getting the LIs and DIs
    dis, lis, Nij = [], [], []

    for i in range(len(Smo)):
        li = 2 * np.trace(np.dot(Smo[i], Smo[i]))
        lis.append(li)
        Nij.append(2 * np.trace(Smo[i]))

        for j in range(i + 1, len(Smo)):
            di = 4 * np.trace(np.dot(Smo[i], Smo[j]))
            dis.append(di)
    print(" ------------------------------------------------------- ")
    print(" |  Atom    N(Sij)         Lapl.       loc.       dloc. ")
    print(" ------------------------------------------------------- ")

    for i in range(len(Smo)):
        print(" | {} {:>2d}    {:10.6f}     *******   {:8.4f}   {:8.4f} ".format(
                symbols[i], i + 1, Nij[i], lis[i], Nij[i] - lis[i]))
    print(" ------------------------------------------------------- ")
    print(" | TOT:    {:10.6f}     *******   {:8.4f}   {:8.4f}".format(
            sum(Nij), sum(Nij) - sum(dis), sum(dis)))
    print(" ------------------------------------------------------- ")

    print(" ------------------------ ")
    print(" |    Pair         DI ")
    print(" ------------------------ ")
    for i in range(len(Smo)):
        for j in range(i + 1, len(Smo)):
            print( " | {} {}-{} {}   {:8.4f}".format(
                    symbols[i], str(i + 1).rjust(2), symbols[j],
                    str(j + 1).rjust(2), 4 * np.trace(np.dot(Smo[i], Smo[j]))))
    print(" ------------------------ ")
    print(" |   TOT:      {:8.4f} ".format(np.sum(dis) + np.sum(lis)))
    print(" |   LOC:      {:8.4f} ".format(np.sum(lis)))
    print(" | DELOC:      {:8.4f} ".format(np.sum(dis)))
    print(" ------------------------ ")


########### AROMATICITY STUDIES ###########

# AROMATICITY UNRESTRICTED


def arom_unrest(Smo, rings, partition, mol, mci=False, av1245=False, flurefs=None, homarefs=None, connectivity=None, geom=None, num_threads=None,):
    """Population analysis, localization and delocalization indices and aromaticity indicators
    for unrestricted AOMs.

    Arguments:

       Smo: list of matrices / string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       rings: list
          Contains a list of the indices of the atoms in the ring connectivity for the aromaticity calculations.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

       mol: an instance of SCF class. Default: None
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mci: boolean. Default: None
          Whether to compute the MCI index.

       av1245: boolean. Default: None
          Whether to compute the AV1245 (and AVmin) indices.

       flurefs: dictionary. Default: None
          User-provided references for the Delocalization Indices for the FLU index.

       homarefs: dictionary. Default: None
          User-provided references for the distance and polarizability for the HOMA or HOMER indices.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       geom: list. Default: None
          The molecular coordinates as given by the mol.atom_coords() function.

       num_threads: integer
          Number of threads required for the calculation.
    """

    print(" ----------------------------------------------------------------------")
    print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
    print(" |                     Iring [PCCP 2, 3381 (2000)]")
    print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
    print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
    print(" |                           [PCCP 20, 2787 (2018)]")
    print(" |  For a recent review see: [CSR 44, 6434 (2015)]")
    print(" ----------------------------------------------------------------------")

    # Checking if the list rings is contains more than one ring to analyze

    if not isinstance(rings[0], list):
        rings = [rings]

    # Looping through each of the rings

    for ring_index, ring in enumerate(rings):
        print(" ----------------------------------------------------------------------")
        print(" |")
        print(" | Ring  {} ({}):   {}".format(
                ring_index + 1, len(ring), "  ".join(str(num) for num in ring)))
        print(" |")
        print(" ----------------------------------------------------------------------")

        # Starting the calculation of the aromaticity indicators
        connectivity = [symbols[int(i) - 1] for i in ring]

        if nalpha_equal_nbeta(Smo) == "singlet":
            print(" | Same number of alpha and beta electrons. Computing HOMA")
            if homarefs is not None:
                print(" | Using HOMA references provided by the user")
            else:
                print(" | Using default HOMA references")

            homas = compute_homa(ring, mol, geom=geom, homarefs=homarefs, connectivity=connectivity)
            if homas is None:
                print(" | Connectivity could not match parameters")
            else:
                print(" | EN           {} =  {:>.6f}".format(ring_index + 1, homas[1]))
                print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, homas[2]))
                print(" | HOMA         {} =  {:>.6f}".format(ring_index + 1, homas[0]))
                print(" ----------------------------------------------------------------------")
                blas = compute_bla(ring, mol, geom=geom)

                print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
                print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
                print(" ----------------------------------------------------------------------")

        elif nalpha_equal_nbeta(Smo) == "triplet":
            print(" | Different number of alpha and beta electrons. Computing HOMER")
            if homarefs is not None:
                print(" | Using HOMER references provided by the user")
            else:
                print(" | Using default HOMER references")

            homers = compute_homer(ring, mol, geom=geom, homarefs=homarefs, connectivity=connectivity)

            print(" | EN           {} =  {:>.6f}".format(ring_index + 1, homers[1]))
            print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, homers[2]))
            print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, homers[0]))
            print(" ----------------------------------------------------------------------")
            blas = compute_bla(ring, mol, geom=geom)

            print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
            print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
            print(" ----------------------------------------------------------------------")
        else:
            print(" | No singlet nor triplet. Could not compute HOMA/HOMER")

        print(" ----------------------------------------------------------------------")
        flus_alpha = compute_flu(ring, mol, Smo[0], flurefs, connectivity, partition=partition)
        if flus_alpha is None:
            print(" | Could not compute FLU")
        else:
            flus_beta = compute_flu(ring, mol, Smo[1], flurefs, connectivity, partition=partition)
            if flurefs is not None:
                print(" | Using FLU references provided by the user")
            else:
                print(" | Using the default FLU references")
            print(" | Atoms  :   {}".format("  ".join(str(atom) for atom in connectivity)))

            print(" |")
            print(" | *** FLU_ALPHA ***")
            print(" | FLU_aa       {} =  {:>.6f}".format(ring_index + 1, flus_alpha))
            print(" |")
            print(" | *** FLU_BETA ***")
            print(" | FLU_bb       {} =  {:>.6f}".format(ring_index + 1, flus_beta))
            print(" |")
            print(" | *** FLU_TOTAL ***")
            print(" | FLU          {} =  {:>.6f}".format(ring_index + 1, flus_alpha + flus_beta))
            print(" ----------------------------------------------------------------------")

        boas_alpha = compute_boa(ring, Smo[0])
        boas_beta = compute_boa(ring, Smo[1])

        print(" |")
        print(" | *** BOA_ALPHA ***")
        print(" | BOA_aa       {} =  {:>.6f}".format(ring_index + 1, boas_alpha[0]))
        print(" | BOA_c_aa     {} =  {:>.6f}".format(ring_index + 1, boas_alpha[1]))
        print(" |")
        print(" | *** BOA_BETA ***")
        print(" | BOA_bb       {} =  {:>.6f}".format(ring_index + 1, boas_beta[0]))
        print(" | BOA_c_bb     {} =  {:>.6f}".format(ring_index + 1, boas_beta[1]))
        print(" |")
        print(" | *** BOA_TOTAL ***")
        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, boas_alpha[0] + boas_beta[0]))
        print(" | BOA_c        {} =  {:>.6f}".format(ring_index + 1, boas_alpha[1] + boas_beta[1]))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdis_alpha = compute_pdi(ring, Smo[0])
            pdis_beta = compute_pdi(ring, Smo[1])
            print(" |")
            print(" | *** PDI_ALPHA ***")

            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdis_alpha[1][0]))
            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdis_alpha[1][1]))
            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdis_alpha[1][2]))
            print(" | PDI_alpha     {} =  {:.4f} ".format(ring_index + 1, pdis_alpha[0]))
            print(" |")
            print(" | *** PDI_BETA ***")
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdis_beta[1][0]))
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdis_beta[1][1]))
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdis_beta[1][2]))
            print(" | PDI_beta      {} =  {:.4f} ".format(ring_index + 1, pdis_beta[0]))
            print(" |")
            print(" | *** PDI_TOTAL ***")
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdis_alpha[1][0] + pdis_beta[1][0]))
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdis_alpha[1][1] + pdis_beta[1][1]))
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdis_alpha[1][2] + pdis_beta[1][2]))
            print(" | PDI           {} =  {:.4f} ".format(ring_index + 1, pdis_alpha[0] + pdis_beta[0]))
            print(" ---------------------------------------------------------------------- ")

        if av1245 == True:
            if len(ring) < 6:
                print(" | AV1245 could not be calculated as the number of centers is smaller than 6 ")

            else:
                avs_alpha = np.array(compute_av1245(ring, Smo[0],partition), dtype=object)
                avs_beta = np.array(compute_av1245(ring, Smo[1],partition), dtype=object)

                print(" |")
                print(" | *** AV1245_ALPHA ***")
                for j in range(len(ring)):
                    print( " |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}".format(
                            str(ring[j]).rjust(2), symbols[(ring[j % len(ring)] - 1)],
                            str(ring[(j + 1) % len(ring)]).rjust(2), symbols[(ring[(j + 1) % len(ring)] - 1)],
                            str(ring[(j + 3) % len(ring)]).rjust(2), symbols[(ring[(j + 3) % len(ring)] - 1)],
                            str(ring[(j + 4) % len(ring)]).rjust(2), symbols[(ring[(j + 4) % len(ring)] - 1)],
                            np.array(avs_alpha[2][(ring[j] - 1) % len(ring)])))
                print(" |   AV1245_alpha {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[0]))
                print(" |    AVmin_alpha {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[1]))

                print(" |")
                print(" | *** AV1245_BETA ***")

                for j in range(len(ring)):
                    print( " |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}".format(
                            str(ring[j]).rjust(2), symbols[(ring[j % len(ring)] - 1)],
                            str(ring[(j + 1) % len(ring)]).rjust(2), symbols[(ring[(j + 1) % len(ring)] - 1)],
                            str(ring[(j + 3) % len(ring)]).rjust(2), symbols[(ring[(j + 3) % len(ring)] - 1)],
                            str(ring[(j + 4) % len(ring)]).rjust(2), symbols[(ring[(j + 4) % len(ring)] - 1)],
                            np.array(avs_beta[2][(ring[j] - 1) % len(ring)])))
                print(" |   AV1245_beta  {} =             {:>9.4f}".format(ring_index + 1, avs_beta[0]))
                print(" |    AVmin_beta  {} =             {:>9.4f}".format(ring_index + 1, avs_beta[1]))
                print(" |")
                print(" | *** AV1245_TOTAL ***")
                print(" |   AV1245       {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[0] + avs_beta[0]))
                print(" |    AVmin       {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[1] + avs_beta[1]))
                print(" ---------------------------------------------------------------------- ")

        iring_alpha = np.array(compute_iring(ring, Smo[0]), dtype=object)
        iring_beta = np.array(compute_iring(ring, Smo[1]), dtype=object)
        iring_total = iring_alpha + iring_beta

        print(" | Iring_alpha  {} =  {:>6f}".format(ring_index + 1, iring_alpha))
        print(" | Iring_beta   {} =  {:>6f}".format(ring_index + 1, iring_beta))
        print(" | Iring        {} =  {:>6f}".format(ring_index + 1, iring_total))

        if iring_total < 0:
            print(" | Iring**(1/n) {} =  {:>6f}".format(ring_index + 1, -(np.abs(iring_total) ** (1 / len(ring)))))

        else:
            print(" | Iring**(1/n) {} =  {:>6f}".format(ring_index + 1, iring_total ** (1 / len(ring))))
        print(" ---------------------------------------------------------------------- ")

        if mci == True:
            import time

            if num_threads is None:
                num_threads = 1

            # SINGLE-CORE
            if num_threads == 1:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_alpha = sequential_mci(ring, Smo[0],partition)
                mci_beta = sequential_mci(ring, Smo[1],partition)
                mci_total = mci_alpha + mci_beta
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            # MULTI-CORE
            else:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_alpha = multiprocessing_mci(ring, Smo[0], num_threads,partition)
                mci_beta = multiprocessing_mci(ring, Smo[1], num_threads,partition)
                mci_total = mci_alpha + mci_beta
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))

            print(" ---------------------------------------------------------------------- ")


# AROMATICITY RESTRICTED


def arom_rest(Smo, rings, partition, mol, mci=False, av1245=False, flurefs=None, homarefs=None, connectivity=None, geom=None, num_threads=1):
    """Population analysis, localization and delocalization indices and aromaticity indicators
    for restricted AOMs.

    Arguments:

       Smo: list of matrices / string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       rings: list
          Contains a list of the indices of the atoms in the ring connectivity for the aromaticity calculations.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

       mol: an instance of SCF class. Default: None
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mci: boolean. Default: None
          Whether to compute the MCI index.

       av1245: boolean. Default: None
          Whether to compute the AV1245 (and AVmin) indices.

       flurefs: dictionary. Default: None
          User-provided references for the Delocalization Indices for the FLU index.

       homarefs: dictionary. Default: None
          User-provided references for the distance and polarizability for the HOMA or HOMER indices.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       geom: list. Default: None
          The molecular coordinates as given by the mol.atom_coords() function.

       num_threads: integer
          Number of threads required for the calculation.
    """

    print(" ----------------------------------------------------------------------")
    print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
    print(" |                     Iring [PCCP 2, 3381 (2000)]")
    print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
    print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
    print(" |                           [PCCP 20, 2787 (2018)]")
    print(" |  For a recent review see: [CSR 44, 6434 (2015)]")

    # Checking if the list rings is contains more than one ring to analyze

    if not isinstance(rings[0], list):
        rings = [rings]

    # Looping through each of the rings

    for ring_index, ring in enumerate(rings):
        print(" ----------------------------------------------------------------------")
        print(" ----------------------------------------------------------------------")

        print(" |")
        print(" | Ring  {} ({}):   {}".format(ring_index + 1, len(ring), "  ".join(str(num) for num in ring)))
        print(" |")
        print(" ----------------------------------------------------------------------")
        connectivity = [symbols[int(i) - 1] for i in ring]
        if homarefs is not None:
            print(" | Using HOMA references provided by the user")
        else:
            print(" | Using default HOMA references")
        homas = compute_homa(ring, mol, geom=geom, homarefs=homarefs, connectivity=connectivity)
        if homas is None:
            print(" | Connectivity could not match parameters")
        else:
            print(" | EN           {} =  {:>.6f}".format(ring_index + 1, homas[1]))
            print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, homas[2]))
            print(" | HOMA         {} =  {:>.6f}".format(ring_index + 1, homas[0]))
            print( " ----------------------------------------------------------------------")

            blas = compute_bla(ring, mol, geom=geom)

            print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
            print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
            print(" ----------------------------------------------------------------------")

        print(" ----------------------------------------------------------------------")

        flus = compute_flu(ring, mol, Smo, flurefs, connectivity, partition=partition)
        if flus is None:
            print(" | Could not compute FLU")
        else:
            if flurefs is not None:
                print(" | Using FLU references provided by the user")
            else:
                print(" | Using the default FLU references")
            print(" | Atoms  :   {}".format("  ".join(str(atom) for atom in connectivity)))
            print(" |")
            print(" | FLU          {} =  {:>.6f}".format(ring_index + 1, flus))
            print(" ----------------------------------------------------------------------")

        boas = compute_boa(ring, Smo)

        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, boas[0]))
        print(" | BOA_cc       {} =  {:>.6f}".format(ring_index + 1, boas[1]))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdis = 2 * np.array(compute_pdi(ring, Smo), dtype=object)
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[0], ring[3], 2 * pdis[1][0]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[1], ring[4], 2 * pdis[1][1]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[2], ring[5], 2 * pdis[1][2]))
            print(" | PDI          {} =  {:.4f} ".format(ring_index + 1, pdis[0]))
        print(" ----------------------------------------------------------------------")
        print(" ----------------------------------------------------------------------")

        if av1245 == True:
            if len(ring) < 6:
                print(" | AV1245 could not be calculated as the number of centers is smaller than 6 ")

            else:
                avs = 2 * np.array(compute_av1245(ring, Smo,partition), dtype=object)
                av1245_pairs = [(ring[i % len(ring)],ring[(i + 1) % len(ring)],ring[(i + 3) % len(ring)],ring[(i + 4) % len(ring)])
                    for i in range(len(ring)) ]

                for j in range(len(ring)):
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}".format(
                            str(ring[j]).rjust(2), symbols[av1245_pairs[j][0] - 1].ljust(2),
                            str(ring[(j + 1) % len(ring)]).rjust(2), symbols[av1245_pairs[j][1] - 1].ljust(2),
                            str(ring[(j + 3) % len(ring)]).rjust(2), symbols[av1245_pairs[j][2] - 1].ljust(2),
                            str(ring[(j + 4) % len(ring)]).rjust(2), symbols[av1245_pairs[j][3] - 1].ljust(2),
                            2 * avs[2][(ring[j] - 1) % len(ring)]))
                print(" | AV1245 {} =             {:.4f}".format(ring_index + 1, avs[0]))
                print(" |  AVmin {} =             {:.4f}".format(ring_index + 1, avs[1]))
                print(" ---------------------------------------------------------------------- ")

        iring_total = 2 * compute_iring(ring, Smo)
        print(" | Iring        {} =  {:>.6f}".format(ring_index + 1, iring_total))

        if iring_total < 0:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, -(np.abs(iring_total) ** (1 / len(ring)))))

        else:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, iring_total ** (1 / len(ring))))

        print(" ---------------------------------------------------------------------- ")

        if mci == True:
            import time

            if num_threads is None:
                num_threads = 1

            # SINGLE-CORE
            if num_threads == 1:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_total = 2 * sequential_mci(ring, Smo,partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            # MULTI-CORE
            else:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_total = 2 * multiprocessing_mci(ring, Smo, num_threads,partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))
        print(" ---------------------------------------------------------------------- ")


def arom_unrest_from_aoms(Smo, rings, partition, mol, mci=False, av1245=False, flurefs=None, homarefs=None, connectivity=None, geom=None, num_threads=None):
    """Population analysis, localization and delocalization indices and aromaticity indicators
    for previously saved restricted AOMs.

    Arguments:

       Smo: list of matrices / string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       rings: list
          Contains a list of the indices of the atoms in the ring connectivity for the aromaticity calculations.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

       mol: an instance of SCF class. Default: None
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mci: boolean. Default: None
          Whether to compute the MCI index.

       av1245: boolean. Default: None
          Whether to compute the AV1245 (and AVmin) indices.

       flurefs: dictionary. Default: None
          User-provided references for the Delocalization Indices for the FLU index.

       homarefs: dictionary. Default: None
          User-provided references for the distance and polarizability for the HOMA or HOMER indices.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       geom: list. Default: None
          The molecular coordinates as given by the mol.atom_coords() function.

       num_threads: integer
          Number of threads required for the calculation.
    """

    print(" ----------------------------------------------------------------------")
    print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
    print(" |                     Iring [PCCP 2, 3381 (2000)]")
    print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
    print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
    print(" |                           [PCCP 20, 2787 (2018)]")
    print(" |  For a recent review see: [CSR 44, 6434 (2015)]")
    print(" ----------------------------------------------------------------------")

    # Checking if the list rings is contains more than one ring to analyze

    if not isinstance(rings[0], list):
        rings = [rings]

    # Looping through each of the rings

    for ring_index, ring in enumerate(rings):
        print(" ----------------------------------------------------------------------")
        print(" |")
        print(" | Ring  {} ({}):   {}".format(ring_index + 1, len(ring), "  ".join(str(num) for num in ring)))
        print(" |")
        print(" ----------------------------------------------------------------------")
        if connectivity is None:
            if symbols is not None:
                connectivity = [symbols[int(i) - 1] for i in ring]
            else:
                connectivity = None

        if connectivity is None:
            print(" | Connectivity not found. Could not compute geometric indices")
        else:
            if nalpha_equal_nbeta(Smo) == "singlet":
                print(" | Same number of alpha and beta electrons. Computing HOMA")
                if homarefs is not None:
                    print(" | Using HOMA references provided by the user")
                else:
                    print(" | Using default HOMA references")

                homas = compute_homa(ring, mol, geom=geom, homarefs=homarefs, connectivity=connectivity)
                if homas is None:
                    print(" | Connectivity could not match parameters")
                else:
                    print(" | EN           {} =  {:>.6f}".format(ring_index + 1, homas[1]))
                    print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, homas[2]))
                    print(" | HOMA         {} =  {:>.6f}".format(ring_index + 1, homas[0]))
                    print(" ----------------------------------------------------------------------")
                    blas = compute_bla(ring, mol, geom=geom)

                    print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
                    print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
                    print(" ----------------------------------------------------------------------")

            elif nalpha_equal_nbeta(Smo) == "triplet":
                print(" | Different number of alpha and beta electrons. Computing HOMER")
                if homarefs is not None:
                    print(" | Using HOMER references provided by the user")
                else:
                    print(" | Using default HOMER references")

                homers = compute_homer(ring, mol, geom=geom, homarefs=homarefs, connectivity=connectivity)
                if homers is None:
                    print(" | Connectivity could not match parameters")
                else:
                    print(" | EN           {} =  {:>.6f}".format(ring_index + 1, homers[1]))
                    print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, homers[2]))
                    print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, homers[0]))
                    print(" ----------------------------------------------------------------------")
                    blas = compute_bla(ring, mol, geom=geom)

                    print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
                    print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
                    print(" ----------------------------------------------------------------------")
            else:
                print(" | No singlet nor triplet. Could not compute HOMA/HOMER")

        print(" ----------------------------------------------------------------------")

        if connectivity is not None:
            if isinstance(connectivity[0], str) and mol is None and mol_info is None:
                if connectivity[ring_index-1] is None:
                    print(ring_index)
                    print(" | If no 'mol' nor 'molinfo', only one connectivity can be given")
            flus_alpha = compute_flu(ring, mol, Smo[0], flurefs, connectivity, partition=partition)
            if flus_alpha is None:
                print(" | Could not compute FLU")
            else:
                flus_beta = compute_flu(ring, mol, Smo[1], flurefs, connectivity, partition=partition)
                if flurefs is not None:
                    print(" | Using FLU references provided by the user")
                else:
                    print(" | Using the default FLU references")
                print(" | Atoms  :   {}".format("  ".join(str(atom) for atom in connectivity)))
                print(" |")
                print(" | *** FLU_ALPHA ***")
                print(" | FLU_aa       {} =  {:>.6f}".format(ring_index + 1, flus_alpha))
                print(" |")
                print(" | *** FLU_BETA ***")
                print(" | FLU_bb       {} =  {:>.6f}".format(ring_index + 1, flus_beta))
                print(" |")
                print(" | *** FLU_TOTAL ***")
                print(" | FLU          {} =  {:>.6f}".format(ring_index + 1, flus_alpha + flus_beta))
        print(" ----------------------------------------------------------------------")

        boas_alpha = compute_boa(ring, Smo[0])
        boas_beta = compute_boa(ring, Smo[1])

        print(" |")
        print(" | *** BOA_ALPHA ***")
        print(" | BOA_aa       {} =  {:>.6f}".format(ring_index + 1, boas_alpha[0]))
        print(" | BOA_c_aa     {} =  {:>.6f}".format(ring_index + 1, boas_alpha[1]))
        print(" |")
        print(" | *** BOA_BETA ***")
        print(" | BOA_bb       {} =  {:>.6f}".format(ring_index + 1, boas_beta[0]))
        print(" | BOA_c_bb     {} =  {:>.6f}".format(ring_index + 1, boas_beta[1]))
        print(" |")
        print(" | *** BOA_TOTAL ***")
        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, boas_alpha[0] + boas_beta[0]))
        print(" | BOA_c        {} =  {:>.6f}".format(ring_index + 1, boas_alpha[1] + boas_beta[1]))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdis_alpha = compute_pdi(ring, Smo[0])
            pdis_beta = compute_pdi(ring, Smo[1])
            print(" |")
            print(" | *** PDI_ALPHA ***")

            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdis_alpha[1][0]))
            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdis_alpha[1][1]))
            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdis_alpha[1][2]))
            print(" | PDI_alpha     {} =  {:.4f} ".format(ring_index + 1, pdis_alpha[0]))
            print(" |")
            print(" | *** PDI_BETA ***")
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdis_beta[1][0]))
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdis_beta[1][1]))
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdis_beta[1][2]))
            print(" | PDI_beta      {} =  {:.4f} ".format(ring_index + 1, pdis_beta[0]))
            print(" |")
            print(" | *** PDI_TOTAL ***")
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdis_alpha[1][0] + pdis_beta[1][0]))
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdis_alpha[1][1] + pdis_beta[1][1]))
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdis_alpha[1][2] + pdis_beta[1][2]))
            print(" | PDI           {} =  {:.4f} ".format(ring_index + 1, pdis_alpha[0] + pdis_beta[0]))
            print(" ---------------------------------------------------------------------- ")

        if av1245 == True:
            if len(ring) < 6:
                print(" | AV1245 could not be calculated as the number of centers is smaller than 6 ")

            else:
                avs_alpha = np.array(compute_av1245(ring, Smo[0],partition), dtype=object)
                avs_beta = np.array(compute_av1245(ring, Smo[1],partition), dtype=object)

                print(" |")
                print(" | *** AV1245_ALPHA ***")
                for j in range(len(ring)):
                    print(" |   A {} -  A {} -  A {} -  A {}  |  {:>9.4f}".format(
                            str(ring[j]).rjust(2), str(ring[(j + 1) % len(ring)]).rjust(2),
                            str(ring[(j + 3) % len(ring)]).rjust(2), str(ring[(j + 4) % len(ring)]).rjust(2),
                            np.array(avs_alpha[2][(ring[j] - 1) % len(ring)])))
                print(" |   AV1245_alpha {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[0]))
                print(" |    AVmin_alpha {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[1]))

                print(" |")
                print(" | *** AV1245_BETA ***")

                for j in range(len(ring)):
                    print( " |   A {} -  A {} -  A {} -  A {}  |  {:>9.4f}".format(
                            str(ring[j]).rjust(2), str(ring[(j + 1) % len(ring)]).rjust(2),
                            str(ring[(j + 3) % len(ring)]).rjust(2), str(ring[(j + 4) % len(ring)]).rjust(2),
                            np.array(avs_beta[2][(ring[j] - 1) % len(ring)])))
                print(" |   AV1245_beta  {} =             {:>9.4f}".format( ring_index + 1, avs_beta[0]))

                print(" |")
                print(" | *** AV1245_TOTAL ***")
                print(" |   AV1245       {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[0] + avs_beta[0]))
                print(" |    AVmin       {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[1] + avs_beta[1]))
                print(" ---------------------------------------------------------------------- ")

        iring_alpha = compute_iring(ring, Smo[0])
        iring_beta = compute_iring(ring, Smo[1])
        iring_total = iring_alpha + iring_beta

        print(" | Iring_alpha  {} =  {:>6f}".format(ring_index + 1, iring_alpha))
        print(" | Iring_beta   {} =  {:>6f}".format(ring_index + 1, iring_beta))
        print(" | Iring        {} =  {:>6f}".format(ring_index + 1, iring_total))

        if iring_total < 0:
            print(" | Iring**(1/n) {} =  {:>6f}".format(ring_index + 1, -(np.abs(iring_total) ** (1 / len(ring)))))

        else:
            print(" | Iring**(1/n) {} =  {:>6f}".format(ring_index + 1, iring_total ** (1 / len(ring))))
        print(" ---------------------------------------------------------------------- ")

        if mci == True:
            import time

            if num_threads is None:
                num_threads = 1

            # SINGLE-CORE
            if num_threads == 1:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_alpha = sequential_mci(ring, Smo[0],partition)
                mci_beta = sequential_mci(ring, Smo[1],partition)
                end_mci = time.time()
                mci_total = mci_alpha + mci_beta
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            # MULTI-CORE
            else:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_alpha = multiprocessing_mci(ring, Smo[0], num_threads,partition)
                mci_beta = multiprocessing_mci(ring, Smo[1], num_threads,partition)
                mci_total = mci_alpha + mci_beta
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))
            print(" ---------------------------------------------------------------------- ")


# AROMATICITY RESTRICTED


def arom_rest_from_aoms(Smo, rings, partition, mol, mci=True, av1245=True, flurefs=None, homarefs=None, connectivity=None, geom=None, num_threads=1):
    """Population analysis, localization and delocalization indices and aromaticity indicators
    for previously saved restricted AOMs.

    Arguments:

       Smo: list of matrices / string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       rings: list
          Contains a list of the indices of the atoms in the ring connectivity for the aromaticity calculations.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

       mol: an instance of SCF class. Default: None
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mci: boolean. Default: None
          Whether to compute the MCI index.

       av1245: boolean. Default: None
          Whether to compute the AV1245 (and AVmin) indices.

       flurefs: dictionary. Default: None
          User-provided references for the Delocalization Indices for the FLU index.

       homarefs: dictionary. Default: None
          User-provided references for the distance and polarizability for the HOMA or HOMER indices.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       geom: list. Default: None
          The molecular coordinates as given by the mol.atom_coords() function.

       num_threads: integer
          Number of threads required for the calculation.
    """

    print(" ----------------------------------------------------------------------")
    print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
    print(" |                     Iring [PCCP 2, 3381 (2000)]")
    print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
    print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
    print(" |                           [PCCP 20, 2787 (2018)]")
    print(" |  For a recent review see: [CSR 44, 6434 (2015)]")
    print(" ----------------------------------------------------------------------")

    # Checking if the list rings is contains more than one ring to analyze

    if not isinstance(rings[0], list):
        rings = [rings]

    # Looping through each of the rings

    for ring_index, ring in enumerate(rings):
        print(" ----------------------------------------------------------------------")
        print(" |")
        print(" | Ring  {} ({}):   {}".format(ring_index + 1, len(ring), "  ".join(str(num) for num in ring)))
        print(" |")
        print(" ----------------------------------------------------------------------")

        # Starting the calculation of the aromaticity indicators
        if connectivity is None:
            if symbols is None:
                print(" | Connectivity not found. Could not compute geometric indices")
            else:
                connectivity = [symbols[int(i) - 1] for i in ring]
        else:
            if isinstance(connectivity[0], str) and mol is None and mol_info is None:
                print(" | If no 'mol' nor 'molinfo', only one connectivity can be given")
            else:
                homas = compute_homa(ring, mol, geom=geom, homarefs=homarefs, connectivity=connectivity)
                if homas is None:
                    print(" | Connectivity could not match parameters")
                else:
                    print(" ----------------------------------------------------------------------")
                    if homarefs is not None:
                        print(" | Using HOMA references provided by the user")
                    elif homarefs is None:
                        print(" | Using default HOMA references")
                    print(" | EN           {} =  {:>.6f}".format(ring_index + 1, homas[1]))
                    print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, homas[2]))
                    print(" | HOMA         {} =  {:>.6f}".format(ring_index + 1, homas[0]))
                    print(" ----------------------------------------------------------------------")
                    blas = compute_bla(ring, mol, geom=geom)

                    print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
                    print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
                    print(" ----------------------------------------------------------------------")

        print(" ----------------------------------------------------------------------")

        if connectivity is not None:
            if isinstance(connectivity[0], str) and mol is None and mol_info is None:
                print(" | If no 'mol' nor 'molinfo', only one connectivity can be given")
            else:
                flus = compute_flu(ring, mol, Smo, flurefs, connectivity, partition=partition)
                if flus is None:
                    print(" | Could not compute FLU")
                    print(" ----------------------------------------------------------------------")
                else:
                    if flurefs is not None:
                        print(" | Using FLU references provided by the user")
                    else:
                        print(" | Using the default FLU references")
                    print(" | Atoms  :   {}".format("  ".join(str(atom) for atom in connectivity)))
                    print(" |")
                    print(" | FLU          {} =  {:>.6f}".format(ring_index + 1, flus))
                    print(" ----------------------------------------------------------------------")

        boas = compute_boa(ring, Smo)

        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, boas[0]))
        print(" | BOA_cc       {} =  {:>.6f}".format(ring_index + 1, boas[1]))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdis = 2 * np.array(compute_pdi(ring, Smo), dtype=object)
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[0], ring[3], 2 * pdis[1][0]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[1], ring[4], 2 * pdis[1][1]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[2], ring[5], 2 * pdis[1][2]))
            print(" | PDI          {} =  {:.4f} ".format(ring_index + 1, pdis[0]))
        print(" ----------------------------------------------------------------------")
        print(" ----------------------------------------------------------------------")

        if av1245 == True:
            if len(ring) < 6:
                print(" | AV1245 could not be calculated as the number of centers is smaller than 6 ")

            else:
                avs = 2 * np.array(compute_av1245(ring, Smo,partition), dtype=object)
                av1245_pairs = [( ring[i % len(ring)], ring[(i + 1) % len(ring)], ring[(i + 3) % len(ring)], ring[(i + 4) % len(ring)])
                    for i in range(len(ring)) ]

                for j in range(len(ring)):
                    print(" |   A {} -  A {} -  A {} -  A {}  |  {:>6.4f}".format(
                            str(ring[j]).rjust(2), str(ring[(j + 1) % len(ring)]).rjust(2),
                            str(ring[(j + 3) % len(ring)]).rjust(2), str(ring[(j + 4) % len(ring)]).rjust(2),
                            2 * avs[2][(ring[j] - 1) % len(ring)]))
                print(" | AV1245 {} =             {:.4f}".format(ring_index + 1, avs[0]))
                print(" |  AVmin {} =             {:.4f}".format(ring_index + 1, avs[1]))
                print(" ---------------------------------------------------------------------- ")

        iring_total = 2 * compute_iring(ring, Smo)
        print(" | Iring        {} =  {:>.6f}".format(ring_index + 1, iring_total))

        if iring_total < 0:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, -(np.abs(iring_total) ** (1 / len(ring)))))

        else:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, iring_total ** (1 / len(ring))))
        print(" ---------------------------------------------------------------------- ")

        if mci == True:
            import time

            if num_threads is None:
                num_threads = 1

            # SINGLE-CORE
            if num_threads == 1:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_total = 2 * sequential_mci(ring, Smo,partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            # MULTI-CORE
            else:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_total = 2 * sequential_mci(ring, Smo,partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))

        print(" ---------------------------------------------------------------------- ")


def rest_or_unrest(Smo):
    """Checks the topology of the AOMs to determine whether it is restricted or unrestricted.

    Arguments:

       Smo: list of matrices / string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.
    Returns:

       "rest" for restricted AOMs, "unrest" for unrestricted AOMs.
    """

    # Checking if the list rings is contains more than one ring to analyze
    if isinstance(Smo[0][0][0], float):
        return "rest"
    elif (Smo[0][0][0], list):
        return "unrest"
    else:
        raise NameError("Could not find the type of wave function from the AOMs")


def nalpha_equal_nbeta(Smo):
    """Checks the topology of the AOMs for alpha and beta are the same shape.

    Arguments:
       Smo: list of matrices
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

    Returns:
       "singlet" if the number of alpha and beta electrons is the same or "notsinglet" if it is different.
    """

    if len(Smo[0][0]) == len(Smo[1][0]):
        return "singlet"
    elif len(Smo[0][0]) == (len(Smo[1][0]) + 2):
        return "triplet"
    else:
        return None


def load_aoms(Smo):
    """Loads the AOMs from a Smo file.

    Arguments:
       Smo: string
          Contains the name or the path containing the AOMs.

    Returns:
       Smo: list of matrices
          The AOMs required for the ESIpy code.
    """
    with open(Smo, "rb") as f:
        Smo = np.load(f, allow_pickle=True)
    return Smo

def save_aoms(Smo, name):
    """Saves the AOMs in a Smo file.

    Arguments:
       Smo: List of matrices
          Contains the AOMs.
       name: String
          Contains the name of the file to save the AOMs.
    """
with open(name, "wb") as f:
    np.save(f, Smo)

##################################################################
########### COMPUTATION OF THE AROMATICITY DESCRIPTORS ###########
##################################################################

########## Iring ###########

# Computing the Iring (Restricted and Unrestricted)


def compute_iring(arr, Smo):
    """Calculation of the Iring aromaticity index

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

    Returns:
       iring: float
          The Iring for the given ring connectivity

    """

    product = np.identity(Smo[0].shape[0])
    for i in arr:
        product = np.dot(product, Smo[i - 1])
    iring = 2 ** (len(arr) - 1) * np.trace(product)

    return iring


########### MCI ###########


def sequential_mci(arr, Smo,part):
    """Computes the MCI sequentially by recursively generating all the permutations using Heaps'
    algorithm and computing the Iring for each without storing them. Does not have memory
    requirements and is the default for ESIpy if no number of threads are specified.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring onnectivity.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

    Returns:
       mci_value: float
          MCI value for the given ring.
    """

    from math import factorial
    from itertools import permutations, islice

    if part == 'mulliken':
      iterable2 = islice(permutations(arr), factorial(len(arr)-1))
    else: # remove reversed permutations
      iterable = islice(permutations(arr), factorial(len(arr)-1)-factorial(len(arr)-2))
      iterable2 = (x for x in iterable if x[0]== arr[0] and x[1] < x[-1])

    mci_value = sum(compute_iring(p, Smo) for p in iterable2 )
    return mci_value


def multiprocessing_mci(arr, Smo, num_threads, partition):

    """Computes the MCI split in different threads by generating all the permutations
    for a later distribution along the specified number of threads.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       num_threads: integer
          Number of threads required for the calculation.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

    Returns:
       mci_value: float
          MCI value for the given ring.
    """

    from multiprocessing import Pool
    from math import factorial
    from functools import partial
    from itertools import permutations, islice

    pool=Pool(processes=num_threads)
    dumb=partial(compute_iring,Smo=Smo)
    chunk_size=50000

    if partition == 'mulliken':
      iterable2 = islice(permutations(arr), factorial(len(arr)-1))
    else: # remove reversed permutations
      iterable = islice(permutations(arr), factorial(len(arr)-1)-factorial(len(arr)-2))
      iterable2 = (x for x in iterable if x[0]== arr[0] and x[1] < x[-1])

    results=pool.imap(dumb, iterable2,chunk_size)
    trace = sum(results)

    return trace


########### AV1245 ###########

# Calculation of the AV1245 index (Restricted and Unrestricted)


def compute_av1245(arr, Smo, partition):
    """Computes the AV1245 index by generating the pair of indices that fulfill the 1-2-4-5 condition
    for a latter calculation of each MCI. Computes the AVmin as the minimum absolute value the index
    can get. It is not computed for rings smaller than n=6.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       Smo: list of matrices
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       partition: string. Default: None
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

    Returns:
       tuple
          The tuple contains the AV1245 index, the AVmin index and each of the AV1245 in a list for the output, respectively.
    """

    def av1245_pairs(arr):
        return [(arr[i % len(arr)], arr[(i + 1) % len(arr)], arr[(i + 3) % len(arr)], arr[(i + 4) % len(arr)])
            for i in range(len(arr)) ]

    min_product, av1245_value, avmin_value = 0, 0, 0
    val = 0
    product = 0
    products = []

    for cp in av1245_pairs(arr):
        product = sequential_mci(list(cp), Smo, partition)
        products.append(1000 * product / 3)

    min_product = min(products, key=abs)
    av1245_value = np.mean(products)
    avmin_value = min_product

    return av1245_value, avmin_value, products


########### PDI ###########

# Calculation of the PDI (Restricted and Unrestricted)


def compute_pdi(arr, Smo):
    """Computes the PDI as the average of the DIs in para-position in the 6-membered ring.
    It is only computed for rings of n=6.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       Smo: list of matrices
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

    Returns:
       list
          The list contains the PDI value and each of the DIs in para position.
    """

    if len(arr) == 6:
        pdi_a = 2 * np.trace(np.dot(Smo[arr[0] - 1], Smo[arr[3] - 1]))
        pdi_b = 2 * np.trace(np.dot(Smo[arr[1] - 1], Smo[arr[4] - 1]))
        pdi_c = 2 * np.trace(np.dot(Smo[arr[2] - 1], Smo[arr[5] - 1]))
        pdi_value = (pdi_a + pdi_b + pdi_c) / 3

        return pdi_value, [pdi_a, pdi_b, pdi_c]

    else:
        return None


# Calculation of auxiliary information


def find_distances(arr, mol, geom):
    distances = []
    for i in range(len(arr)):

        if mol is None:
            coord1 = geom[arr[i] - 1]
            coord2 = geom[arr[(i + 1) % len(arr)] - 1]
            distances.append(np.linalg.norm(coord1 - coord2) * 0.529177249)  # From Bohr to Angstrom
        else:
            coord1 = mol.atom_coords()[arr[i] - 1]
            coord2 = mol.atom_coords()[arr[(i + 1) % len(arr)] - 1]
            distances.append(np.linalg.norm(coord1 - coord2) * 0.529177249)  # From Bohr to Angstrom
    return distances


def find_dis(arr, Smo):
    return [4 * np.trace(np.dot(Smo[arr[i] - 1], Smo[arr[(i + 1) % len(arr)] - 1])) for i in range(len(arr)) ]


def find_lis(arr, Smo):
    return [2 * np.trace(np.dot(Smo[arr[i] - 1], Smo[arr[i] - 1])) for i in range(len(arr)) ]

def find_ns(arr, Smo):
    return [2 * np.trace(Smo[arr[i] - 1]) for i in range(len(arr))]


########### FLU ###########

# Calculation of the FLU (Restricted and Unrestricted)


def find_flurefs(partition=None):
    """Sets the reference of the FLU index based on the provided partition.
    The available options are "CC" from benzene, "CN" from pyridine,
    "BN" from borazine, "NN" from pyridazine and "CS" from thiophene,
    all obtained from optimized and single-point calculations at HF/6-31G(d)
    level of theory

    Arguments:
       partition: string.
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

    Returns:
       flurefs: dictionary
          Contains the reference DI for each bond.
    """

    if partition == "qtaim":
        flurefs = {"CC": 1.3993, "CN": 1.1958, "BN": 0.3934, "NN": 1.5252, "CS": 1.2369}

    elif partition == "mulliken":
        flurefs = {"CC": 1.4530, "CN": 1.4149, "BN": 1.0944, "NN": 1.3044, "CS": 1.1024}

    elif partition == "lowdin":

        flurefs = {"CC": 1.5000, "CN": 1.6257, "BN": 1.6278, "NN": 1.5252, "CS": 1.2675}
    elif partition == "meta_lowdin":
        flurefs = {"CC": 1.4394, "CN": 1.4524, "BN": 1.3701, "NN": 1.5252, "CS": 1.1458}

    elif partition == "nao":
        flurefs = {"CC": 1.4338, "CN": 1.4117, "BN": 0.9238, "NN": 1.3706, "CS": 1.1631}

    elif partition == "iao":
        flurefs = {"CC": 1.4378, "CN": 1.4385, "BN": 1.1638, "NN": 1.3606, "CS": 1.1436}
    return flurefs


def compute_flu(arr, mol, Smo, flurefs=None, connectivity=None, partition=None):
    """Computes the FLU index from references from the find_flurefs() function.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       mol: an instance of SCF class. Default: None
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       flurefs: dictionary. Default: None
          User-provided references for the Delocalization Indices for the FLU index.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       partition: string.
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

    Returns:
       flu_value: float
          Value of the FLU index.
    """

    dis = []
    flu_value, flu_polar = 0, 0
    if partition is None:
        print(" | No partition provided. Could not find proper references")
        return None
    elif mol is None:
        print(" | No mol object provided. Using the connectivity provided by the user")
        if connectivity is None:
            print(" | No FLU connectivity provided by the user")
            return None
        else:
            atom_symbols = connectivity
            bond_types = ["".join(sorted([atom_symbols[i], atom_symbols[(i + 1) % len(arr)]]))
                for i in range(len(arr)) ]
    else:
        atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        bond_types = ["".join( sorted([atom_symbols[arr[i] - 1], atom_symbols[arr[(i + 1) % len(arr)] - 1]]))
            for i in range(len(arr)) ]

    # Setting and update of the reference values
    flu_refs = find_flurefs(partition)
    if flurefs is not None:
        flu_refs.update(flurefs)

    dis = find_dis(arr, Smo)
    lis = find_lis(arr, Smo)
    ns = find_ns(arr, Smo)
    for i in range(len(arr)):
        if bond_types[i] not in flu_refs:
            print(f"No parameters found for bond type {bond_types[i]}")
            return None

        flu_deloc = (dis[i] - flu_refs[bond_types[i]]) / flu_refs[bond_types[i]]
        a_to_b = dis[i] / 2 * (ns[i] - lis[i])
        b_to_a = dis[i] / 2 * (ns[(i + 1) % len(arr)] - lis[(i + 1) % len(arr)])
        flu_polar = a_to_b / b_to_a

        if flu_polar < 1:
            flu_polar = 1 / flu_polar

        flu_value += float(flu_deloc * flu_polar) ** 2
    return flu_value / len(arr)


########### BOA ###########

# Calculation of the BOA (Restricted and Unrestricted)


def compute_boa(arr, Smo):
    """Computes the BOA and BOA_c indices.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

    Returns:
       list
          Contains the BOA and the BOA_c indices, respectively.
    """

    n1 = len([i for i in arr if i % 2 != 0])
    n2 = len([i for i in arr if i % 2 == 1])

    def find_di(Smo, i, j):
        return 4 * np.trace(np.dot(Smo[i - 1], Smo[j - 1]))

    sum_odd = sum(find_di(Smo, arr[i - 1], arr[i]) for i in range(0, len(arr), 2))
    sum_even = sum(find_di(Smo, arr[i + 1], arr[i]) for i in range(0, len(arr) - 1, 2))
    boa = abs(sum_odd / n1 - sum_even / n2)

    boa_c = 0
    for i in range(len(arr)):
        diff_di = abs(find_di(Smo, arr[i - 1], arr[i]) - find_di(Smo, arr[(i + 1) % len(arr) - 1], arr[(i + 1) % len(arr)]))
        boa_c += diff_di / len(arr)
    return boa, boa_c


######## GEOMETRIC INDICES ########

# Setting of the parameters for the HOMA calculation (Restricted and Unrestricted)


def compute_homa(arr, mol, geom=None, homarefs=None, connectivity=None):
    """Sets the references for the HOMA index and calls the function that computes the index.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       mol: an instance of SCF class
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       geom: list
          The molecular coordinates as given by the mol.atom_coords() function.

       homarefs: dictionary
          User-provided references for the distance and polarizability for the HOMA or HOMER indices.

       connectivity: list
          The atomic symbols of the atoms in the ring in 'mol' order.

    Returns:
       list
          Contains the HOMA value as well as the EN and GEO components.
    """

    refs = {
        "CC": {"r_opt": 1.388, "alpha": 257.7},
        "CN": {"r_opt": 1.334, "alpha": 93.52},
        "NN": {"r_opt": 1.309, "alpha": 130.33},
        "CO": {"r_opt": 1.265, "alpha": 157.38},
    }
    if homarefs is not None:
        refs.update(homarefs)
    return make_homaer(arr, mol, geom=geom, refs=refs, connectivity=connectivity)


# Setting of the parameters for the HOMER calculation (Restricted and Unrestricted)


def compute_homer(arr, mol, geom=None, homarefs=None, connectivity=None):
    """Sets the references for the HOMER index and calls the function that computes the index.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       mol: an instance of SCF class.
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       geom: list
          The molecular coordinates as given by the mol.atom_coords() function.

       homarefs: dictionary
          User-provided references for the distance and polarizability for the HOMER index.

       connectivity: list
          The atomic symbols of the atoms in the ring in 'mol' order.

    Returns:
       list
          Contains the HOMA value as well as the EN and GEO components.
    """

    refs = {
        "CC": {"r_opt": 1.437, "alpha": 950.74},
        "CN": {"r_opt": 1.390, "alpha": 506.43},
        "NN": {"r_opt": 1.375, "alpha": 187.36},
        "CO": {"r_opt": 1.379, "alpha": 164.96},
    }

    if homarefs is not None:
        refs.update(homarefs)
    return make_homaer(arr, mol, geom=geom, refs=refs, connectivity=connectivity)


# Actual calculation of the HOMA and/or HOMER indices (Restricted and Unrestricted)


def make_homaer(arr, mol, geom, refs, connectivity):

    if mol is None:
        print(" | No mol object provided. Using the data provided by the user")
        if connectivity is None:
            print(" | No HOMA connectivity provided by the user")
            return None
        if geom is None:
            print(" | No geometry provided by the user")
            return None
        else:
            atom_symbols = connectivity
            bond_types = ["".join(sorted([atom_symbols[i], atom_symbols[(i + 1) % len(arr)]])) for i in range(len(arr)) ]
    else:
        atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        bond_types = ["".join(sorted([atom_symbols[arr[i] - 1], atom_symbols[arr[(i + 1) % len(arr)] - 1]]))
            for i in range(len(arr)) ]

    for i in range(len(arr)):
        if bond_types[i] not in refs:
            print(f"No parameters found for bond type {bond_types[i]}")
            return None

    distances = find_distances(arr, mol, geom)

    EN, GEO = 0, 0
    for i in range(len(arr)):
        EN += np.mean(refs[bond_types[i]]["alpha"] * (refs[bond_types[i]]["r_opt"] - np.mean(distances)) ** 2) / len(arr)
    GEO = np.mean(refs[bond_types[i]]["alpha"] * np.sum((distances - np.mean(distances)) ** 2) / len(arr))

    homa_value = 1 - (EN + GEO)

    return homa_value, EN, GEO


# Calculation of the BLA (Restricted and Unrestricted)


def compute_bla(arr, mol, geom=None):
    """Computes the BLA and BLA_c indices.

    Arguments:
       arr: string
          A string containing the indices of the atoms in ring connectivity.

       mol: an instance of SCF class.
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       geom: list
          The molecular coordinates as given by the mol.atom_coords() function.

    Returns:
       list
          Contains the BLA and the BLA_c indices, respectively.
    """

    distances = find_distances(arr, mol, geom)

    sum1 = sum(distances[i] for i in range(0, len(arr), 2))
    sum2 = sum(distances[i] for i in range(1, len(arr), 2))

    bla = abs(sum1 / (len(arr) // 2) - sum2 / (len(arr) - len(arr) // 2))

    bla_c = 0
    for i in range(len(arr)):
        bla_c += abs(distances[i] - distances[(i + 1) % len(distances)]) / len(distances)

    return bla, bla_c


###############################################
########### ATOMIC OVERLAP MATRICES ###########
###############################################

# Generating the Atomic Overlap Matrices


def make_aoms(mol, mf, partition=None, save=None):
    """Generates the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals (MO) basis.

    Arguments:
       mol: an instance of SCF class
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mf: an instance of SCF class
          mf object holds all parameters to control SCF.

       partition: string
          Type of atomic partition. The available tools are 'mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao'.

       save: string
          Sets the name of the file if they want to be stored in disk. Reccomended '.aoms' extension.

    Returns:
       Smo: list
          Contains the atomic overlap matrices. For restricted calculations, it returns a list of matrices with the AOMS.
          For unrestricted calculations, it returns a list containing both alpha and beta lists of matrices as [Smo_alpha, Smo_beta].

    """

    from pyscf import lo
    import os

    # UNRESTRICTED
    if (mf.__class__.__name__ == "UHF" or mf.__class__.__name__ == "UKS" or mf.__class__.__name__ == "SymAdaptedUHF" or mf.__class__.__name__ == "SymAdaptedUKS"):
        # Getting specific information
        S = mf.get_ovlp()
        nocc_alpha = mf.mo_occ[0].astype(int)
        nocc_beta = mf.mo_occ[1].astype(int)
        occ_coeff_alpha = mf.mo_coeff[0][:, : nocc_alpha.sum()]
        occ_coeff_beta = mf.mo_coeff[0][:, : nocc_beta.sum()]

        # Building the Atomic Overlap Matrices

        Smo_alpha = []
        Smo_beta = []

        if partition == "lowdin" or partition == "meta_lowdin" or partition == "nao":
            if partition == "lowdin":
                U_inv = lo.orth_ao(mf, partition, pre_orth_ao=None)
            elif partition == "meta_lowdin" or partition == "nao":
                U_inv = lo.orth_ao(mf, partition, pre_orth_ao="ANO")
            U = np.linalg.inv(U_inv)

            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, U.T, eta[i]))
                SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, U.T, eta[i]))
                Smo_alpha.append(np.dot(SCR_alpha, SCR_alpha.T))
                Smo_beta.append(np.dot(SCR_beta, SCR_beta.T))

        # Special case IAO
        elif partition == "iao":
            U_alpha_iao_nonortho = lo.iao.iao(mol, occ_coeff_alpha)
            U_beta_iao_nonortho = lo.iao.iao(mol, occ_coeff_beta)
            U_alpha_inv = np.dot(U_alpha_iao_nonortho, lo.orth.lowdin(
                    np.linalg.multi_dot((U_alpha_iao_nonortho.T, S, U_alpha_iao_nonortho))))
            U_beta_inv = np.dot(U_beta_iao_nonortho, lo.orth.lowdin(
                    np.linalg.multi_dot((U_beta_iao_nonortho.T, S, U_beta_iao_nonortho))))
            U_alpha = np.dot(S, U_alpha_inv)
            U_beta = np.dot(S, U_beta_inv)
            pmol = lo.iao.reference_mol(mol)
            nbas_iao = pmol.nao

            eta = [np.zeros((pmol.nao, pmol.nao)) for i in range(pmol.natm)]
            for i in range(pmol.natm):
                start = pmol.aoslice_by_atom()[i, -2]
                end = pmol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(pmol.natm):
                SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, U_alpha, eta[i]))
                SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, U_beta, eta[i]))
                Smo_alpha.append(np.dot(SCR_alpha, SCR_alpha.T))
                Smo_beta.append(np.dot(SCR_beta, SCR_beta.T))

        # Special case plain Mulliken
        elif partition == "mulliken":
            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, S, eta[i], occ_coeff_alpha))
                SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, S, eta[i], occ_coeff_beta))
                Smo_alpha.append(SCR_alpha)
                Smo_beta.append(SCR_beta)

        else:
            raise NameError("Hilbert-space scheme not available")

        Smo = [Smo_alpha, Smo_beta]
        if save is not None:
            from pickle import dump

            with open(save, "wb") as f:
                dump(Smo, f)

        return Smo

    # RESTRICTED

    elif (mf.__class__.__name__ == "RHF" or mf.__class__.__name__ == "RKS" or mf.__class__.__name__ == "SymAdaptedRHF" or mf.__class__.__name__ == "SymAdaptedRKS"):
        # Getting specific information
        S = mf.get_ovlp()
        occ_coeff = mf.mo_coeff[:, mf.mo_occ > 0]

        # Building the Atomic Overlap Matrices

        Smo = []

        if partition == "lowdin" or partition == "meta_lowdin" or partition == "nao":
            if partition == "lowdin":
                U_inv = lo.orth_ao(mf, partition, pre_orth_ao=None)
            elif partition == "meta_lowdin" or partition == "nao":
                U_inv = lo.orth_ao(mf, partition, pre_orth_ao="ANO")
            U = np.linalg.inv(U_inv)

            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((occ_coeff.T, U.T, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

        # Special case IAO
        elif partition == "iao":
            U_iao_nonortho = lo.iao.iao(mol, occ_coeff)
            U_inv = np.dot(U_iao_nonortho, lo.orth.lowdin(
                    np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))))
            U = np.dot(S, U_inv)
            pmol = lo.iao.reference_mol(mol)
            nbas_iao = pmol.nao

            eta = [np.zeros((pmol.nao, pmol.nao)) for i in range(pmol.natm)]
            for i in range(pmol.natm):
                start = pmol.aoslice_by_atom()[i, -2]
                end = pmol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(pmol.natm):
                SCR = np.linalg.multi_dot((occ_coeff.T, U, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

        # Special case plain Mulliken
        elif partition == "mulliken":
            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((occ_coeff.T, S, eta[i], occ_coeff))
                Smo.append(SCR)

        else:
            raise NameError("Hilbert-space scheme not available")

        if save is not None:
            with open(save, "wb") as f:
                np.save(f, Smo)

        return Smo

    else: 
        print(" Only restricted and unrestricted HF and KS-DFT available with this version of the program")
        return

def mol_info(mol, mf, save=None, partition=None):
    """Obtains a series of information about the molecule and the calculation
    to complement the main aromaticity() function without requiring the 'mol'
    and 'mf' objects.

    Arguments:
        mol: an instance of SCF class
           Mole class and helper functions to handle parameters and attributes for GTO integrals.

        mf: an instance of SCF class
           mf object holds all parameters to control SCF.

        save: string
           Sets the name of the file if they want to be stored in disk. Recomended '.molinfo' extension.

        partition: string
           Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.
    Returns:
       List containing:
          Atomic symbols, atomic charges, basis set, molecular coordinates, type of calculation
    """

    info = []
    info.append([mol.atom_symbol(i) for i in range(mol.natm)])  # Atomic Symbols of the molecule
    info.append([i + 1 for i in range(mol.natm)])  # Indices of the atoms
    info.append(mol.basis.upper())
    info.append(mol.atom_coords())
    info.append(mf.__class__.__name__)
    info.append(mf.e_tot)
    info.append(mf.__module__)
    if "dft" in mf.__module__ and mf.xc is not None:
        info.append(mf.xc)

    if save is not None:
        from pickle import dump

        with open(save, "wb") as f:
            dump(info, f)
    return info

