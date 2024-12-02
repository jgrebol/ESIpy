import numpy as np
from esipy.tools import mol_info, find_multiplicity
from esipy.indicators import *


def deloc_unrest(Smo, mol=None, molinfo=None):
    """Population analysis, localization and delocalization indices for unrestriced AOMs.

    Arguments:
        Smo (list of lists of matrices):
            Specifies the Atomic Overlap Matrices (AOMs) in the MO basis as [Smo_alpha, Smo_beta].

        mol (SCF instance, optional, default: None):
            PySCF's Mole class and helper functions to handle parameters and attributes for GTO integrals.

        molinfo (dict, optional, default: None):
            Contains molecule and calculation details from the 'molinfo()' method inside ESI.
    """
    if molinfo:
        if isinstance(molinfo, str):
            with open(molinfo, "rb") as f:
                molinfo = np.load(f, allow_pickle=True)
        symbols = molinfo["symbols"]
    elif mol:
        symbols = mol_info(mol=mol)["symbols"]
    else:
        raise NameError("Could not find mol nor molinfo")

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
            diaa = 2 * np.trace(np.dot(Smo[0][i], Smo[0][j]))
            dibb = 2 * np.trace(np.dot(Smo[1][i], Smo[1][j]))
            dis_alpha.append(diaa)
            dis_beta.append(dibb)

    print(" ------------------------------------------------------------------- ")
    print(" |  Atom     N(Sij)      Na(Sij)     Nb(Sij)     dloc_a     dloc_b  ")
    print(" ------------------------------------------------------------------- ")

    for i in range(len(Smo[0])):
        print(" | {} {:>2d}   {:10.6f}  {:10.6f}  {:10.6f}   {:8.4f}   {:8.4f} ".format(
            symbols[i], i + 1, Nij_alpha[i] + Nij_beta[i], Nij_alpha[i], Nij_beta[i], Nij_alpha[i] - lis_alpha[i],
                        Nij_beta[i] - lis_beta[i]))
    print(" ------------------------------------------------------------------- ")
    print(" | TOT:   {:10.6f}  {:10.6f}  {:10.6f}   {:8.4f}   {:8.4f}".format(
        sum(Nij_alpha) + sum(Nij_beta), sum(Nij_alpha), sum(Nij_beta), sum(Nij_alpha) - sum(lis_alpha),
        sum(Nij_beta) - sum(lis_beta)))
    print(" ------------------------------------------------------------------- ")
    print(" ------------------------------------------- ")
    print(" |    Pair         DI       DIaa      DIbb ")
    print(" ------------------------------------------- ")

    for i in range(len(symbols)):
        for j in range(i, len(symbols)):
            if i == j:
                print(" | {} {:>2}-{} {:>2}   {:8.4f}  {:8.4f}  {:8.4f}".format(
            symbols[i], str(i + 1).center(2), symbols[j],
                    str(j + 1).center(2), lis_alpha[i] + lis_beta[i], lis_alpha[i], lis_beta[i]))
            else:
                dia = 2 * np.trace(np.dot(Smo[0][i], Smo[0][j]))
                dib = 2 * np.trace(np.dot(Smo[1][i], Smo[1][j]))
                ditot = dia + dib
                print(" | {} {:>2}-{} {:>2}   {:8.4f}  {:8.4f}  {:8.4f}".format(
            symbols[i], str(i + 1).center(2), symbols[j],
                    str(j + 1).center(2), ditot, dia, dib))
    print(" ------------------------------------------- ")
    print(" |    TOT:    {:>9.4f} {:>9.4f} {:>9.4f} ".format(
        sum(dis_alpha) + sum(dis_beta) + sum(lis_alpha) + sum(lis_beta), sum(dis_alpha) + sum(lis_alpha),
        sum(dis_beta) + sum(lis_beta)))
    print(" |    LOC:    {:>9.4f} {:>9.4f} {:>9.4f} ".format(
        sum(lis_alpha) + sum(lis_beta), sum(lis_alpha), sum(lis_beta)))
    print(" |  DELOC:    {:>9.4f} {:>9.4f} {:>9.4f} ".format(
        sum(dis_alpha) + sum(dis_beta), sum(dis_alpha), sum(dis_beta)))
    print(" ------------------------------------------- ")

def arom_unrest(Smo, rings, partition, mol, mci=False, av1245=False, flurefs=None, homarefs=None, homerrefs=None,
                connectivity=None, geom=None, ncores=1, molinfo=None):
    """Population analysis, localization and delocalization indices and aromaticity indicators
    for unrestricted AOMs.

    Arguments:
        Smo (list of matrices or str):
            Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.

        rings (list or list of lists of int):
            Contains the indices defining the ring connectivity of a system. Can contain several rings as a list of lists.

        partition (str):
            Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.

        mol (SCF instance):
            PySCF's Mole class and helper functions to handle parameters and attributes for GTO integrals.

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
            Contains molecule and calculation details from the 'molinfo()' method inside ESI.

        ncores (int, optional, default: 1):
            Specifies the number of cores for multi-processing MCI calculation.
    """

    print(" ----------------------------------------------------------------------")
    print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
    print(" |                     Iring [PCCP 2, 3381 (2000)]")
    print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
    print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
    print(" |                           [PCCP 20, 2787 (2018)]")
    print(" |  For a recent review see: [CSR 44, 6434 (2015)]")
    print(" ----------------------------------------------------------------------")

    if molinfo:
        if isinstance(molinfo, str):
            with open(molinfo, "rb") as f:
                molinfo = np.load(f, allow_pickle=True)
        symbols = molinfo["symbols"]
    elif mol:
        symbols = mol_info(mol=mol)["symbols"]
    else:
        raise NameError("Could not find mol nor molinfo")

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

            if find_multiplicity(Smo) == "triplet":
                print(" | Triplet AOMs. Computing HOMER")
                if homerrefs is not None:
                    print(" | Using HOMER references provided by the user")
                else:
                    print(" | Using the default HOMER references")

                homer = compute_homer(ring, mol, geom=geom, homerrefs=homerrefs, connectivity=connectivity)
                print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, homer))
                print(" ----------------------------------------------------------------------")

        blas = compute_bla(ring, mol, geom=geom)

        print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
        print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
        print(" ----------------------------------------------------------------------")
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
                avs_alpha = np.array(compute_av1245(ring, Smo[0], partition), dtype=object)
                avs_beta = np.array(compute_av1245(ring, Smo[1], partition), dtype=object)

                print(" |")
                print(" | *** AV1245_ALPHA ***")
                for j in range(len(ring)):
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}".format(
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
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}".format(
                        str(ring[j]).rjust(2), symbols[(ring[j % len(ring)] - 1)],
                        str(ring[(j + 1) % len(ring)]).rjust(2), symbols[(ring[(j + 1) % len(ring)] - 1)],
                        str(ring[(j + 3) % len(ring)]).rjust(2), symbols[(ring[(j + 3) % len(ring)] - 1)],
                        str(ring[(j + 4) % len(ring)]).rjust(2), symbols[(ring[(j + 4) % len(ring)] - 1)],
                        np.array(avs_beta[2][(ring[j] - 1) % len(ring)])))
                print(" |   AV1245_beta  {} =             {:>9.4f}".format(ring_index + 1, avs_beta[0]))
                print(" |    AVmin_beta  {} =             {:>9.4f}".format(ring_index + 1, avs_beta[1]))
                print(" | ")
                print(" | *** AV1245_TOTAL ***")
                for j in range(len(ring)):
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}".format(
                        str(ring[j]).rjust(2), symbols[(ring[j % len(ring)] - 1)],
                        str(ring[(j + 1) % len(ring)]).rjust(2), symbols[(ring[(j + 1) % len(ring)] - 1)],
                        str(ring[(j + 3) % len(ring)]).rjust(2), symbols[(ring[(j + 3) % len(ring)] - 1)],
                        str(ring[(j + 4) % len(ring)]).rjust(2), symbols[(ring[(j + 4) % len(ring)] - 1)],
                        np.array(avs_alpha[2][(ring[j] - 1) % len(ring)]) + np.array(
                            avs_beta[2][(ring[j] - 1) % len(ring)])))
                print(" |   AV1245       {} =             {:>9.4f}".format(ring_index + 1, avs_alpha[0] + avs_beta[0]))
                all_avs = [avs_alpha[2][i] + avs_beta[2][i] for i in range(len(avs_alpha[2]))]
                print(" |    AVmin       {} =             {:>9.4f}".format(ring_index + 1, min(all_avs, key=abs)))
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

            # SINGLE-CORE
            if ncores == 1:
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                start_mci = time.time()
                mci_alpha = sequential_mci(ring, Smo[0], partition)
                mci_beta = sequential_mci(ring, Smo[1], partition)
                mci_total = mci_alpha + mci_beta
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            # MULTI-CORE
            else:
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                start_mci = time.time()
                mci_alpha = multiprocessing_mci(ring, Smo[0], ncores, partition)
                mci_beta = multiprocessing_mci(ring, Smo[1], ncores, partition)
                mci_total = mci_alpha + mci_beta
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(ncores, time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))

            print(" ---------------------------------------------------------------------- ")


def arom_unrest_from_aoms(Smo, rings, partition, mol, mci=False, av1245=False, flurefs=None, homarefs=None,
                          homerrefs=None, connectivity=None, geom=None, ncores=1):
    """Population analysis, localization and delocalization indices and aromaticity indicators
    for previously saved restricted AOMs.

    Arguments:
        Smo (list of matrices or str):
            Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.

        rings (list or list of lists of int):
            Contains the indices defining the ring connectivity of a system. Can contain several rings as a list of lists.

        partition (str):
            Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.

        mol (SCF instance):
            PySCF's Mole class and helper functions to handle parameters and attributes for GTO integrals.

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
            Contains molecule and calculation details from the 'molinfo()' method inside ESI.

        ncores (int, optional, default: 1):
            Specifies the number of cores for multi-processing MCI calculation.
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
            try:
                symbols
            except NameError:
                symbols = None
            if symbols is not None:
                connectivity = [symbols[int(i) - 1] for i in ring]
            else:
                print(" | Connectivity not found. Could not compute geometric indices")
        else:
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

                if find_multiplicity(Smo) == "triplet":
                    print(" | Different number of alpha and beta electrons. Computing HOMER")
                    if homerrefs is not None:
                        print(" | Using HOMER references provided by the user")
                    else:
                        print(" | Using default HOMER references")

                    homer = compute_homer(ring, mol, geom=geom, homerrefs=homerrefs, connectivity=connectivity)
                    if homer is None:
                        print(" | Connectivity could not match parameters")
                    else:
                        print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, homer))
                        print(" ----------------------------------------------------------------------")

                blas = compute_bla(ring, mol, geom=geom)

                print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
                print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
                print(" ----------------------------------------------------------------------")
        print(" ----------------------------------------------------------------------")

        if connectivity is not None:
            if isinstance(connectivity[0], str) and mol is None and mol_info is None:
                if connectivity[ring_index - 1] is None:
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
                avs_alpha = np.array(compute_av1245(ring, Smo[0], partition), dtype=object)
                avs_beta = np.array(compute_av1245(ring, Smo[1], partition), dtype=object)

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
                    print(" |   A {} -  A {} -  A {} -  A {}  |  {:>9.4f}".format(
                        str(ring[j]).rjust(2), str(ring[(j + 1) % len(ring)]).rjust(2),
                        str(ring[(j + 3) % len(ring)]).rjust(2), str(ring[(j + 4) % len(ring)]).rjust(2),
                        np.array(avs_beta[2][(ring[j] - 1) % len(ring)])))
                print(" |   AV1245_beta  {} =             {:>9.4f}".format(ring_index + 1, avs_beta[0]))

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

            # SINGLE-CORE
            if ncores == 1:
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                start_mci = time.time()
                mci_alpha = sequential_mci(ring, Smo[0], partition)
                mci_beta = sequential_mci(ring, Smo[1], partition)
                end_mci = time.time()
                mci_total = mci_alpha + mci_beta
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            # MULTI-CORE
            else:
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                start_mci = time.time()
                mci_alpha = multiprocessing_mci(ring, Smo[0], ncores, partition)
                mci_beta = multiprocessing_mci(ring, Smo[1], ncores, partition)
                mci_total = mci_alpha + mci_beta
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(ncores, time_mci))
                print(" | MCI_alpha    {} =  {:>6f}".format(ring_index + 1, mci_alpha))
                print(" | MCI_beta     {} =  {:>6f}".format(ring_index + 1, mci_beta))
                print(" | MCI          {} =  {:>6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))
            print(" ---------------------------------------------------------------------- ")
