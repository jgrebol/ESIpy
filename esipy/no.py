import numpy as np

from esipy.tools import load_file, find_di_no
from esipy.indicators import *

def deloc_no(Smo, mol, molinfo=None):
    """Population analysis, localization and delocalization indices for restricted AOMs.

    Arguments:

       mol: an instance of SCF class
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

    """

    occ = Smo[1]
    Smo = Smo[0]

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
    difs, dixs, lifs, lixs, N = [], [], [], [], []

    print(" ---------------------------------------------------------- ")
    print(" | Atom     N(Sij)    dlocF     dlocX      locF      locX ")
    print(" ---------------------------------------------------------- ")

    for i in range(len(Smo)):
        lif = np.trace(np.linalg.multi_dot((occ**(1/2), Smo[i], occ**(1/2), Smo[i])))
        lix = 0.5 * np.trace(np.linalg.multi_dot((occ, Smo[i], occ, Smo[i])))
        lifs.append(lif)
        lixs.append(lix)
        N.append(np.trace(occ * Smo[i]))

        dlocF = 0
        dlocX = 0
        for j in range(len(Smo)):
            if i != j:
                dif = np.trace(np.linalg.multi_dot((occ**(1/2), Smo[i], occ**(1/2), Smo[j])))
                dix = 0.5 * np.trace(np.linalg.multi_dot((occ, Smo[i], occ, Smo[j])))
                dlocF += dif
                dlocX += dix
                difs.append(dif)
                dixs.append(dix)

        print(" | {} {:>2d}   {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}".format(
            symbols[i], i + 1, N[i], dlocF, dlocX, lif, lix))
    print(" ---------------------------------------------------------- ")
    print(" | TOT:   {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}".format(
        sum(N), sum(difs), sum(dixs), sum(lifs), sum(lixs)))
    print(" ---------------------------------------------------------- ")

    print(" ---------------------------------- ")
    print(" |    Pair       DI(F)     DI(X) ")
    print(" ---------------------------------- ")
    for i in range(len(Smo)):
        for j in range(i, len(Smo)):
            if i == j:
                print(" | {} {:>2}-{} {:>2}  {:>8.4f}  {:>8.4f}".format(
                        symbols[i], i + 1, symbols[j], j + 1, lifs[i], lixs[i]))
            else:
                print(" | {} {:>2}-{} {:>2}  {:>8.4f}  {:>8.4f}".format(
            symbols[i], i + 1, symbols[j], j + 1, 2 * difs[i * len(Smo) + j - (i + 1)], 2 * dixs[i * len(Smo) + j - (i + 1)]))
    print(" ---------------------------------- ")
    print(" |   TOT:      {:>8.4f}  {:>8.4f}  ".format(np.sum(difs) + np.sum(lifs), np.sum(dixs) + np.sum(lixs)))
    print(" |   LOC:      {:>8.4f}  {:>8.4f} ".format(np.sum(lifs), np.sum(lixs)))
    print(" | DELOC:      {:>8.4f}  {:>8.4f} ".format(np.sum(difs), np.sum(dixs)))
    print(" ---------------------------------- ")

def arom_no(Smo, rings, partition, mol, mci=False, av1245=False, flurefs=None,
                  homarefs=None, homerrefs=None, connectivity=None, geom=None, ncores=1,
                  molinfo=None):
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

       homerrefs: dictionary. Default: None
          User-provided references for the optimal distance and polarizability for the HOMA or HOMER indices.

       connectivity: list. Default: None
          The atomic symbols of the atoms in the ring in 'mol' order.

       geom: list. Default: None
          The molecular coordinates as given by the mol.atom_coords() function.

       ncores: integer
          Number of threads required for the calculation.
    """

    print(" | Fulton index used for the calculation of aromaticity indicators     ")
    print(" ----------------------------------------------------------------------")
    print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
    print(" |                     Iring [PCCP 2, 3381 (2000)]")
    print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
    print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
    print(" |                           [PCCP 20, 2787 (2018)]")
    print(" |  For a recent review see: [CSR 44, 6434 (2015)]")

    # Checking if the list rings is contains more than one ring to analyze
    if mol:
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    elif molinfo:
        if isinstance(molinfo, str):
            load_file(molinfo)
        symbols = molinfo["symbols"]
        geom = molinfo["geom"]
        partition = molinfo["partition"]
    else:
        raise NameError(" Could not find 'mol' nor 'molinfo' file")

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
            print(" ----------------------------------------------------------------------")

            blas = compute_bla(ring, mol, geom=geom)

            print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, blas[0]))
            print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, blas[1]))
            print(" ----------------------------------------------------------------------")

        print(" ----------------------------------------------------------------------")

        print(" The code's current version does not allow FLU for correlated wavefunctions")
        print(" ----------------------------------------------------------------------")

        boas = compute_boa_no(ring, Smo)

        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, boas[0]))
        print(" | BOA_cc       {} =  {:>.6f}".format(ring_index + 1, boas[1]))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdis = np.array(compute_pdi_no(ring, Smo), dtype=object)
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[0], ring[3], pdis[1][0]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[1], ring[4], pdis[1][1]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[2], ring[5], pdis[1][2]))
            print(" | PDI          {} =  {:.4f} ".format(ring_index + 1, pdis[0]))
        print(" ----------------------------------------------------------------------")
        print(" ----------------------------------------------------------------------")

        if av1245 == True:
            if len(ring) < 6:
                print(" | AV1245 could not be calculated as the number of centers is smaller than 6 ")

            else:
                avs = np.array(compute_av1245_no(ring, Smo, partition), dtype=object)
                av1245_pairs = [(ring[i % len(ring)], ring[(i + 1) % len(ring)], ring[(i + 3) % len(ring)],
                                 ring[(i + 4) % len(ring)])
                                for i in range(len(ring))]

                for j in range(len(ring)):
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}".format(
                        str(ring[j]).rjust(2), symbols[av1245_pairs[j][0] - 1].ljust(2),
                        str(ring[(j + 1) % len(ring)]).rjust(2), symbols[av1245_pairs[j][1] - 1].ljust(2),
                        str(ring[(j + 3) % len(ring)]).rjust(2), symbols[av1245_pairs[j][2] - 1].ljust(2),
                        str(ring[(j + 4) % len(ring)]).rjust(2), symbols[av1245_pairs[j][3] - 1].ljust(2),
                        avs[2][(ring[j] - 1) % len(ring)]))
                print(" | AV1245 {} =             {:.4f}".format(ring_index + 1, avs[0]))
                print(" |  AVmin {} =             {:.4f}".format(ring_index + 1, avs[1]))
                print(" ---------------------------------------------------------------------- ")

        iring_total = compute_iring_no(ring, Smo)
        print(" | Iring        {} =  {:>.6f}".format(ring_index + 1, iring_total))

        if iring_total < 0:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, -(np.abs(iring_total) ** (1 / len(ring)))))

        else:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, iring_total ** (1 / len(ring))))

        print(" ---------------------------------------------------------------------- ")

        if mci == True:
            import time

            if ncores is None:
                ncores = 1

            # SINGLE-CORE
            if ncores == 1:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_total = sequential_mci_no(ring, Smo, partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

                # MULTI-CORE
            else:
                start_mci = time.time()
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                mci_total = multiprocessing_mci_no(ring, Smo, ncores, partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(ncores, time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))
            print(" ---------------------------------------------------------------------- ")



def arom_no_from_aoms(Smo, rings, partition, mol, mci=True, av1245=True, flurefs=None, homarefs=None, homerrefs=None,
                        connectivity=None, geom=None, ncores=1):
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

        # Starting the calculation of the aromaticity indicators
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
                flus = compute_flu_no(ring, mol, Smo, flurefs, connectivity, partition=partition)
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

        boas = compute_boa_no(ring, Smo)

        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, boas[0]))
        print(" | BOA_cc       {} =  {:>.6f}".format(ring_index + 1, boas[1]))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdis = np.array(compute_pdi_no(ring, Smo), dtype=object)
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
                avs = np.array(compute_av1245_no(ring, Smo, partition), dtype=object)

                for j in range(len(ring)):
                    print(" |   A {} -  A {} -  A {} -  A {}  |  {:>6.4f}".format(
                        str(ring[j]).rjust(2), str(ring[(j + 1) % len(ring)]).rjust(2),
                        str(ring[(j + 3) % len(ring)]).rjust(2), str(ring[(j + 4) % len(ring)]).rjust(2),
                        2 * avs[2][(ring[j] - 1) % len(ring)]))
                print(" | AV1245 {} =             {:.4f}".format(ring_index + 1, avs[0]))
                print(" |  AVmin {} =             {:.4f}".format(ring_index + 1, avs[1]))
                print(" ---------------------------------------------------------------------- ")

        iring_total = compute_iring_no(ring, Smo)
        print(" | Iring        {} =  {:>.6f}".format(ring_index + 1, iring_total))

        if iring_total < 0:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, -(np.abs(iring_total) ** (1 / len(ring)))))

        else:
            print(" | Iring**(1/n) {} =  {:>.6f}".format(ring_index + 1, iring_total ** (1 / len(ring))))
        print(" ---------------------------------------------------------------------- ")

        if mci == True:
            import time

            # SINGLE-CORE
            if ncores == 1:
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                start_mci = time.time()
                mci_total = sequential_mci_no(ring, Smo, partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            # MULTI-CORE
            else:
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                start_mci = time.time()
                mci_total = sequential_mci_no(ring, Smo, partition)
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(ncores, time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))

        print(" ---------------------------------------------------------------------- ")
