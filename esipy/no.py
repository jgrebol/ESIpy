import numpy as np

from esipy.tools import format_partition


def info_no(aom, molinfo):
    """Prints the initial information for Natural Orbitals calculations.
    Args:
        aom: list of matrices or string
            Atomic Overlap Matrices (AOMs) in the MO basis.
        molinfo: dictionary
            Contains the information about the molecule and the calculation.
    """

    aom, occ = aom
    partition = format_partition(molinfo["partition"])
    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    print(" | Number of Atoms:          {}".format(len(aom)))
    print(" | Occ. Mol. Orbitals:       {}".format(np.shape(aom[0])[0]))
    print(" | Wavefunction type:        Natural Orbitals")
    print(" | Atomic partition:         {}".format(partition.upper() if partition else "Not specified"))
    print(" ------------------------------------------- ")
    print(" ------------------------------------------- ")
    print(" | Method:                  ", molinfo["calctype"])

    if "dft" in molinfo["method"] and molinfo["xc"] is not None:
        print(" | Functional:              ", molinfo["xc"])

    print(" | Basis set:               ", molinfo["basisset"].upper())
    if isinstance(molinfo["energy"], str):
        print(" | Total energy:          {}".format(molinfo["energy"]))
    else:
        print(" | Total energy:             {:<13f}".format(molinfo["energy"]))
    print(" ------------------------------------------- ")
    trace = np.sum([np.trace(matrix) for matrix in aom])
    print(" | Tr(Enter):    {:.13f}".format(trace))
    print(" ------------------------------------------- ")


def deloc_no(aom, molinfo):
    """Population analysis, localization and delocalization indices for Natural Orbitals calculations.
    Args:
        aom: list of matrices
            Atomic Overlap Matrices (AOMs) in the MO basis.
        molinfo: dictionary
            Contains the information about the molecule and the calculation.
    """

    aom, occ = aom
    symbols = molinfo["symbols"]

    # Getting the LIs and DIs
    difs, dixs, lifs, lixs, N = [], [], [], [], []

    print(" ---------------------------------------------------------- ")
    print(" | Atom     N(Sij)    dlocF     dlocX      locF      locX ")
    print(" ---------------------------------------------------------- ")

    for i in range(len(aom)):
        lif = np.trace(np.linalg.multi_dot((occ ** (1 / 2), aom[i], occ ** (1 / 2), aom[i])))
        lix = 0.5 * np.trace(np.linalg.multi_dot((occ, aom[i], occ, aom[i])))
        lifs.append(lif)
        lixs.append(lix)
        N.append(np.trace(np.dot(occ, aom[i])))

        dlocF = 0
        dlocX = 0
        for j in range(len(aom)):
            if i != j:
                dif = np.trace(np.linalg.multi_dot((occ ** (1 / 2), aom[i], occ ** (1 / 2), aom[j])))
                dix = 0.5 * np.trace(np.linalg.multi_dot((occ, aom[i], occ, aom[j])))
                dlocF += dif
                dlocX += dix
                difs.append(dif)
                dixs.append(dix)

        print(" | {:>2} {:>2d}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}".format(
            symbols[i], i + 1, N[i], N[i] - lif, dlocX, lif, lix))
    print(" ---------------------------------------------------------- ")
    print(" | TOT:   {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}".format(
        sum(N), sum(N) - sum(lifs), sum(dixs), sum(lifs), sum(lixs)))
    print(" ---------------------------------------------------------- ")

    print(" ---------------------------------- ")
    print(" |    Pair       DI(F)     DI(X) ")
    print(" ---------------------------------- ")
    for i in range(len(aom)):
        for j in range(i, len(aom)):
            if i == j:
                print(" | {:>2}{:>2}-{:>2}{:>2}  {:>8.4f}  {:>8.4f}".format(
                    symbols[i], i + 1, symbols[j], j + 1, lifs[i], lixs[i]))
            else:
                print(" | {:>2}{:>2}-{:>2}{:>2}  {:>8.4f}  {:>8.4f}".format(
                    symbols[i], i + 1, symbols[j], j + 1, 2 * difs[i * len(aom) + j - (i + 1)],
                                2 * dixs[i * len(aom) + j - (i + 1)]))
    print(" ---------------------------------- ")
    print(" |   TOT:     {:>8.4f}  {:>8.4f}  ".format(np.sum(difs) + np.sum(lifs), np.sum(dixs) + np.sum(lixs)))
    print(" |   LOC:     {:>8.4f}  {:>8.4f} ".format(np.sum(lifs), np.sum(lixs)))
    print(" | DELOC:     {:>8.4f}  {:>8.4f} ".format(np.sum(difs), np.sum(dixs)))
    print(" ---------------------------------- ")


def arom_no(rings, molinfo, indicators, mci=False, av1245=False, partition=None, flurefs=None, homarefs=None,
            homerrefs=None,
            ncores=1):
    """
    Output for the aromaticity indices for Natural Orbitals calculations. Will use Fulton's approximation.
    Args:
        rings: list of lists
            Contains the indices of the atoms in the rings.
        molinfo: dictionary
            Contains the information about the molecule and the calculation.
        indicators: class
            Contains the aromaticity indicators.
        mci: boolean
            If True, the MCI is computed.
        av1245: boolean
            If True, the AV1245 is computed.
        partition: string
            Contains the name of the partition.
        flurefs: dictionary
            Contains the custom references for the FLU aromaticity index.
        homarefs: dictionary
            Contains the custom references for the HOMA aromaticity index.
        homerrefs: dictionary
            Contains the custom references for the HOMER aromaticity index.
        ncores: integer
            Number of cores to use in the MCI calculation. By default, 1.
    """

    print(" | Fulton index used for the calculation of aromaticity indicators     ")
    if partition == "iao":
        print(" | WARNING: IAOs transformation matrix is built upon the HF instance")

    print(" ----------------------------------------------------------------------")
    print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]           ")
    print(" |                      HOMA [Tetrahedron 52, 10255 (1996)]")
    print(" |                       FLU [JCP 122, 014109 (2005)]      ")
    print(" |                     Iring [PCCP 2, 3381 (2000)]         ")
    if mci is True:
        print(" |                       MCI [JPOC 18, 706 (2005)]         ")
    if av1245 is True:
        print(" |                    AV1245 [PCCP 18, 11839 (2016)]       ")
        print(" |                    AVmin  [JPCC 121, 27118 (2017)]      ")
        print(" |                           [PCCP 20, 2787 (2018)]        ")
    print(" |  For a recent review see: [CSR 44, 6434 (2015)]         ")
    print(" ----------------------------------------------------------------------")

    # Checking where to read the atomic symbols from
    if molinfo:
        symbols = molinfo["symbols"]
        partition = molinfo["partition"]
    else:
        raise NameError(" 'molinfo' not found. Check input")

    if not isinstance(rings[0], list):
        rings = [rings]

    # Looping through each of the rings

    for ring_index, ring in enumerate(rings):
        print(" ----------------------------------------------------------------------")

        print(" |")
        print(" | Ring  {} ({}):   {}".format(ring_index + 1, len(ring), "  ".join(str(num) for num in ring)))
        print(" |")
        print(" ----------------------------------------------------------------------")
        if homarefs is not None:
            print(" | Using HOMA references provided by the user")
        else:
            print(" | Using default HOMA references")
        homa = indicators[ring_index].homa
        if homa is None:
            print(" | Connectivity could not match parameters")
        else:
            print(" | EN           {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].en))
            print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].geo))
            print(" | HOMA         {} =  {:>.6f}".format(ring_index + 1, homa))
            if homerrefs:
                print(" | ")
                print(" | Found custom HOMER references 'alpha' and 'r_opt'. Computing")
                print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].homer))
            print(" ----------------------------------------------------------------------")
            print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].bla))
            print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].bla_c))
            print(" ----------------------------------------------------------------------")

        print(" ----------------------------------------------------------------------")
        print(" | Current version does not allow FLU for correlated wavefunctions")
        print(" ----------------------------------------------------------------------")

        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa))
        print(" | BOAc         {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa_c))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdi_list = indicators[ring_index].pdi_list
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[0], ring[3], pdi_list[0]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[1], ring[4], pdi_list[1]))
            print(" | DI ({:>2} -{:>2} )   =  {:.4f}".format(ring[2], ring[5], pdi_list[2]))
            print(" | PDI          {} =  {:.4f} ".format(ring_index + 1, indicators[ring_index].pdi))
        print(" ----------------------------------------------------------------------")

        if av1245 == True:
            if len(ring) < 6:
                print(" | AV1245 could not be calculated as the number of centers is smaller than 6 ")

            else:
                av1245_list = indicators[ring_index].av1245_list
                av1245_pairs = [(ring[i % len(ring)], ring[(i + 1) % len(ring)], ring[(i + 3) % len(ring)],
                                 ring[(i + 4) % len(ring)])
                                for i in range(len(ring))]

                for j in range(len(ring)):
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}".format(
                        str(ring[j]).rjust(2), symbols[av1245_pairs[j][0] - 1].ljust(2),
                        str(ring[(j + 1) % len(ring)]).rjust(2), symbols[av1245_pairs[j][1] - 1].ljust(2),
                        str(ring[(j + 3) % len(ring)]).rjust(2), symbols[av1245_pairs[j][2] - 1].ljust(2),
                        str(ring[(j + 4) % len(ring)]).rjust(2), symbols[av1245_pairs[j][3] - 1].ljust(2),
                        av1245_list[(ring[j] - 1) % len(ring)]))
                print(" | AV1245 {} =             {:.4f}".format(ring_index + 1, indicators[ring_index].av1245))
                print(" |  AVmin {} =             {:.4f}".format(ring_index + 1, indicators[ring_index].avmin))
                print(" ---------------------------------------------------------------------- ")

        iring_total = indicators[ring_index].iring
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
                mci_total = indicators[ring_index].mci
                end_mci = time.time()
                time_mci = end_mci - start_mci
                print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

                # MULTI-CORE
            else:
                if partition is None:
                    print(" | Partition not specified. Will assume symmetric AOMs")
                start_mci = time.time()
                mci_total = indicators[ring_index].mci
                end_mci = time.time()
                time_mci = end_mci - start_mci

                print(" | The MCI calculation using {} cores took {:.4f} seconds".format(ncores, time_mci))
                print(" | MCI          {} =  {:.6f}".format(ring_index + 1, mci_total))

            if mci_total < 0:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, -((np.abs(mci_total)) ** (1 / len(ring)))))

            else:
                print(" | MCI**(1/n)   {} =  {:>6f}".format(ring_index + 1, mci_total ** (1 / len(ring))))
            print(" ---------------------------------------------------------------------- ")
