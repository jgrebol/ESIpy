import numpy as np

from esipy.tools import format_partition


def info_rest(aom, molinfo):
    """
    Print the information of the molecule and the calculation.
    Args:
        aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
        molinfo: Information about the molecule and the calculation.
    """
    partition = format_partition(molinfo["partition"])

    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    print(" | Number of Atoms:          {}".format(len(aom)))
    print(" | Occ. Mol. Orbitals:       {}".format(np.shape(aom[0])[0]))
    print(" | Wavefunction type:        Restricted")
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


def deloc_rest(aom, molinfo):
    """Population analysis, localization and delocalization indices for restricted, single-determinant calculations.
    Args:
        aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
        molinfo: Information about the molecule and the calculation.
    """

    # Checking where to read the atomic symbols from
    symbols = molinfo["symbols"]

    print(" ------------------------------------- ")
    print(" | Atom    N(Sij)     loc.      dloc. ")
    print(" ------------------------------------- ")

    # Getting the LIs and DIs
    dis, lis, N = [], [], []
    for i in range(len(aom)):
        li = 2 * np.trace(np.dot(aom[i], aom[i]))
        lis.append(li)
        N.append(2 * np.trace(aom[i]))

        print(" | {:>2}{:>2d}  {:>8.4f}  {:>8.4f}  {:>8.4f} ".format(
            symbols[i], i + 1, N[i], lis[i], N[i] - lis[i]))

        for j in range(i + 1, len(aom)):
            di = 4 * np.trace(np.dot(aom[i], aom[j]))
            dis.append(di)
    print(" ------------------------------------- ")
    print(" | TOT:  {:>8.4f}  {:>8.4f}  {:>8.4f}".format(
        sum(N), sum(N) - sum(dis), sum(dis)))
    print(" ------------------------------------- ")

    print(" ------------------------ ")
    print(" |    Pair         DI ")
    print(" ------------------------ ")
    for i in range(len(aom)):
        for j in range(i, len(aom)):
            if i == j:
                print(" | {:>2}{:>2}-{:>2}{:>2}   {:>8.4f}".format(
                    symbols[i], str(i + 1).center(2), symbols[j],
                    str(j + 1).center(2), lis[i]))
            else:
                print(" | {:>2}{:>2}-{:>2}{:>2}   {:>8.4f}".format(
                    symbols[i], str(i + 1).center(2), symbols[j],
                    str(j + 1).center(2), 4 * np.trace(np.dot(aom[i], aom[j]))))
    print(" ------------------------ ")
    print(" |   TOT:      {:>8.4f} ".format(np.sum(dis) + np.sum(lis)))
    print(" |   LOC:      {:>8.4f} ".format(np.sum(lis)))
    print(" | DELOC:      {:>8.4f} ".format(np.sum(dis)))
    print(" ------------------------ ")


def arom_rest(rings, molinfo, indicators, mci=False, av1245=False, flurefs=None, homarefs=None, homerrefs=None,
              ncores=1):
    """
    Output for the aromaticity indices for restricted, single-determinant calculations.
    Args:
        rings: List of the atoms in the rings.
        molinfo: Information about the molecule and the calculation.
        indicators: Class containing the indicators for each ring.
        mci: Boolean to compute the MCI.
        av1245: Boolean to compute the AV1245.
        flurefs: Dictionary with custom references for the FLU.
        homarefs: Dictionary with custom references for the HOMA.
        homerrefs: Dictionary with custom references for the HOMER.
        ncores: Number of cores to use for the MCI calculation. By default, 1.
    """

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
        connectivity = [symbols[int(i) - 1] for i in ring]
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
            print(" | HOMA         {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].homa))

        if homerrefs:
            print(" | ")
            print(" | Found custom HOMER references 'alpha' and 'r_opt'. Computing")
            print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].homer))
        print(" ----------------------------------------------------------------------")

        print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].bla))
        print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].bla_c))

        flu = indicators[ring_index].flu
        if flu is None:
            print(" | Could not compute FLU")
        else:
            if flurefs is not None:
                print(" | Using FLU references provided by the user")
            else:
                print(" | Using the default FLU references")
            print(" ----------------------------------------------------------------------")
            print(" | Atoms  :   {}".format("  ".join(str(atom) for atom in connectivity)))
            print(" |")
            print(" | FLU          {} =  {:>.6f}".format(ring_index + 1, flu))
        print(" ----------------------------------------------------------------------")

        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa))
        print(" | BOA_cc       {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa_c))
        print(" ----------------------------------------------------------------------")

        # Checking the length of the ring. PDI only computed for len(ring)=6.
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
