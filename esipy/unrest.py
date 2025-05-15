import numpy as np

from esipy.tools import format_partition, find_multiplicity


def info_unrest(aom, molinfo, nfrags=0):
    """
    Prints the information of the calculation for unrestricted wavefunctions.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param molinfo: Contains the information of the molecule and the calculation.
    :type molinfo: dict
    """

    partition = format_partition(molinfo["partition"])
    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    print(" | Number of Atoms:          {}".format(len(aom[0])-nfrags))
    print(" | Occ. Mol. Orbitals:       {}({})".format(np.shape(aom[0][0])[0], np.shape(aom[1][0])[0]))
    print(" | Wavefunction type:        Unrestricted")
    print(" | Atomic partition:         {}".format(partition.upper() if partition else "Not specified"))
    print(" ------------------------------------------- ")
    print(" ------------------------------------------- ")
    print(" | Method:                  ", molinfo["calctype"])

    if "dft" in molinfo["method"] and molinfo["xc"] is not None:
        print(" | Functional:              ", molinfo["xc"])

    if isinstance(molinfo["basisset"], dict):
        for key in molinfo["basisset"]:
            print(" | Basis set for {:>2}:         {}".format(key, molinfo["basisset"][key].upper()))
    elif isinstance(molinfo["basisset"], str):
        print(" | Basis set:               ", molinfo["basisset"].upper())
    if isinstance(molinfo["energy"], str):
        print(" | Total energy:          {}".format(molinfo["energy"]))
    else:
        print(" | Total energy:             {:<13f}".format(molinfo["energy"]))
    print(" ------------------------------------------- ")
    trace_a = np.sum([np.trace(matrix) for matrix in aom[0]][:len(molinfo["symbols"])])
    trace_b = np.sum([np.trace(matrix) for matrix in aom[1]][:len(molinfo["symbols"])])
    print(" | Tr(alpha):    {:.13f}".format(trace_a))
    print(" | Tr(beta) :    {:.13f}".format(trace_b))
    print(" | Tr(Enter):    {:.13f}".format(trace_a + trace_b))
    print(" ------------------------------------------- ")


def deloc_unrest(aom, molinfo, fragmap={}):
    """
    Population analysis, localization and delocalization indices for unrestricted, single-determinant calculations.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param molinfo: Contains the information of the molecule and the calculation.
    :type molinfo: dict
    """

    presymbols = molinfo["symbols"]
    symbols = presymbols + ["FF"] * (len(fragmap))

    # Getting the LIs and DIs
    dis_alpha, dis_beta = [], []
    lis_alpha, lis_beta = [], []
    Nij_alpha, Nij_beta = [], []

    for i in range(len(aom[0])):
        li_alpha = np.trace(np.dot(aom[0][i], aom[0][i]))
        li_beta = np.trace(np.dot(aom[1][i], aom[1][i]))
        lis_alpha.append(li_alpha)
        lis_beta.append(li_beta)
        Nij_alpha.append(np.trace(aom[0][i]))
        Nij_beta.append(np.trace(aom[1][i]))

        for j in range(i + 1, len(aom[0])):
            diaa = 2 * np.trace(np.dot(aom[0][i], aom[0][j]))
            dibb = 2 * np.trace(np.dot(aom[1][i], aom[1][j]))
            dis_alpha.append(diaa)
            dis_beta.append(dibb)

    print(" ------------------------------------------------------------------- ")
    print(" |  Atom     N(Sij)    Na(Sij)   Nb(Sij)    dloc_a    dloc_b  ")
    print(" ------------------------------------------------------------------- ")

    for i in range(len(aom[0])):
        print(" | {:>2}{:>2d}    {:8.4f}  {:8.4f}  {:8.4f}   {:8.4f}   {:8.4f} ".format(
            symbols[i], i + 1, Nij_alpha[i] + Nij_beta[i], Nij_alpha[i], Nij_beta[i], Nij_alpha[i] - lis_alpha[i],
                        Nij_beta[i] - lis_beta[i]))
    print(" ------------------------------------------------------------------- ")
    Ntota = np.sum(Nij_alpha[:len(presymbols)])
    Ntotb = np.sum(Nij_beta[:len(presymbols)])
    listota = np.sum(lis_alpha[:len(presymbols)])
    listotb = np.sum(lis_beta[:len(presymbols)])
    distota = Ntota - listota
    distotb = Ntotb - listotb
    print(" | TOT:    {:8.4f}  {:8.4f}  {:8.4f}   {:8.4f}   {:8.4f}".format(Ntota+Ntotb, Ntota, Ntotb, distota, distotb))
    print(" ------------------------------------------------------------------- ")
    print(" ------------------------------------------- ")
    print(" |    Pair        DI       DIaa      DIbb ")
    print(" ------------------------------------------- ")

    for i in range(len(symbols)):
        for j in range(i, len(symbols)):
            if i == j:
                print(" | {:>2}{:>2}-{:>2}{:>2}  {:8.4f}  {:8.4f}  {:8.4f}".format(
                    symbols[i], str(i + 1).center(2), symbols[j],
                    str(j + 1).center(2), lis_alpha[i] + lis_beta[i], lis_alpha[i], lis_beta[i]))
            else:
                dia = 2 * np.trace(np.dot(aom[0][i], aom[0][j]))
                dib = 2 * np.trace(np.dot(aom[1][i], aom[1][j]))
                ditot = dia + dib
                print(" | {:>2}{:>2}-{:>2}{:>2}  {:8.4f}  {:8.4f}  {:8.4f}".format(
                    symbols[i], str(i + 1).center(2), symbols[j],
                    str(j + 1).center(2), ditot, dia, dib))
    print(" ------------------------------------------- ")
    print(" |    TOT:   {:>9.4f} {:>9.4f} {:>9.4f} ".format(
        Ntota + Ntotb, Ntota, Ntotb))
    print(" |    LOC:   {:>9.4f} {:>9.4f} {:>9.4f} ".format(
        listota + listotb, listota, listotb))
    print(" |  DELOC:   {:>9.4f} {:>9.4f} {:>9.4f} ".format(
        distota + distotb, distota, distotb))
    print(" ------------------------------------------- ")


def arom_unrest(aom, rings, molinfo, indicators, mci=False, av1245=False, partition=None, flurefs=None, homarefs=None,
                homerrefs=None, ncores=1, fragmap=None):
    """
    Outputs the aromaticity indices for unrestricted, single-determinant wavefunctions.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param rings: List containing the atoms in the ring.
    :type rings: list of lists
    :param molinfo: Contains the information of the molecule and the calculation.
    :type molinfo: dict
    :param indicators: Class containing the aromaticity indicators.
    :type indicators: class
    :param mci: If True, the MCI will be calculated.
    :type mci: bool, optional
    :param av1245: If True, the AV1245 will be calculated.
    :type av1245: bool, optional
    :param partition: The atomic partition used in the calculation.
    :type partition: str, optional
    :param flurefs: Dictionary with custom references for the FLU.
    :type flurefs: dict, optional
    :param homarefs: Dictionary with custom references for the HOMA.
    :type homarefs: dict, optional
    :param homerrefs: Dictionary with custom references for the HOMER.
    :type homerrefs: dict, optional
    :param ncores: Number of cores to use in the MCI calculation. By default, 1.
    :type ncores: int, optional
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
    if not molinfo:
        raise NameError(" 'molinfo' not found. Check input")

    symbols = molinfo["symbols"] + ["FF"] * (len(fragmap))
    partition = molinfo["partition"]

    # Checking if the list rings is contains more than one ring to analyze
    if not isinstance(rings[0], list):
        rings = [rings]

    # Looping through each of the rings
    for ring_index, ring in enumerate(rings.copy()):
        frag = False
        if any(tuple(r) in fragmap for r in ring if isinstance(r, (set, list))):
            frag = True
        connectivity = None if frag else [symbols[int(i) - 1] for i in ring]
        print(" ----------------------------------------------------------------------")
        print(" |")
        print(" | Ring  {} ({}):   {}".format(
            ring_index + 1, len(ring), "  ".join(str(num) for num in ring)))
        print(" |")
        print(" ----------------------------------------------------------------------")
        goodring = ring
        ring = list(np.arange(1, len(ring) + 1))

        # Starting the calculation of the aromaticity indicators
        if homarefs is not None:
            print(" | Using HOMA references provided by the user")
        else:
            print(" | Using default HOMA references")

        if frag:
            print(" | Could not compute geometric indicators between fragments")
            homa = None
        else:
            homa = indicators[ring_index].homa
        if homa is None:
            print(" | Connectivity could not match parameters")
        else:
            print(" | EN           {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].en))
            print(" | GEO          {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].geo))
            print(" | HOMA         {} =  {:>.6f}".format(ring_index + 1, homa))
            print(" ----------------------------------------------------------------------")

            if find_multiplicity(aom) == "triplet":
                print(" | Triplet AOMs. Computing HOMER")
                if homerrefs is not None:
                    print(" | Using HOMER references provided by the user")
                else:
                    print(" | Using the default HOMER references")

                print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].homer))

        if molinfo["geom"] is not None:
            pass
        else:
            bla = indicators[ring_index].bla
            if bla is None:
                pass
            else:
                bla_c = indicators[ring_index].bla_c
                print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, bla))
                print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, bla_c))
        print(" ----------------------------------------------------------------------")
        if not frag:
            flu = indicators[ring_index].flu
            if flu is None:
                print(" | Could not compute FLU")
            else:
                if flurefs is not None:
                    print(" | Using FLU references provided by the user")
                else:
                    print(" | Using the default FLU references")
                print(" | Atoms  :   {}".format("  ".join(str(atom) for atom in connectivity)))
                print(" |")
                print(" | *** FLU_ALPHA ***")
                print(" | FLU_aa       {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].flu_alpha))
                print(" |")
                print(" | *** FLU_BETA ***")
                print(" | FLU_bb       {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].flu_beta))
                print(" |")
                print(" | *** FLU_TOTAL ***")
                print(" | FLU          {} =  {:>.6f}".format(ring_index + 1, flu))
            print(" ----------------------------------------------------------------------")

        print(" |")
        print(" | *** BOA_ALPHA ***")
        print(" | BOA_aa       {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa_alpha))
        print(" | BOAc_aa      {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa_c_alpha))
        print(" |")
        print(" | *** BOA_BETA ***")
        print(" | BOA_bb       {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa_beta))
        print(" | BOAc_bb      {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa_c_beta))
        print(" |")
        print(" | *** BOA_TOTAL ***")
        print(" | BOA          {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa))
        print(" | BOAc         {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].boa_c))
        print(" ----------------------------------------------------------------------")

        # Printing the PDI

        if len(ring) != 6:
            print(" |   PDI could not be calculated as the number of centers is not 6")

        else:
            pdi_list_alpha = indicators[ring_index].pdi_list_alpha
            pdi_list_beta = indicators[ring_index].pdi_list_beta
            print(" |")
            print(" | *** PDI_ALPHA ***")
            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdi_list_alpha[0]))
            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdi_list_alpha[1]))
            print(" | DIaa ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdi_list_alpha[2]))
            print(" | PDI_alpha     {} =  {:.4f} ".format(ring_index + 1, indicators[ring_index].pdi_alpha))
            print(" |")
            print(" | *** PDI_BETA ***")
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdi_list_beta[0]))
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdi_list_beta[1]))
            print(" | DIbb ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdi_list_beta[2]))
            print(" | PDI_beta      {} =  {:.4f} ".format(ring_index + 1, indicators[ring_index].pdi_beta))
            print(" |")
            print(" | *** PDI_TOTAL ***")
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[0], ring[3], pdi_list_alpha[0] + pdi_list_beta[0]))
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[1], ring[4], pdi_list_alpha[1] + pdi_list_beta[1]))
            print(" | DI   ({:>2} -{:>2} )  =  {:.4f}".format(ring[2], ring[5], pdi_list_alpha[2] + pdi_list_beta[2]))
            print(" | PDI           {} =  {:.4f} ".format(ring_index + 1, indicators[ring_index].pdi))
            print(" ---------------------------------------------------------------------- ")

        if av1245 == True:
            if len(ring) < 6:
                print(" | AV1245 could not be calculated as the number of centers is smaller than 6 ")

            else:
                av1245_list_alpha = indicators[ring_index].av1245_list_alpha
                av1245_list_beta = indicators[ring_index].av1245_list_beta
                av1245_pairs, av1245_indices = [], []
                for i in range(len(goodring)):
                    first = fragmap[tuple(goodring[i % len(goodring)])] if isinstance(goodring[i % len(goodring)], set) else goodring[i % len(goodring)]
                    second = fragmap[tuple(goodring[(i + 1) % len(goodring)])] if isinstance(
                        goodring[(i + 1) % len(goodring)], set) else goodring[(i + 1) % len(goodring)]
                    third = fragmap[tuple(goodring[(i + 3) % len(goodring)])] if isinstance(
                        goodring[(i + 3) % len(goodring)], set) else goodring[(i + 3) % len(goodring)]
                    fourth = fragmap[tuple(goodring[(i + 4) % len(goodring)])] if isinstance(
                        goodring[(i + 4) % len(goodring)], set) else goodring[(i + 4) % len(goodring)]


                    # Create the ring with corresponding symbols
                    symbs = [
                        symbols[first - 1] if isinstance(first, int) else symbols[first],
                        symbols[second - 1] if isinstance(second, int) else symbols[second],
                        symbols[third - 1] if isinstance(third, int) else symbols[third],
                        symbols[fourth - 1] if isinstance(fourth, int) else symbols[fourth]
                    ]

                    av1245_pairs.append(symbs)
                    av1245_indices.append((first, second, third, fourth))

                print(" |")
                print(" | *** AV1245_ALPHA ***")
                for j in range(len(ring)):
                    symbs = av1245_pairs[j]
                    av1245_idx = av1245_indices[j]
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}".format(
                    str(av1245_idx[0]).rjust(2), symbs[0].ljust(2),
                    str(av1245_idx[1]).rjust(2), symbs[1].ljust(2),
                    str(av1245_idx[2]).rjust(2), symbs[2].ljust(2),
                    str(av1245_idx[3]).rjust(2), symbs[3].ljust(2),
                    av1245_list_alpha[(av1245_idx[0] - 1) % len(ring)]))
                print(" |")
                print(" | *** AV1245_BETA ***")
                for j in range(len(ring)):
                    symbs = av1245_pairs[j]
                    av1245_idx = av1245_indices[j]
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}".format(
                        str(av1245_idx[0]).rjust(2), symbs[0].ljust(2),
                        str(av1245_idx[1]).rjust(2), symbs[1].ljust(2),
                        str(av1245_idx[2]).rjust(2), symbs[2].ljust(2),
                        str(av1245_idx[3]).rjust(2), symbs[3].ljust(2),
                        av1245_list_beta[(av1245_idx[0] - 1) % len(ring)]))
                print(" | ")
                print(" | *** AV1245_TOTAL ***")
                for j in range(len(ring)):
                    symbs = av1245_pairs[j]
                    av1245_idx = av1245_indices[j]
                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}".format(
                        str(av1245_idx[0]).rjust(2), symbs[0].ljust(2),
                        str(av1245_idx[1]).rjust(2), symbs[1].ljust(2),
                        str(av1245_idx[2]).rjust(2), symbs[2].ljust(2),
                        str(av1245_idx[3]).rjust(2), symbs[3].ljust(2),
                        av1245_list_alpha[(av1245_idx[0] - 1) % len(ring)]
                        +av1245_list_beta[(av1245_idx[0] - 1) % len(ring)]))
            print(" |")
            print(" | AV1245 {} =             {:.4f}".format(ring_index + 1, indicators[ring_index].av1245))
            print(" |  AVmin {} =             {:.4f}".format(ring_index + 1, indicators[ring_index].avmin))
            print(" ---------------------------------------------------------------------- ")



        iring_alpha = indicators[ring_index].iring_alpha
        iring_beta = indicators[ring_index].iring_beta
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
                mci_alpha = indicators[ring_index].mci_alpha
                mci_beta = indicators[ring_index].mci_beta
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
                mci_alpha = indicators[ring_index].mci_alpha
                mci_beta = indicators[ring_index].mci_beta
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
