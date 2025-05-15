import numpy as np

from esipy.tools import format_partition


def info_no(aom, molinfo, nfrags=0):
    """
    Prints the initial information for correlated wavefunctions.

    :param aom: Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices or str
    :param molinfo: Contains the information about the molecule and the calculation.
    :type molinfo: dict
    """

    aom, occ = aom
    partition = format_partition(molinfo["partition"])
    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    print(" | Number of Atoms:          {}".format(len(aom)-nfrags))
    print(" | Occ. Mol. Orbitals:       {}".format(np.shape(aom[0])[0]))
    print(" | Wavefunction type:        Natural Orbitals")
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
    trace = np.sum([np.trace(matrix) for matrix in aom][:len(aom)-nfrags])
    print(" | Tr(Enter):    {:.13f}".format(trace))
    print(" ------------------------------------------- ")


def deloc_no(aom, molinfo, fragmap={}):
    """
    Population analysis, localization and delocalization indices for correlated wavefunctions.

    :param aom: Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices or str
    :param molinfo: Contains the information about the molecule and the calculation.
    :type molinfo: dict
    """

    aom, occ = aom
    presymbols = molinfo["symbols"]
    symbols = presymbols + ["FF"] * (len(aom)-len(presymbols))
    if len(aom)-len(presymbols) > 0:
        print(" | WARNING: Beta version of fragments with correlated wavefunctions")
        print(" |          Results may not be accurate")

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
                if symbols[j] != "FF":
                    dlocF += dif
                    dlocX += dix
                    difs.append(dlocF)
                    dixs.append(dlocX)

        print(" | {:>2} {:>2d}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}".format(
            symbols[i], i + 1, N[i], N[i]-lif, dlocX, lif, lix))
    #print(" ---------------------------------------------------------- ")
    Ntot = np.sum(N[:len(presymbols)])
    lifstot = np.sum(lifs[:len(presymbols)])
    lixstot = np.sum(lixs[:len(presymbols)])
    dixstot = np.sum(dixs[:len(presymbols)])
    difstot = Ntot - lifstot
    print(" | TOT:   {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}".format(
        Ntot, Ntot-lifstot, dixstot, lifstot, lixstot))
    print(" ---------------------------------------------------------- ")

    print(" ---------------------------------- ")
    print(" |    Pair       DI(F)     DI(X) ")
    print(" ---------------------------------- ")
    for i in range(len(aom)):
        for j in range(i, len(aom)):
            if symbols[i] == "FF" or symbols[j] == "FF":
                continue
            if i == j:
                print(" | {:>2}{:>2}-{:>2}{:>2}  {:>8.4f}  {:>8.4f}".format(
                    symbols[i], i + 1, symbols[j], j + 1, lifs[i], lixs[i]))
            else:
                dif = 2 * np.trace(np.linalg.multi_dot((occ ** (1 / 2), aom[i], occ ** (1 / 2), aom[j])))
                dix = np.trace(np.linalg.multi_dot((occ, aom[i], occ, aom[j])))
                if symbols[i] != "FF" and symbols[j] != "FF":  # Exclude FF atoms from contributing
                    print(" | {:>2}{:>2}-{:>2}{:>2}  {:>8.4f}  {:>8.4f}".format(
                        symbols[i], i + 1, symbols[j], j + 1, dif, dix))
    print(" ---------------------------------- ")
    print(" |   TOT:     {:>8.4f}  {:>8.4f}  ".format(Ntot, lixstot + dixstot))
    print(" |   LOC:     {:>8.4f}  {:>8.4f} ".format(lifstot, lixstot))
    print(" | DELOC:     {:>8.4f}  {:>8.4f} ".format(difstot, dixstot))
    print(" ---------------------------------- ")


def arom_no(rings, molinfo, indicators, mci=False, av1245=False, partition=None, flurefs=None, homarefs=None,
            homerrefs=None, ncores=1, fragmap=None):
    """
    Output for the aromaticity indices for Natural Orbitals calculations. Will use Fulton's approximation.

    :param rings: Contains the indices of the atoms in the rings.
    :type rings: list of lists
    :param molinfo: Contains the information about the molecule and the calculation.
    :type molinfo: dict
    :param indicators: Contains the aromaticity indicators.
    :type indicators: class
    :param mci: If True, the MCI is computed.
    :type mci: bool, optional
    :param av1245: If True, the AV1245 is computed.
    :type av1245: bool, optional
    :param partition: Contains the name of the partition.
    :type partition: str, optional
    :param flurefs: Contains the custom references for the FLU aromaticity index.
    :type flurefs: dict, optional
    :param homarefs: Contains the custom references for the HOMA aromaticity index.
    :type homarefs: dict, optional
    :param homerrefs: Contains the custom references for the HOMER aromaticity index.
    :type homerrefs: dict, optional
    :param ncores: Number of cores to use in the MCI calculation. By default, 1.
    :type ncores: int, optional
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
    if not molinfo:
        raise NameError(" 'molinfo' not found. Check input")

    symbols = molinfo["symbols"] + ["FF"] * (len(fragmap))
    partition = molinfo["partition"]

    if not isinstance(rings[0], list):
        rings = [rings]

    # Looping through each of the rings
    for ring_index, ring in enumerate(rings):
        frag = False
        if any(tuple(r) in fragmap for r in ring if isinstance(r, (set, list))):
            frag = True
        connectivity = None if frag else [symbols[int(i) - 1] for i in ring]
        print(" ----------------------------------------------------------------------")
        print(" |")
        print(" | Ring  {} ({}):   {}".format(ring_index + 1, len(ring), "  ".join(str(num) for num in ring)))
        print(" |")
        print(" ----------------------------------------------------------------------")
        goodring = ring
        ring = list(np.arange(1, len(ring) + 1))
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
            if homerrefs:
                print(" | ")
                print(" | Found custom HOMER references 'alpha' and 'r_opt'. Computing")
                print(" | HOMER        {} =  {:>.6f}".format(ring_index + 1, indicators[ring_index].homer))

            print(" ----------------------------------------------------------------------")
            if molinfo["geom"] is not None:
                pass
            else:
                bla = indicators[ring_index].bla
                if bla[0] is None:
                    pass
                else:
                    bla_c = indicators[ring_index].bla_c
                    print(" | BLA          {} =  {:>.6f}".format(ring_index + 1, bla))
                    print(" | BLAc         {} =  {:>.6f}".format(ring_index + 1, bla_c))

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

                    print(" |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}".format(
                        str(av1245_indices[-1][0]).rjust(2), symbs[0].ljust(2),
                        str(av1245_indices[-1][1]).rjust(2), symbs[1].ljust(2),
                        str(av1245_indices[-1][2]).rjust(2), symbs[2].ljust(2),
                        str(av1245_indices[-1][3]).rjust(2), symbs[3].ljust(2),
                        av1245_list[(av1245_indices[-1][0] - 1) % len(ring)]))

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
