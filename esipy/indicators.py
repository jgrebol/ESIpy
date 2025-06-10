import numpy as np

from esipy.tools import find_dis, find_di, find_di_no, find_lis, find_ns, find_distances, av1245_pairs


########## Iring ###########

# Computing the Iring (Restricted and Unrestricted)

def compute_iring(arr, aom):
    """
    Calculation of the Iring aromaticity index.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices

    :returns: The Iring for the given ring connectivity.
    :rtype: float
    """

    product = np.identity(aom[0].shape[0])
    for i in arr:
        product = np.dot(product, aom[i - 1])
    iring = (2 ** (len(arr) - 1)) * np.trace(product)

    return iring


def compute_iring_no(arr, aom):
    """
    Calculation of the Iring aromaticity index for correlated wavefunctions.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices

    :returns: The Iring for the given ring connectivity.
    :rtype: float
    """

    aom, occ = aom
    product = np.identity(aom[0].shape[0])
    for i in arr:
        product = np.dot(product, np.dot(occ, aom[i - 1]))
    return np.trace(product)


########### MCI ###########

def sequential_mci(arr, aom, partition):
    """
    Computes the MCI sequentially by computing the Iring without storing the permutations.
    Default option if no number of cores is specified.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param partition: Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
    :type partition: str

    :returns: MCI value for the given ring.
    :rtype: float
    """

    from math import factorial
    from itertools import permutations, islice

    iterable2 = islice(permutations(arr), factorial(len(arr) - 1))
    if partition == 'mulliken' or partition == "non-symmetric":
        # We account for twice the value for symmetric AOMs
        val = 0.5 * sum(compute_iring(p, aom) for p in iterable2)
        return val
    else:  # Remove reversed permutations
        iterable2 = (x for x in iterable2 if x[1] < x[-1])
        val = sum(compute_iring(p, aom) for p in iterable2)
        return val


def sequential_mci_no(arr, aom, partition):
    """
    Computes the MCI for correlated wavefunctions sequentially by computing the Iring without storing the permutations.
    Default option if no number of cores is specified.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param partition: Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
    :type partition: str

    :returns: MCI value for the given ring.
    :rtype: float
    """

    from math import factorial
    from itertools import permutations, islice

    iterable2 = islice(permutations(arr), factorial(len(arr) - 1))
    if partition == 'mulliken' or partition == "non-symmetric":
        # We account for twice the value for symmetric AOMs
        val = 0.5 * sum(compute_iring_no(p, aom) for p in iterable2)
        return val
    else:  # Remove reversed permutations
        iterable2 = (x for x in iterable2 if x[1] < x[-1])
        val = sum(compute_iring_no(p, aom) for p in iterable2)
        return val


def multiprocessing_mci(arr, aom, ncores, partition):
    """
       Computes the MCI by generating all the permutations for a later distribution along the specified number of cores.

       :param arr: Contains the indices defining the ring connectivity.
       :type arr: list of int
       :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
       :type aom: list of matrices
       :param ncores: Specifies the number of cores for multi-processing MCI calculation.
       :type ncores: int
       :param partition: Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
       :type partition: str

       :returns: MCI value for the given ring.
       :rtype: float
    """

    from multiprocessing import Pool
    from math import factorial
    from functools import partial
    from itertools import permutations, islice

    pool = Pool(processes=ncores)
    dumb = partial(compute_iring, aom=aom)
    chunk_size = 50000

    iterable2 = islice(permutations(arr), factorial(len(arr) - 1))
    if partition == 'mulliken' or partition == "non-symmetric":
        # We account for twice the value for symmetric AOMs
        val = 0.5 * sum(pool.imap(dumb, iterable2, chunk_size))
        return val
    else:  # Remove reversed permutations
        iterable2 = (x for x in iterable2 if x[1] < x[-1])
        val = sum(pool.imap(dumb, iterable2, chunk_size))
        return val

def multiprocessing_mci_no(arr, aom, ncores, partition):
    """
       Computes the MCI for correlated wavefunctions by generating all the permutations for a later distribution along the specified number of cores.

       :param arr: Contains the indices defining the ring connectivity.
       :type arr: list of int
       :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
       :type aom: list of matrices
       :param ncores: Specifies the number of cores for multi-processing MCI calculation.
       :type ncores: int
       :param partition: Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
       :type partition: str

       :returns: MCI value for the given ring.
       :rtype: float
    """

    from multiprocessing import Pool
    from math import factorial
    from functools import partial
    from itertools import permutations, islice

    pool = Pool(processes=ncores)
    dumb = partial(compute_iring_no, aom=aom)
    chunk_size = 50000

    iterable2 = islice(permutations(arr), factorial(len(arr) - 1))
    if partition == "mulliken":
        # We account for twice the value for symmetric AOMs
        val = 0.5 * sum(pool.imap(dumb, iterable2, chunk_size))
        return val
    else:  # Remove reversed permutations
        iterable2 = (x for x in iterable2 if x[1] < x[-1])
        val = sum(pool.imap(dumb, iterable2, chunk_size))
        return val

########### AV1245 ###########

# Calculation of the AV1245 index (Restricted and Unrestricted)

def compute_av1245(arr, aom, partition):
    """
     Computes the AV1245 and AVmin indices. Not available for rings smaller than 6 members.

     :param arr: Contains the indices defining the ring connectivity.
     :type arr: list of int
     :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
     :type aom: list of matrices
     :param partition: Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
     :type partition: str

     :returns: The AV1245 index, the AVmin index, and each of the AV1245 in a list for the output, respectively.
     :rtype: tuple
    """

    products = []
    for cp in av1245_pairs(arr.copy()):
        product = sequential_mci(list(cp), aom, partition)
        products.append(1000 * product / 3)

    av1245_value = np.mean(products)
    avmin_value = min(products, key=abs)

    return av1245_value, avmin_value, products


def compute_av1245_no(arr, aom, partition):
    """
     Computes the AV1245 and AVmin indices for correlated wavefunctions. Not available for rings smaller than 6 members.

     :param arr: Contains the indices defining the ring connectivity.
     :type arr: list of int
     :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
     :type aom: list of matrices
     :param partition: Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
     :type partition: str

     :returns: The AV1245 index, the AVmin index, and each of the AV1245 in a list for the output, respectively.
     :rtype: tuple
     """

    products = []
    for cp in av1245_pairs(arr):
        product = sequential_mci_no(list(cp), aom, partition)
        products.append(1000 * product / 3)

    av1245_value = np.mean(products)
    avmin_value = min(products, key=abs)

    return av1245_value, avmin_value, products


########### PDI ###########

# Calculation of the PDI (Restricted and Unrestricted)

def compute_pdi(arr, aom):
    """
    Computes the PDI for the given 6-membered ring connectivity. Only computed for rings n=6.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices

    :returns: The PDI value and each of the DIs in para position.
    :rtype: tuple
    """

    if len(arr) == 6:
        pdi_a = 2 * np.trace(np.dot(aom[arr[0] - 1], aom[arr[3] - 1]))
        pdi_b = 2 * np.trace(np.dot(aom[arr[1] - 1], aom[arr[4] - 1]))
        pdi_c = 2 * np.trace(np.dot(aom[arr[2] - 1], aom[arr[5] - 1]))
        pdi_value = (pdi_a + pdi_b + pdi_c) / 3

        return pdi_value, [pdi_a, pdi_b, pdi_c]

    else:
        return None


def compute_pdi_no(arr, aom):
    """
    Computes the PDI for the given 6-membered ring connectivity. Only computed for rings n=6.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices

    :returns: The PDI value and each of the DIs in para position.
    :rtype: tuple
    """
    aom, occ = aom

    if len(arr) == 6:

        i1, i2, i3, i4, i5, i6 = arr[0] - 1, arr[1] - 1, arr[2] - 1, arr[3] - 1, arr[4] - 1, arr[5] - 1
        pdi_a = 2 * np.trace(np.linalg.multi_dot((occ ** (1 / 2), aom[i1], occ ** (1 / 2), aom[i4])))
        pdi_b = 2 * np.trace(np.linalg.multi_dot((occ ** (1 / 2), aom[i2], occ ** (1 / 2), aom[i5])))
        pdi_c = 2 * np.trace(np.linalg.multi_dot((occ ** (1 / 2), aom[i3], occ ** (1 / 2), aom[i6])))
        pdi_value = (pdi_a + pdi_b + pdi_c) / 3

        return pdi_value, [pdi_a, pdi_b, pdi_c]

    else:
        return None

########### FLU ###########

# Calculation of the FLU (Restricted and Unrestricted)

def find_flurefs(partition=None):
    """
    Sets the reference of the FLU index based on the provided partition.
    The available options are "CC" from benzene, "CN" from pyridine,
    "BN" from borazine, "NN" from pyridazine and "CS" from thiophene,
    all obtained from optimized and single-point calculations at HF/6-31G(d)
    level of theory.

    :param partition: Specifies the atom-in-molecule partition scheme.
                      Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
    :type partition: str

    :returns: Contains the reference DI for each bond.
    :rtype: dict
    """

    if partition == "qtaim":
        return {"CC": 1.3993, "CN": 1.1958, "BN": 0.3934, "NN": 1.5252, "CS": 1.2369}

    elif partition == "mulliken":
        return {"CC": 1.4530, "CN": 1.4149, "BN": 1.0944, "NN": 1.3044, "CS": 1.1024}

    elif partition == "lowdin":
        return {"CC": 1.5000, "CN": 1.6257, "BN": 1.6278, "NN": 1.5252, "CS": 1.2675}

    elif partition == "meta_lowdin":
        return {"CC": 1.4394, "CN": 1.4524, "BN": 1.3701, "NN": 1.5252, "CS": 1.1458}

    elif partition == "nao":
        return {"CC": 1.4338, "CN": 1.4117, "BN": 0.9238, "NN": 1.3706, "CS": 1.1631}

    elif partition == "iao":
        return {"CC": 1.4378, "CN": 1.4385, "BN": 1.1638, "NN": 1.3606, "CS": 1.1436}


def compute_flu(arr, molinfo, aom, flurefs=None, partition=None):
    """
    Computes the FLU index.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param molinfo: Contains the molecular information.
    :type molinfo: dict
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param flurefs: User-provided references for the FLU index.
    :type flurefs: dict, optional
    :param partition: Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.
    :type partition: str, optional

    :returns: The FLU value for the given ring connectivity.
    :rtype: float
    """

    flu_value, flu_polar = 0, 0
    symbols = molinfo["symbols"]
    atom_symbols = [symbols[int(i) - 1] for i in arr]
    bond_types = ["".join(sorted([atom_symbols[i].upper(), atom_symbols[(i + 1) % len(arr)].upper()]))
                  for i in range(len(arr))]

    # Setting and update of the reference values
    flu_refs = find_flurefs(partition)
    if flurefs is not None:
        flu_refs.update(flurefs)
    flu_refs = {key.upper(): value for key, value in flu_refs.items()}

    dis = find_dis(arr, aom)
    lis = find_lis(arr, aom)
    ns = find_ns(arr, aom)
    for i in range(len(arr)):
        if bond_types[i] not in flu_refs:
            print(f" | No parameters found for bond type {bond_types[i]}")
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

def compute_boa(arr, aom):
    """
    Computes the BOA and BOA\_c indices.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices

    :returns: Contains the BOA and the BOA\_c indices, respectively.
    :rtype: tuple
    """

    n1 = len([i for i in arr if i % 2 != 0])
    n2 = len([i for i in arr if i % 2 == 1])

    sum_odd = sum(find_di(aom, arr[i - 1], arr[i]) for i in range(0, len(arr), 2))
    sum_even = sum(find_di(aom, arr[i + 1], arr[i]) for i in range(0, len(arr) - 1, 2))
    boa = abs(sum_odd / n1 - sum_even / n2)

    boa_c = 0
    for i in range(len(arr)):
        diff_di = abs(
            find_di(aom, arr[i - 1], arr[i]) - find_di(aom, arr[(i + 1) % len(arr) - 1], arr[(i + 1) % len(arr)]))
        boa_c += diff_di / len(arr)
    return boa, boa_c


def compute_boa_no(arr, aom):
    """
    Computes the BOA and BOA\_c indices for correlated wavefunctions.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices

    :returns: Contains the BOA and the BOA\_c indices, respectively.
    :rtype: tuple
    """

    n1 = len([i for i in arr if i % 2 != 0])
    n2 = len([i for i in arr if i % 2 == 1])

    sum_odd = sum(find_di_no(aom, arr[i - 1], arr[i]) for i in range(0, len(arr), 2))
    sum_even = sum(find_di_no(aom, arr[i + 1], arr[i]) for i in range(0, len(arr) - 1, 2))
    boa = abs(sum_odd / n1 - sum_even / n2)

    boa_c = 0
    for i in range(len(arr)):
        diff_di = abs(
            find_di_no(aom, arr[i - 1], arr[i]) - find_di_no(aom, arr[(i + 1) % len(arr) - 1], arr[(i + 1) % len(arr)]))
        boa_c += diff_di / len(arr)
    return boa, boa_c


######## GEOMETRIC INDICES ########

# Calculation of the HOMA and/or HOMER indices (Restricted and Unrestricted)

def compute_homer(arr, molinfo, homerrefs=None):
    """
    Computes the HOMER index.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param molinfo: Contains the molecular information.
    :type molinfo: dict
    :param homerrefs: User-provided references for the HOMER index.
    :type homerrefs: dict, optional

    :returns: HOMER value for the given ring connectivity.
    :rtype: float
    """

    geom = molinfo["geom"]
    refs = {
        "CC": {"alpha": 950.74, "r_opt": 1.437},
        "CN": {"alpha": 506.43, "r_opt": 1.390},
        "NN": {"alpha": 187.36, "r_opt": 1.375},
        "CO": {"alpha": 164.96, "r_opt": 1.379}
    }
    if homerrefs is not None:
        refs.update(homerrefs)

    atom_symbols = molinfo["symbols"]
    bond_types = ["".join(sorted([atom_symbols[arr[i] - 1], atom_symbols[arr[(i + 1) % len(arr)] - 1]]))
                  for i in range(len(arr))]

    for i in range(len(arr)):
        if bond_types[i] not in refs:
            print(f"No parameters found for bond type {bond_types[i]}")
            return None

    alpha = refs[bond_types[i]]["alpha"]
    r_opt = refs[bond_types[i]]["r_opt"]

    distances = find_distances(arr, geom)
    diff = np.mean([r_opt - distances[i] for i in range(len(arr))])
    homer_value = 1 - alpha * diff ** 2

    return homer_value


def compute_homa(arr, molinfo, homarefs=None):
    """
    Computes the HOMA index.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param molinfo: Contains the molecular information.
    :type molinfo: dict
    :param homarefs: User-provided references for the HOMA index.
    :type homarefs: dict, optional

    :returns: HOMA value for the given ring connectivity.
    :rtype: float
    """

    refs = {
        "CC": {"n_opt": 1.590, "c": 0.1702, "r1": 1.467},
        "CN": {"n_opt": 1.589, "c": 0.2828, "r1": 1.465},
        "CO": {"n_opt": 1.602, "c": 0.2164, "r1": 1.367},
        "CP": {"n_opt": 1.587, "c": 0.2510, "r1": 1.814},
        "CS": {"n_opt": 1.584, "c": 0.2828, "r1": 1.807},
        "NN": {"n_opt": 1.590, "c": 0.2395, "r1": 1.420},
        "NO": {"n_opt": 1.586, "c": 0.3621, "r1": 1.415},
        "CSe": {"n_opt": 1.590, "c": 0.2970, "r1": 1.959},  # n_opt taken as C-C
        "BB": {"n_opt": 1.590, "c": 0.2510, "r1": 1.814},  # n_opt taken as C-C
        "BC": {"n_opt": 1.590, "c": 0.1752, "r1": 1.647},  # n_opt taken as C-C
        "BN": {"n_opt": 1.590, "c": 0.2900, "r1": 1.564},  # n_opt taken as C-C
        "alpha": 257.7, "r_opt": 1.388
    }
    if homarefs is not None:
        refs.update(homarefs)
    refs = {key.upper(): value for key, value in refs.items()}

    geom = molinfo["geom"]
    if geom is None:
        return None
    symbols = molinfo["symbols"]

    atom_symbols = [symbols[int(i) - 1] for i in arr]
    bond_types = ["".join(sorted([atom_symbols[i].upper(), atom_symbols[(i + 1) % len(arr)].upper()]))
                  for i in range(len(arr))]

    for i in range(len(arr)):
        if bond_types[i] not in refs:
            print(f" | No parameters found for bond type {bond_types[i]}")
            return None

    distances = find_distances(arr, geom)
    alpha = refs["ALPHA"]
    r_opt = refs["R_OPT"]

    ravs, bonds = [], []
    for i in range(len(arr)):
        c = refs[bond_types[i]]["c"]
        r1 = refs[bond_types[i]]["r1"]

        bond = np.exp((r1 - distances[i]) / c)
        rn = 1.467 - 0.1702 * np.log(bond)
        ravs.append(rn)

    rav = sum(ravs) / len(arr)

    if np.mean(rav) > r_opt:
        EN = alpha * (r_opt - rav) ** 2
    else:
        EN = -alpha * (r_opt - rav) ** 2

    GEO = 0
    for i in range(len(arr)):
        GEO += (rav - ravs[i]) ** 2
    GEO = GEO * alpha / len(arr)

    homa_value = 1 - (EN + GEO)
    return homa_value, EN, GEO


# Calculation of the BLA (Restricted and Unrestricted)

def compute_bla(arr, molinfo):
    """
    Computes the BLA and BLA\_c indices.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param molinfo: Contains the molecular information.
    :type molinfo: dict

    :returns: Contains the BLA and the BLA\_c indices, respectively.
    :rtype: tuple
    """

    if molinfo["geom"] is None:
        return None

    distances = find_distances(arr, molinfo["geom"])

    sum1 = sum(distances[i] for i in range(0, len(arr), 2))
    sum2 = sum(distances[i] for i in range(1, len(arr), 2))

    bla = abs(sum1 / (len(arr) // 2) - sum2 / (len(arr) - len(arr) // 2))

    bla_c = 0
    for i in range(len(arr)):
        bla_c += abs(distances[i] - distances[(i + 1) % len(distances)]) / len(distances)

    return bla, bla_c
