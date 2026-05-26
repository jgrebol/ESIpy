import numpy as np


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
    :param partition: Specifies the atom-in-molecule partition scheme.
                      Options include 'mulliken', 'lowdin', 'meta-lowdin', 'nao', and 'iao'.
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
    Computes the MCI for correlated wavefunctions sequentially by computing the Iring
    without storing the permutations. Default option if no number of cores is specified.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param partition: Specifies the atom-in-molecule partition scheme.
                      Options include 'mulliken', 'lowdin', 'meta-lowdin', 'nao', and 'iao'.
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
    Computes the MCI by generating all the permutations for a later distribution
    along the specified number of cores.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param ncores: Specifies the number of cores for multi-processing MCI calculation.
    :type ncores: int
    :param partition: Specifies the atom-in-molecule partition scheme.
                      Options include 'mulliken', 'lowdin', 'meta-lowdin', 'nao', and 'iao'.
    :type partition: str

    :returns: MCI value for the given ring.
    :rtype: float
    """
    from multiprocessing import get_context
    from math import factorial
    from functools import partial
    from itertools import permutations, islice

    ctx = get_context('spawn')
    dumb = partial(compute_iring, aom=aom)
    chunk_size = 50000

    iterable2 = islice(permutations(arr), factorial(len(arr) - 1))
    with ctx.Pool(processes=ncores) as pool:
        if partition in ('mulliken', "non-symmetric"):
            # We account for twice the value for symmetric AOMs
            val = 0.5 * sum(pool.imap(dumb, iterable2, chunk_size))
        else:  # Remove reversed permutations
            iterable2 = (x for x in iterable2 if x[1] < x[-1])
            val = sum(pool.imap(dumb, iterable2, chunk_size))
    return val


def multiprocessing_mci_no(arr, aom, ncores, partition):
    """
    Computes the MCI for correlated wavefunctions by generating all the permutations
    for a later distribution along the specified number of cores.

    :param arr: Contains the indices defining the ring connectivity.
    :type arr: list of int
    :param aom: Specifies the Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param ncores: Specifies the number of cores for multi-processing MCI calculation.
    :type ncores: int
    :param partition: Specifies the atom-in-molecule partition scheme.
                      Options include 'mulliken', 'lowdin', 'meta-lowdin', 'nao', and 'iao'.
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
        val = sum(pool.imap(dumb, iterable2, chunk_size))
        return 0.5 * val
    else:  # Remove reversed permutations
        iterable2 = (x for x in iterable2 if x[1] < x[-1])
        val = sum(pool.imap(dumb, iterable2, chunk_size))
        return val

