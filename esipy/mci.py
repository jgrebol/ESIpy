import numpy as np
import multiprocessing as mp
from math import factorial
from time import time

from esipy.tools import (
    wf_type, mapping, filter_connec, find_node_distances, build_connectivity
)

def _kernel_exact(args):
    """DFS worker for exact MCI."""
    mats, j, sym_prune = args
    n = len(mats)

    # Init path: 0 -> j
    P = np.dot(mats[0], mats[j])
    mask = (1 << 0) | (1 << j)

    # Stack: (depth, visited_mask, current_product)
    stack = [(2, mask, P)]
    tr_sum = 0.0

    while stack:
        depth, mask, P = stack.pop()

        # Leaf node
        if depth == n - 1:
            # Find the single unvisited node index
            rem = 0
            while (mask >> rem) & 1:
                rem += 1

            # For symmetric AOMs, consider only those with the second node index less than the last
            if sym_prune and (j > rem):
                continue

            # Trace contraction: Tr(A @ B) = sum(A * B.T)
            # Reduced time complexity if we want to keep only trace from O(N^3) to O(N^2)
            tr_sum += np.sum(P * mats[rem].T)
            continue

        # Recurse like James Harden, step back
        for i in range(1, n):
            if not ((mask >> i) & 1):
                stack.append((depth + 1, mask | (1 << i), np.dot(P, mats[i])))

    return tr_sum


def _prep_matrices(arr, aom):
    """Normalize input into list of matrices, handling NO tuples."""
    if isinstance(aom, (tuple, list)) and len(aom) == 2 and not isinstance(aom[0], np.ndarray):
        # For Natural Orbitals: Pre-multiply Occ @ AOM
        real_aoms, occ = aom
        return [np.dot(occ, real_aoms[idx - 1]) for idx in arr]
    return [aom[idx - 1] for idx in arr]


def mci(arr, aom, partition='mulliken', n_cores=None):
    """
    Computes Exact MCI using DFS.

    Parameters
    ----------
    arr : list
        Atom indices involved in the ring.
    aom : np.ndarray or tuple
        Atomic Overlap Matrices. If tuple (AOMs, Occ), assumes Natural Orbitals.
    partition : str
        'mulliken', 'symmetric', etc. Controls path pruning.
    n_cores : int, optional
        Number of processes. Defaults to cpu_count.
    """
    mats = _prep_matrices(arr, aom)
    n = len(arr)
    if n < 3: return 0.0

    # Symmetric AOMs?
    sym_prune = (partition == 'symmetric')

    # Tasks: Iterate over the second node in the path (0 -> j -> ...)
    tasks = [(mats, j, sym_prune) for j in range(1, n)]

    if n_cores is None:
        n_cores = mp.cpu_count()

    if n_cores > 1 and len(tasks) > 1:
        with mp.Pool(n_cores) as pool:
            results = pool.map(_kernel_exact, tasks, chunksize=1)
        total_trace = sum(results)
    else:
        total_trace = sum(_kernel_exact(t) for t in tasks)

    # Normalization
    # SD-wf is 2**(n-1), but we double count perms in non-symmetric cases
    is_sd = wf_type(aom) in ["rest", "unrest"]
    prefactor = 2 ** (n - 2) if is_sd else 0.5

    return prefactor * total_trace



def sequential_mci(arr, aom, partition='mulliken'):
    return mci(arr, aom, partition, n_cores=1)


def multiprocessing_mci(arr, aom, ncores, partition='mulliken'):
    return mci(arr, aom, partition, n_cores=ncores)


# Aliases for Natural Orbitals
sequential_mci_no = sequential_mci
multiprocessing_mci_no = multiprocessing_mci
aproxmci = mci_approx
