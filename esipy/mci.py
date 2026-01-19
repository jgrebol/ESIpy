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


def _kernel_approx(args):
    """
    DFS worker with distance/parity filtering.
    args: (mats, dists, start_idx, alg_type, d_cut, sym_prune)
    """
    mats, dists, j, alg, d_cut, sym_prune = args
    n = len(mats)

    # 1. Filter Initial Step (0 -> j)
    d0 = dists[0, j]
    if d0 > d_cut: return 0.0, 0

    # Parity checks: Alg 3 (Odd only), Alg 4 (Even only)
    if alg == 3 and (d0 % 2 == 0): return 0.0, 0
    if alg == 4 and (d0 % 2 != 0): return 0.0, 0

    cur_max = d0
    P = np.dot(mats[0], mats[j])
    mask = (1 << 0) | (1 << j)

    # Stack: (depth, mask, product, prev_node, current_max_dist)
    stack = [(2, mask, P, j, cur_max)]

    tr_sum = 0.0
    n_perms = 0

    while stack:
        depth, mask, P, prev, cur_max = stack.pop()

        # Leaf node
        if depth == n - 1:
            rem = 0
            while (mask >> rem) & 1:
                rem += 1

            if sym_prune and (j > rem): continue

            # Validate closing edges (prev->rem, rem->0)
            d_last = dists[prev, rem]
            d_close = dists[rem, 0]

            if d_last > d_cut or d_close > d_cut: continue

            if alg == 3 and ((d_last % 2 == 0) or (d_close % 2 == 0)): continue
            if alg == 4 and ((d_last % 2 != 0) or (d_close % 2 != 0)): continue

            # Alg 2: Cycle max distance must equal cutoff d
            if alg == 2:
                if max(cur_max, d_last, d_close) != d_cut: continue

            tr_sum += np.sum(P * mats[rem].T)
            n_perms += 1
            continue

        # Recurse
        for i in range(1, n):
            if not ((mask >> i) & 1):
                d_step = dists[prev, i]

                if d_step > d_cut: continue
                if alg == 3 and (d_step % 2 == 0): continue
                if alg == 4 and (d_step % 2 != 0): continue

                # For Alg 2, propagate max dist
                new_max = max(cur_max, d_step) if alg == 2 else 0

                stack.append((depth + 1, mask | (1 << i), np.dot(P, mats[i]), i, new_max))

    return tr_sum, n_perms

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


def mci_approx(arr, aom, partition=None, alg=0, d=1, n_cores=1,
               rings_thres=0.3, connec=None, **kwargs):
    """
    Computes Approximate MCI with topological filtering.

    alg (int): 0=Exact, 1=Dist, 2=MaxDist, 3=Odd, 4=Even
    """
    t0 = time()

    # Exact if alg=0
    if alg == 0:
        val = mci(arr, aom, partition, n_cores)
        # Factorial scaling for display purposes
        nperms = factorial(len(arr) - 1)
        if partition != "mulliken" and partition != "non-symmetric":
            nperms *= 0.5
        return val, nperms, time() - t0

    if connec is None:
        connec = build_connectivity(aom, rings_thres)

    # Generate distance matrix for the relevant subgraph
    full_dists = find_node_distances(filter_connec(connec))
    n = len(arr)
    sub_dists = np.array([[full_dists[arr[i]][arr[j]] for j in range(n)] for i in range(n)])

    mats = _prep_matrices(arr, aom)
    if n < 3: return 0.0, 0, time() - t0

    # 2. Prepare Tasks
    is_mulliken = (partition in ["mulliken", "non-symmetric"])
    sym_prune = not is_mulliken

    tasks = [(mats, sub_dists, j, alg, d, sym_prune) for j in range(1, n)]

    total_trace = 0.0
    total_perms = 0

    if n_cores > 1 and len(tasks) > 1:
        with mp.Pool(n_cores) as pool:
            results = pool.map(_kernel_approx, tasks, chunksize=1)
        for t, p in results:
            total_trace += t
            total_perms += p
    else:
        for t in tasks:
            t_val, p_count = _kernel_approx(t)
            total_trace += t_val
            total_perms += p_count

    # 3. Normalization
    is_sd = wf_type(aom) in ["rest", "unrest"]
    prefactor = 2 ** (n - 1) if is_sd else 0.5

    if is_mulliken:
        total_trace *= 0.5

    return prefactor * total_trace, total_perms, time() - t0

def sequential_mci(arr, aom, partition='mulliken'):
    return mci(arr, aom, partition, n_cores=1)


def multiprocessing_mci(arr, aom, ncores, partition='mulliken'):
    return mci(arr, aom, partition, n_cores=ncores)


# Aliases for Natural Orbitals
sequential_mci_no = sequential_mci
multiprocessing_mci_no = multiprocessing_mci
aproxmci = mci_approx
