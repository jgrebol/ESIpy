import numpy as np
import multiprocessing as mp
from esipy.tools import wf_type


# ==============================================================================
# WORKER
# ==============================================================================

def _mci_worker(args):
    """
    Worker function to compute the partial trace sum.
    Efficiently handles both standard and correlated (NO) cases.
    """
    matrices, second_idx, prune_reversed = args
    n = len(matrices)

    # 1. Initialize DFS path: [0, second_idx]
    # Product = M[0] @ M[second_idx]
    current_prod = np.dot(matrices[0], matrices[second_idx])
    visited_mask = (1 << 0) | (1 << second_idx)

    # Stack: (depth, visited_mask, current_matrix_product)
    stack = [(2, visited_mask, current_prod)]

    local_trace_sum = 0.0

    while stack:
        depth, mask, prod = stack.pop()

        # BASE CASE: One node remaining (Leaf)
        if depth == n - 1:
            # Find the single unvisited node index
            rem_node = -1
            for i in range(n):
                if not (mask & (1 << i)):
                    rem_node = i
                    break

            # PRUNING (For Symmetric Partition)
            # Ensure we only count one direction (e.g., second < last)
            if prune_reversed and (second_idx > rem_node):
                continue

            # TRACE TRICK (O(N^2) instead of O(N^3))
            # Trace(A @ B) = Sum(A * B.T)
            # This works even for non-symmetric matrices (Natural Orbitals).
            term = np.sum(prod * matrices[rem_node].T)
            local_trace_sum += term
            continue

        # RECURSIVE STEP
        for i in range(1, n):
            if not (mask & (1 << i)):
                # Accumulate: New = Old @ Matrix_i
                new_prod = np.dot(prod, matrices[i])
                new_mask = mask | (1 << i)
                stack.append((depth + 1, new_mask, new_prod))

    return local_trace_sum


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def compute_mci(arr, aom, partition='mulliken', n_cores=None):
    """
    Unified MCI calculator.
    - Handles standard matrices AND Natural Orbitals (tuple input).
    - Auto-optimizes pre-multiplication of Occupation matrix.
    - Uses DFS and Multiprocessing.
    """
    # ---------------------------------------------------------
    # 1. Input Normalization & Optimization
    # ---------------------------------------------------------
    # Check if input is (matrices, occupation_matrix)
    if isinstance(aom, (tuple, list)) and len(aom) == 2 and not isinstance(aom[0], np.ndarray):
        real_aoms, occ = aom
        # OPTIMIZATION: Pre-multiply Occupation @ AOM once.
        # This moves the cost from O(N!) inside the loop to O(N) here.
        matrices = [np.dot(occ, real_aoms[idx - 1]) for idx in arr]
    else:
        # Standard case
        matrices = [aom[idx - 1] for idx in arr]

    n = len(arr)
    if n < 3: return 0.0

    # ---------------------------------------------------------
    # 2. Setup Logic
    # ---------------------------------------------------------
    # If partition is symmetric, we PRUNE reversed paths.
    # If Mulliken, we sum ALL paths (and divide by 2 later).
    prune = (partition == 'symmetric')

    tasks = [(matrices, sec, prune) for sec in range(1, n)]
    total_trace = 0.0

    # ---------------------------------------------------------
    # 3. Execution
    # ---------------------------------------------------------
    if n_cores is None:
        n_cores = mp.cpu_count()

    # Use multiprocessing only if worthwhile
    if n_cores > 1 and len(tasks) > 1:
        with mp.Pool(processes=n_cores) as pool:
            # chunksize=1 is good for heavy tasks
            results = pool.map(_mci_worker, tasks, chunksize=1)
        total_trace = sum(results)
    else:
        for task in tasks:
            total_trace += _mci_worker(task)

    # Normalization of SD-wf is 2**(n-1) but from permutations we double count
    if wf_type(aom) == "rest" or wf_type(aom) == "unrest":
        prefactor = 2 ** (n - 2)
    else:
        # Prefactor just for the generation of permutations. We already make use of the NO occupations precomputing occ@AOM
        prefactor = 0.5

    return prefactor * total_trace


def sequential_mci(arr, aom, partition='mulliken'):
    return compute_mci(arr, aom, partition=partition, n_cores=1)


def sequential_mci_no(arr, aom, partition='mulliken'):
    return compute_mci(arr, aom, partition=partition, n_cores=1)


def multiprocessing_mci(arr, aom, ncores, partition='mulliken'):
    return compute_mci(arr, aom, partition=partition, n_cores=ncores)


def multiprocessing_mci_no(arr, aom, ncores, partition='mulliken'):
    return compute_mci(arr, aom, partition=partition, n_cores=ncores)
