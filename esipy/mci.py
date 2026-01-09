import numpy as np
import multiprocessing as mp
from math import factorial
from time import time
from functools import partial

# Imports from your library
from esipy.tools import (
    wf_type, mapping, filter_connec, find_node_distances, build_connectivity
)


# ==============================================================================
# WORKER 1: EXACT MCI (Original, Unchanged)
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
# WORKER 2: APPROX MCI (New, Optimized)
# ==============================================================================

def _aprox_mci_worker(args):
    """
    Worker function for aproxmci.
    Combines the efficient matrix multiplication of _mci_worker with
    the distance and parity filtering of HamiltonMCI.

    args: (matrices, dist_mat, second_idx, mcialg, d, prune_reversed)
    """
    matrices, dist_mat, second_idx, mcialg, d, prune_reversed = args
    n = len(matrices)

    # --- 1. Filter Initial Step (0 -> second_idx) ---
    dist_0_sec = dist_mat[0, second_idx]

    # Global Constraint: All algs (1-4) require edges <= d
    if dist_0_sec > d: return 0.0, 0

    # Alg 3 (Odd) / Alg 4 (Even) Checks
    if mcialg == 3 and (dist_0_sec % 2 == 0): return 0.0, 0
    if mcialg == 4 and (dist_0_sec % 2 != 0): return 0.0, 0

    # For Alg 2, we track max distance found in path
    current_max = dist_0_sec

    # --- 2. Initialize DFS ---
    current_prod = np.dot(matrices[0], matrices[second_idx])
    visited_mask = (1 << 0) | (1 << second_idx)

    # Stack: (depth, visited_mask, current_prod, last_node_idx, current_max_dist)
    stack = [(2, visited_mask, current_prod, second_idx, current_max)]

    local_trace_sum = 0.0
    local_count = 0

    while stack:
        depth, mask, prod, last_idx, cur_max = stack.pop()

        # --- BASE CASE: One node remaining (Leaf) ---
        if depth == n - 1:
            # Find remaining node
            rem_node = -1
            for i in range(n):
                if not (mask & (1 << i)):
                    rem_node = i
                    break

            # A. Pruning (Symmetric Partition)
            if prune_reversed and (second_idx > rem_node):
                continue

            # B. Validate Last Step (last -> rem)
            dist_last = dist_mat[last_idx, rem_node]
            if dist_last > d: continue
            if mcialg == 3 and (dist_last % 2 == 0): continue
            if mcialg == 4 and (dist_last % 2 != 0): continue

            # C. Validate Closing Step (rem -> 0)
            dist_close = dist_mat[rem_node, 0]
            if dist_close > d: continue
            if mcialg == 3 and (dist_close % 2 == 0): continue
            if mcialg == 4 and (dist_close % 2 != 0): continue

            # D. Alg 2 Special Check: Max dist in cycle must be EXACTLY d
            if mcialg == 2:
                final_max = max(cur_max, dist_last, dist_close)
                if final_max != d: continue

            # E. Compute Trace (Trace Trick)
            term = np.sum(prod * matrices[rem_node].T)
            local_trace_sum += term
            local_count += 1
            continue

        # --- RECURSIVE STEP ---
        for i in range(1, n):
            if not (mask & (1 << i)):
                dist_step = dist_mat[last_idx, i]

                # Filter based on Distance / Parity
                if dist_step > d: continue
                if mcialg == 3 and (dist_step % 2 == 0): continue
                if mcialg == 4 and (dist_step % 2 != 0): continue

                # Update Max Distance (only needed for Alg 2)
                new_max = max(cur_max, dist_step) if mcialg == 2 else 0

                # Optimization for Alg 2: If we already exceeded d, stop (impossible, but good safety)
                # Actually, check above 'dist_step > d' handles this.

                new_prod = np.dot(prod, matrices[i])
                new_mask = mask | (1 << i)
                stack.append((depth + 1, new_mask, new_prod, i, new_max))

    return local_trace_sum, local_count


# ==============================================================================
# MAIN FUNCTION 1: EXACT MCI (Original, Unchanged)
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


# ==============================================================================
# MAIN FUNCTION 2: APPROX MCI (Optimized Integration)
# ==============================================================================

def aproxmci(arr, aom, partition=None, mcialg=0, d=1, ncores=1, minlen=6, maxlen=6, rings_thres=0.3, connec=None):
    start = time()

    # ---------------------------------------------------------
    # 0. Legacy Handling (Full MCI if mcialg=0)
    # ---------------------------------------------------------
    if mcialg == 0:
        if ncores == 1:
            val = sequential_mci(arr, aom, partition)
        else:
            val = multiprocessing_mci(arr, aom, ncores, partition)

        is_mulliken = (partition == "mulliken" or partition == "non-symmetric")
        if is_mulliken:
            nperms = factorial(len(arr) - 1)
        else:
            nperms = factorial(len(arr) - 1) * 0.5

        t = time() - start
        return val, nperms, t

    # ---------------------------------------------------------
    # 1. Connectivity & Distance Setup
    # ---------------------------------------------------------
    # Build full connectivity map
    print(connec)
    if not connec:
        connec = build_connectivity(aom, rings_thres)

    print(connec)
    connec = filter_connec(connec)
    print(connec)
    exit()
    full_distances = find_node_distances(connec)

    # Extract only the subgraph distances relevant to 'arr'
    # arr contains indices in the AOM (1-based usually in this context, so shift if needed)
    # Assuming arr indices match aom indices/connec indices.
    n = len(arr)
    sub_dist_matrix = np.zeros((n, n), dtype=int)
    # Map arr[i] to arr[j] distance
    for i in range(n):
        for j in range(n):
            sub_dist_matrix[i, j] = full_distances[arr[i]][arr[j]]

    # ---------------------------------------------------------
    # 2. Input Normalization (Matrices)
    # ---------------------------------------------------------
    # Pre-multiply Occ @ AOM if tuple (Natural Orbitals)
    if isinstance(aom, (tuple, list)) and len(aom) == 2 and not isinstance(aom[0], np.ndarray):
        real_aoms, occ = aom
        matrices = [np.dot(occ, real_aoms[idx - 1]) for idx in arr]
    else:
        matrices = [aom[idx - 1] for idx in arr]

    if n < 3:
        return 0.0, 0, time() - start

    # ---------------------------------------------------------
    # 3. Setup Tasks
    # ---------------------------------------------------------
    # Determine Pruning based on partition
    # If Mulliken/Non-sym -> We do NOT prune reversed paths in the walker.
    # (We divide by 2 at the end instead).
    is_mulliken = (partition == "mulliken" or partition == "non-symmetric")
    prune_reversed = not is_mulliken

    tasks = [
        (matrices, sub_dist_matrix, sec, mcialg, d, prune_reversed)
        for sec in range(1, n)
    ]

    total_trace = 0.0
    total_perms = 0

    # ---------------------------------------------------------
    # 4. Execution
    # ---------------------------------------------------------
    # Use multiprocessing if worthwhile
    if ncores > 1 and len(tasks) > 1:
        with mp.Pool(processes=ncores) as pool:
            results = pool.map(_aprox_mci_worker, tasks, chunksize=1)

        # Sum up results
        for t_val, p_count in results:
            total_trace += t_val
            total_perms += p_count
    else:
        for task in tasks:
            t_val, p_count = _aprox_mci_worker(task)
            total_trace += t_val
            total_perms += p_count

    # ---------------------------------------------------------
    # 5. Prefactor & Return
    # ---------------------------------------------------------
    # Normalization logic
    if wf_type(aom) in ["rest", "unrest"]:
        prefactor = 2 ** (n - 1)
    else:
        prefactor = 0.5

    # If Mulliken, we calculated full sum (forward + reverse), so we halve it.
    if is_mulliken:
        total_trace *= 0.5

    final_val = prefactor * total_trace

    return final_val, total_perms, time() - start
