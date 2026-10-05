import numpy as np
import multiprocessing as mp
from math import factorial
from time import time
from collections import defaultdict, deque

from esipy.tools import (
    wf_type, mapping, filter_connec, build_connectivity, find_di, find_di_no
)


def find_node_distances(connec):
    """Standard BFS to find the shortest path between nodes."""
    distances = defaultdict(dict)

    for start in connec:
        queue = deque([(start, 0)])
        visited = set()

        while queue:
            cur_node, cur_dist = queue.popleft()

            if cur_node not in visited:
                visited.add(cur_node)
                distances[start][cur_node] = cur_dist

                for neighbor in connec[cur_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, cur_dist + 1))

    return distances


def find_node_distances_onlyodd(connec):
    """BFS tracking parity. Prefers the shortest odd distance, falls back to even."""
    distances = defaultdict(dict)

    for start in connec:
        queue = deque([(start, 0)])
        visited = {(start, 0)}  # Track (node, parity)

        # Temporary storage to track the best of both parities
        shortest_odd = {}
        shortest_even = {}

        while queue:
            cur_node, cur_dist = queue.popleft()

            # Save the shortest distance found for each parity
            if cur_dist % 2 == 1:
                if cur_node not in shortest_odd:
                    shortest_odd[cur_node] = cur_dist
            else:
                if cur_node not in shortest_even:
                    shortest_even[cur_node] = cur_dist

            # Explore neighbors
            for neighbor in connec.get(cur_node, []):
                next_dist = cur_dist + 1
                next_parity = next_dist % 2
                state = (neighbor, next_parity)

                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, next_dist))

        # Decision Time: Merge into final distances
        # Get every unique node we reached during this search
        all_reached_nodes = set(shortest_odd.keys()).union(shortest_even.keys())

        for node in all_reached_nodes:
            if node in shortest_odd:
                distances[start][node] = shortest_odd[node]  # Prefer Odd
            else:
                distances[start][node] = shortest_even[node]  # Fallback to Even

        distances[start][start] = 0  # Starting node is always 0

    return distances
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

            # Tr(A @ B) = sum(A * B.T) (just for the final step, O(N**3) -> O(N**2))
            tr_sum += np.sum(P * mats[rem].T)
            continue

        # Recurse like James Harden, step back
        for i in range(1, n):
            if not ((mask >> i) & 1):
                stack.append((depth + 1, mask | (1 << i), np.dot(P, mats[i])))

    return tr_sum


def _kernel_approx(args):
    """
    DFS worker with distance/parity filtering and lazy prefix matrix multiplication.
    args: (mats, dists, prefix, cur_max, alg_type, d_cut, sym_prune)
      or: (mats, dists, j, alg_type, d_cut, sym_prune)
    """
    mats, dists = args[0], args[1]
    n = len(mats)

    if len(args) == 7:
        prefix, cur_max, alg, d_cut, sym_prune = args[2], args[3], args[4], args[5], args[6]
    else:
        j, alg, d_cut, sym_prune = args[2], args[3], args[4], args[5]
        d0 = dists[0, j]
        if d0 > d_cut: return 0.0, 0
        if (alg == 3 or alg == 5) and (d0 % 2 == 0): return 0.0, 0
        prefix = (0, j)
        cur_max = d0

    # Precompute valid neighbors according to algorithm distance and parity constraints
    valid_neighbors = [
        [v for v in range(n) if v != u and dists[u, v] <= d_cut and ((alg not in (3, 5)) or (dists[u, v] % 2 != 0))]
        for u in range(n)
    ]

    visited = [False] * n
    path = [0] * n
    for idx, node in enumerate(prefix):
        visited[node] = True
        path[idx] = node

    j = prefix[1]
    depth_start = len(prefix)

    # Lazy prefix matrix product cache: P_cache[k] = mats[0] @ mats[path[1]] @ ... @ mats[path[k]]
    P_cache = [None] * n
    P_cache[0] = mats[0]
    valid_depth = [1]

    tr_sum = [0.0]
    n_perms = [0]

    def dfs(prev, depth, cur_max):
        if depth == n - 1:
            rem = visited.index(False)
            if sym_prune and (j > rem): return

            # Validate closing edges (prev->rem, rem->0)
            d_last = dists[prev, rem]
            d_close = dists[rem, 0]
            if d_last > d_cut or d_close > d_cut: return
            if (alg == 3 or alg == 5) and ((d_last % 2 == 0) or (d_close % 2 == 0)): return
            if alg == 2 or alg == 4:
                if max(cur_max, d_last, d_close) != d_cut: return

            # Lazy update of prefix product cache up to depth - 1
            vd = valid_depth[0]
            for k in range(vd, n - 1):
                P_cache[k] = np.dot(P_cache[k - 1], mats[path[k]])
            valid_depth[0] = n - 1

            # Tr(A @ B) = sum(A * B.T) in O(K^2)
            tr_sum[0] += np.sum(P_cache[n - 2] * mats[rem].T)
            n_perms[0] += 1
            return

        for v in valid_neighbors[prev]:
            if not visited[v]:
                visited[v] = True
                path[depth] = v
                if valid_depth[0] > depth:
                    valid_depth[0] = depth
                new_max = max(cur_max, dists[prev, v]) if (alg == 2 or alg == 4) else 0
                dfs(v, depth + 1, new_max)
                visited[v] = False

    dfs(prefix[-1], depth_start, cur_max)
    return tr_sum[0], n_perms[0]


def _kernel_active(args):
    """
    DFS worker with active graph filtering.
    args: (mats, active_adj, j, sym_prune)
    """
    mats, active_adj, j, sym_prune = args
    n = len(mats)

    # If first step 0 -> j is not active, return 0
    if j not in active_adj[0]:
        return 0.0, 0

    P = np.dot(mats[0], mats[j])
    mask = (1 << 0) | (1 << j)

    # Stack: (depth, mask, product, prev_node)
    stack = [(2, mask, P, j)]

    tr_sum = 0.0
    n_perms = 0

    while stack:
        depth, mask, P, prev = stack.pop()

        # Leaf node
        if depth == n - 1:
            rem = 0
            while (mask >> rem) & 1:
                rem += 1

            if sym_prune and (j > rem):
                continue

            # Check if last steps are in the active graph
            if rem not in active_adj[prev] or 0 not in active_adj[rem]:
                continue

            # Tr(A @ B) = sum(A * B.T)
            tr_sum += np.sum(P * mats[rem].T)
            n_perms += 1
            continue

        # Recurse: only visit active neighbors of prev
        for neighbor in active_adj[prev]:
            if neighbor == 0:
                continue
            if not ((mask >> neighbor) & 1):
                stack.append((depth + 1, mask | (1 << neighbor), np.dot(P, mats[neighbor]), neighbor))

    return tr_sum, n_perms


def _prep_matrices(arr, aom):
    """Normalize input into list of matrices, handling NO tuples."""
    if isinstance(aom, (tuple, list)) and len(aom) == 2 and not isinstance(aom[0], np.ndarray):
        # For Natural Orbitals: Pre-multiply Occ @ AOM
        real_aoms, occ = aom
        return [occ[:, None] * real_aoms[idx - 1] for idx in arr]
    return [aom[idx - 1] for idx in arr]


def compute_mci(arr, aom, partition='mulliken', n_cores=None):
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

    alg (int): 0=Exact, 1=Dist, 2=MaxDist, 3=Odd, 4=MaxDistOdd, 5=Odd(Alternate)
    """
    t0 = time()

    # Exact if alg=0
    if alg == 0:
        val = compute_mci(arr, aom, partition, n_cores)
        # Factorial scaling for display purposes
        nperms = factorial(len(arr) - 1)
        if partition != "mulliken" and partition != "non-symmetric":
            nperms *= 0.5
        return val, nperms, time() - t0

    if alg == 6:
        delta = kwargs.get('delta', 1e-4)
        wf = wf_type(aom)
        n = len(arr)
        
        # Build active connectivity graph
        active_adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if wf == "no":
                    di_val = find_di_no(aom, arr[i], arr[j])
                else:
                    di_val = find_di(aom, arr[i], arr[j])
                if abs(di_val) > delta:
                    active_adj[i].append(j)
                    
        mats = _prep_matrices(arr, aom)
        if n < 3: return 0.0, 0, time() - t0
        
        is_mulliken = (partition in ["mulliken", "non-symmetric"])
        sym_prune = not is_mulliken
        
        tasks = [(mats, active_adj, j, sym_prune) for j in range(1, n)]
        
        total_trace = 0.0
        total_perms = 0
        
        if n_cores > 1 and len(tasks) > 1:
            with mp.Pool(n_cores) as pool:
                results = pool.map(_kernel_active, tasks, chunksize=1)
            for t, p in results:
                total_trace += t
                total_perms += p
        else:
            for t in tasks:
                t_val, p_count = _kernel_active(t)
                total_trace += t_val
                total_perms += p_count
                
        is_sd = wf_type(aom) in ["rest", "unrest"]
        prefactor = 2 ** (n - 1) if is_sd else 0.5
        
        if is_mulliken:
            total_trace *= 0.5
            
        return prefactor * total_trace, total_perms, time() - t0

    if connec is None:
        connec = build_connectivity(aom, rings_thres)

    # Subgraph for the ring atoms only (preserving all atoms in arr)
    ring_connec = {
        atom: [neighbor for neighbor in connec.get(atom, []) if neighbor in arr]
        for atom in arr
    }
    # ==========================================================

    # Use onlyodd for alg 4 and 5, but pass the isolated ring_connec!
    if alg == 4 or alg == 5:
        full_dists = find_node_distances_onlyodd(ring_connec)
    else:
        full_dists = find_node_distances(ring_connec)

    n = len(arr)
    sub_dists = np.array([[full_dists[arr[i]].get(arr[j], 999999) for j in range(n)] for i in range(n)])

    mats = _prep_matrices(arr, aom)
    if n < 3: return 0.0, 0, time() - t0

    # 2. Prepare Tasks
    is_mulliken = (partition in ["mulliken", "non-symmetric"])
    sym_prune = not is_mulliken

    # Filter initial valid starts (0 -> v)
    valid_starts = []
    for v in range(1, n):
        d0 = sub_dists[0, v]
        if d0 <= d and ((alg not in (3, 5)) or (d0 % 2 != 0)):
            valid_starts.append(((0, v), d0))

    # If multiprocessing and more cores than valid starts, expand prefixes to saturate all cores
    queue = deque(valid_starts)
    target_tasks = max(len(queue), n_cores * 2) if n_cores > 1 else len(queue)
    while queue and len(queue) < target_tasks:
        if len(queue[0][0]) >= n - 1:
            break
        level_size = len(queue)
        for _ in range(level_size):
            path, cur_max = queue.popleft()
            u = path[-1]
            for v in range(n):
                if v not in path and sub_dists[u, v] <= d:
                    if (alg == 3 or alg == 5) and (sub_dists[u, v] % 2 == 0):
                        continue
                    new_max = max(cur_max, sub_dists[u, v]) if (alg in (2, 4)) else 0
                    queue.append((path + (v,), new_max))
            if len(queue) >= target_tasks:
                break

    tasks = [(mats, sub_dists, path, cur_max, alg, d, sym_prune) for path, cur_max in queue]

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
    return compute_mci(arr, aom, partition, n_cores=1)


def multiprocessing_mci(arr, aom, ncores, partition='mulliken'):
    return compute_mci(arr, aom, partition, n_cores=ncores)


# Aliases for Natural Orbitals
sequential_mci_no = sequential_mci
multiprocessing_mci_no = multiprocessing_mci
aproxmci = mci_approx
