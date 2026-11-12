import numpy as np
import multiprocessing as mp
from math import factorial, comb, fsum
from itertools import combinations
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


_MATS, _LAYER = None, None


def _init_worker(mats, layer):
    global _MATS, _LAYER
    _MATS, _LAYER = mats, layer


def _max_layer(nrest, m, max_mem):
    """Largest subset size whose two consecutive layers fit in max_mem (MB)."""
    k = 0
    for j in range(1, nrest + 1):
        if (comb(nrest, j) + comb(nrest, j - 1)) * m * m * 8 / 1e6 > max_mem:
            break
        k = j
    return k


def _build_layer(mats, k):
    """M_R = sum over orderings of R of the product of its matrices, for every subset R of size k."""
    rest = range(1, len(mats))
    layer = {0: np.identity(mats[0].shape[0])}
    for j in range(1, k + 1):
        layer = {sum(1 << r for r in R): sum(np.dot(mats[r], layer[sum(1 << x for x in R if x != r)]) for r in R)
                 for R in combinations(rest, j)}
    return layer


def _kernel_layer(args):
    """DFS over the prefixes 0 -> j -> ... that close with a stored subset sum."""
    j, k = args
    mats, layer = _MATS, _LAYER
    n = len(mats)
    full = (1 << n) - 2
    stack = [(1 << j, np.dot(mats[0], mats[j]))]
    tr_sum = 0.0
    while stack:
        mask, P = stack.pop()
        if bin(mask).count("1") == n - 1 - k:
            tr_sum += np.sum(P * layer[full ^ mask].T)
            continue
        for i in range(1, n):
            if not (mask >> i) & 1:
                stack.append((mask | (1 << i), np.dot(P, mats[i])))
    return tr_sum


def _prep_matrices(arr, aom):
    """Normalize input into list of matrices, handling NO tuples."""
    if isinstance(aom, (tuple, list)) and len(aom) == 2 and not isinstance(aom[0], np.ndarray):
        # For Natural Orbitals: occ^(1/4) AOM occ^(1/4), same traces as occ^(1/2) AOM and symmetric
        real_aoms, occ = aom
        o = np.sqrt(np.sqrt(np.clip(occ, 0, None)))  # tiny negative occupations are zero
        return [o[:, None] * real_aoms[idx - 1] * o[None, :] for idx in arr]
    return [aom[idx - 1] for idx in arr]


def _mci_subsets(mats, n_cores, max_mem):
    """Sum of Tr over all orderings with atom 0 fixed, using subset sums of the last atoms."""
    n = len(mats)

    # Sums over all orderings of the last k atoms are built once and reused
    k = min(_max_layer(n - 1, mats[0].shape[0], max_mem), n - 2)
    layer = _build_layer(mats, k)
    tasks = [(j, k) for j in range(1, n)]

    if k < n - 2 and n_cores > 1:
        with mp.Pool(n_cores, initializer=_init_worker, initargs=(mats, layer)) as pool:
            total_trace = sum(pool.map(_kernel_layer, tasks, chunksize=1))
    else:
        _init_worker(mats, layer)
        total_trace = sum(_kernel_layer(t) for t in tasks)

    return total_trace


def compute_mci_dfs(arr, aom, partition='mulliken', n_cores=None, max_mem=None):
    """
    Legacy: computes Exact MCI using DFS with lazy product cache and multi-core dispatch.

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
    max_mem : float, optional
        If given (MB), use subset sums instead of the DFS. Much faster, memory grows as 2^n.
    """
    mats = _prep_matrices(arr, aom)
    n = len(arr)
    if n < 3: return 0.0

    if max_mem is not None:
        is_sd = wf_type(aom) in ["rest", "unrest"]
        return (2 ** (n - 2) if is_sd else 2 ** (n / 2 - 1)) * _mci_subsets(mats, n_cores or mp.cpu_count(), max_mem)

    is_mulliken = (partition in ["mulliken", "non-symmetric"])
    sym_prune = not is_mulliken

    if n_cores is None:
        n_cores = mp.cpu_count()

    sub_dists = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
    valid_starts = [((0, v), 0) for v in range(1, n)]

    queue = deque(valid_starts)
    target_tasks = max(len(queue), n_cores * 2) if n_cores > 1 else len(queue)
    while queue and len(queue) < target_tasks:
        if len(queue[0][0]) >= n - 1:
            break
        level_size = len(queue)
        for _ in range(level_size):
            path, _ = queue.popleft()
            for v in range(n):
                if v not in path:
                    queue.append((path + (v,), 0))
            if len(queue) >= target_tasks:
                break

    tasks = [(mats, sub_dists, path, 0, 1, 999999, sym_prune) for path, _ in queue]

    if n_cores > 1 and len(tasks) > 1:
        with mp.Pool(n_cores) as pool:
            results = pool.map(_kernel_approx, tasks, chunksize=1)
        total_trace = sum(t for t, _ in results)
    else:
        total_trace = sum(_kernel_approx(t)[0] for t in tasks)

    is_sd = wf_type(aom) in ["rest", "unrest"]
    prefactor = 2 ** (n - 1) if is_sd else 2 ** (n / 2)
    if is_mulliken:
        total_trace *= 0.5

    return prefactor * total_trace


def mci_approx_dfs(arr, aom, partition=None, alg=0, d=1, n_cores=1,
                   rings_thres=0.3, connec=None, **kwargs):
    """
    Legacy: computes Approximate MCI with topological filtering.

    alg (int): 0=Exact, 1=Dist, 2=MaxDist, 3=Odd, 4=MaxDistOdd, 5=Odd(Alternate)
    """
    t0 = time()

    # Exact if alg=0
    if alg == 0:
        val = compute_mci_dfs(arr, aom, partition, n_cores, max_mem=kwargs.get('max_mem'))
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
        prefactor = 2 ** (n - 1) if is_sd else 2 ** (n / 2)
        
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
    prefactor = 2 ** (n - 1) if is_sd else 2 ** (n / 2)

    if is_mulliken:
        total_trace *= 0.5

    return prefactor * total_trace, total_perms, time() - t0


# ---------------------------------------------------------------------------
# MCI with Glynn's identity. With S_0 fixed in front and X = sum_i delta_i S_i (delta_i = +-1, delta_1 = +1):
#   sum over the orderings of S_1..S_m of Tr(S_0 S_s1 ... S_sm) = 2^-(m-1) sum_delta prod(delta) Tr(S_0 X^m)
# X^m expands into every word of length m and the signs cancel the words that repeat an atom, so only the
# permutations remain. The approximations restrict each step u -> v to an adjacency matrix A (the same BFS
# distances as the DFS); the same signs over the closed walks 0 -> v_1 -> ... -> v_m -> 0 allowed by A keep only
# the allowed permutations. 2^(n-2) independent terms: constant memory, split over processes.
# ---------------------------------------------------------------------------

_PRIMES = (2147483647, 2147483629, 2147483587, 2147483579)


def _glynn_deltas(s0, s1, m):
    """Signs delta (delta_1 = +1, the others from the bits of s) for the terms s0 <= s < s1, and prod(delta)."""
    bits = (np.arange(s0, s1)[:, None] >> np.arange(m - 1)) & 1
    delta = np.ones((s1 - s0, m))
    delta[:, 1:] -= 2 * bits
    return delta, delta.prod(axis=1)


def _factors(mats, sym):
    """
    S = U diag(w) V^T for each matrix with its numerical rank (~ basis functions of the atom); eigenvectors if
    symmetric (V = U). The matrices after the first are zero-padded to a common rank.
    """
    fs = []
    for M in mats:
        if sym:
            w, U = np.linalg.eigh(M)
            V = U
        else:
            U, w, Vt = np.linalg.svd(M)
            V = Vt.T
        keep = np.abs(w) > np.abs(w).max() * len(w) * np.finfo(float).eps
        keep[np.argmax(np.abs(w))] = True
        fs.append((U[:, keep], w[keep], V[:, keep]))
    r = max(len(w) for _, w, _ in fs[1:])
    U, V, W = (np.zeros((len(mats) - 1, len(mats[0]), r)), np.zeros((len(mats) - 1, len(mats[0]), r)),
               np.zeros((len(mats) - 1, r)))
    for i, (u, w, v) in enumerate(fs[1:]):
        U[i, :, :len(w)], V[i, :, :len(w)], W[i, :len(w)] = u, v, w
    return fs[0], U, W, V


def _count_primes(m):
    """Primes whose product exceeds m!, to count the permutations exactly with modular arithmetic."""
    primes, prod = [], 1
    for p in _PRIMES:
        primes.append(p)
        prod *= p
        if prod > factorial(m):
            break
    return primes


def _glynn_counts(delta, sign, adj, counts):
    """Adds the number of permutations of the batch (the walk recursion with all S = 1) modulo each prime."""
    m = delta.shape[1]
    step, first, last = adj[1:, 1:].astype(np.int64), adj[0, 1:].astype(np.int64), adj[1:, 0].astype(np.int64)
    di, si = delta.astype(np.int64), sign.astype(np.int64)
    for i, p in enumerate(_count_primes(m)):
        w = di * first
        for _ in range(m - 1):
            w = (np.dot(w, step) * di) % p
        counts[i] = (counts[i] + int(np.dot(si, np.dot(w, last) % p))) % p
    return counts


def _kernel_glynn_exact(args):
    """
    Terms s0 <= s < s1 of the exact MCI. With S_0 = U diag(w) V^T, Tr(S_0 X^m) = sum_a w_a v_a^T X^m u_a needs
    products of X with a thin block: m/2 of them for symmetric AOMs, u^T X^m u = (X^h u).(X^(m-h) u), else m/2 with X^2.
    """
    mats, _, s0, s1, batch, sym = args
    rest = np.array(mats[1:])
    m, k = rest.shape[0], rest.shape[1]
    (U0, w0, V0), _, _, _ = _factors(mats[:2], sym)
    parts = []
    for a in range(s0, s1, batch):
        delta, sign = _glynn_deltas(a, min(a + batch, s1), m)
        X = np.dot(delta, rest.reshape(m, -1)).reshape(-1, k, k)
        P = X if sym else np.matmul(X, X)
        Y = np.matmul(P, U0)
        for _ in range(m // 2 - 1):
            Y = np.matmul(P, Y)
        Z = np.matmul(X, Y) if m % 2 else Y
        tr = np.einsum('bka,bka,a->b', Y, Z, w0) if sym else np.einsum('ka,bka,a->b', V0, Z, w0)
        parts.append(fsum(sign * tr))
    return fsum(parts), None


def _kernel_glynn_walks(args):
    """
    Terms s0 <= s < s1 over the closed walks allowed by adj, and their number. The walks 0 -> ... -> v are kept as
    r0 x r blocks (S_0 ... S_v = U_0 B_v V_v^T) and one step multiplies by G_uv = adj[u, v] V_u^T U_v. Only the atoms
    that can still close a cycle are kept at each step; for symmetric AOMs two half walks are joined (half the steps).
    """
    mats, adj, s0, s1, batch, sym = args
    m = len(mats) - 1
    (U0, w0, V0), U, W, V = _factors(mats, sym)
    r0, r = len(w0), W.shape[1]
    G = (adj[1:, 1:, None, None] * np.einsum('uka,vkb->uvab', V, U)).transpose(0, 2, 1, 3).reshape(m * r, m * r)
    B1 = (adj[0, 1:, None, None] * np.matmul(V0.T, U)).transpose(1, 0, 2).reshape(r0, m * r)
    end = (adj[1:, 0, None, None] * np.matmul(V.transpose(0, 2, 1), U0)).transpose(2, 0, 1).reshape(r0, m * r)

    # Atoms that can be the t-th of a closed walk of m atoms through 0, as columns of the blocks
    step = adj[1:, 1:].astype(float)
    fwd, back = [adj[0, 1:] > 0], [adj[1:, 0] > 0]
    for _ in range(m - 1):
        fwd.append(fwd[-1] @ step > 0)
        back.append(step @ back[-1] > 0)
    cols = [np.flatnonzero(np.repeat(fwd[t] & back[m - 1 - t], r)) for t in range(m)]
    if any(len(c) == 0 for c in cols):
        return 0.0, [0] * len(_count_primes(m))
    a = (m + 2) // 2 if sym else m
    b = m + 1 - a
    Gs = [G[np.ix_(cols[t], cols[t + 1])] for t in range(a - 1)]

    parts, counts = [], [0] * len(_count_primes(m))
    for i in range(s0, s1, batch):
        delta, sign = _glynn_deltas(i, min(i + batch, s1), m)
        D = np.repeat(delta, r, axis=1) * W.ravel()
        Y = half = np.broadcast_to(B1[:, cols[0]], (len(sign), r0, len(cols[0])))
        for t in range(a - 1):
            Y = ((Y * D[:, None, cols[t]]).reshape(-1, len(cols[t])) @ Gs[t]).reshape(len(sign), r0, -1)
            if t + 2 == b:
                half = Y
        if sym:
            tr = np.einsum('bic,bc,bic,i->b', Y, D[:, cols[a - 1]], half, w0)
        else:
            tr = np.einsum('bic,bc,ic,i->b', Y, D[:, cols[a - 1]], end[:, cols[a - 1]], w0)
        parts.append(fsum(sign * tr))
        counts = _glynn_counts(delta, sign, adj, counts)
    return fsum(parts), counts


def _project(mats):
    """Restricts the matrices to the space spanned by their columns; the traces of their products are unchanged."""
    U, s, _ = np.linalg.svd(np.hstack(mats), full_matrices=False)
    Q = U[:, :max(1, int((s > s[0] * len(s) * np.finfo(float).eps).sum()))]
    return [Q.T @ M @ Q for M in mats] if Q.shape[1] < len(Q) else mats


def _two_sides(adj):
    """
    Order (atom 0, the other side, its own side) when every allowed step joins two sides of equal size, else None.
    Every closed walk then visits each side n/2 times, so one sign per side can be fixed: half the terms.
    """
    n = len(adj)
    side = -np.ones(n, dtype=int)
    side[0], stack = 0, [0]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            if side[v] < 0:
                side[v] = 1 - side[u]
                stack.append(v)
            elif side[v] == side[u]:
                return None
    if n % 2 or n < 4 or (side < 0).any() or 2 * (side == 1).sum() != n:
        return None
    return [0] + [v for v in range(1, n) if side[v] == 1] + [v for v in range(1, n) if side[v] == 0]


def _glynn_sum(mats, adj, n_cores, max_mem):
    """
    Sum over the permutations allowed by adj (None for all of them) and their number, with Glynn's identity.
    The 2^(n-2) terms go in batches of max_mem MB over n_cores processes.
    """
    mats = _project(mats)
    m, k = len(mats) - 1, mats[0].shape[0]
    sym = all(np.abs(M - M.T).max() <= 1e-12 * np.abs(M).max() for M in mats) and (adj is None or (adj == adj.T).all())
    if sym:
        mats = [(M + M.T) / 2 for M in mats]

    # The atom of lowest rank goes first; a second sign is fixed for two equal sides
    ranks = [np.linalg.matrix_rank(M) for M in mats]
    order = [int(np.argmin(ranks))] + [i for i in range(m + 1) if i != np.argmin(ranks)]
    nfix = 1
    if adj is not None:
        adj = adj[np.ix_(order, order)]
        sides = _two_sides(adj) if sym else None
        if sides is not None:
            order, adj, nfix = [order[i] for i in sides], adj[np.ix_(sides, sides)], 2
    mats = [mats[i] for i in order]
    nterms = 1 << (m - nfix)

    kernel = _kernel_glynn_exact if adj is None else _kernel_glynn_walks
    term_mem = 4 * k * k if adj is None else 4 * ranks[order[0]] * m * max(ranks)
    batch = int(min(nterms, max(1, max_mem * 1e6 // (8 * term_mem))))
    if n_cores > 1 and nterms > batch:
        bounds = np.linspace(0, nterms, min(4 * n_cores, -(-nterms // batch)) + 1).astype(int)
        with mp.Pool(n_cores) as pool:
            results = pool.map(kernel, [(mats, adj, a, b, batch, sym) for a, b in zip(bounds[:-1], bounds[1:])],
                               chunksize=1)
    else:
        results = [kernel((mats, adj, 0, nterms, batch, sym))]

    total = fsum(t for t, _ in results) / nterms
    if adj is None:
        return total, factorial(m)
    # Number of permutations from its residues (each one is nterms times the count)
    count, mod = 0, 1
    for i, p in enumerate(_count_primes(m)):
        res = sum(c[i] for _, c in results) * pow(2, -(m - nfix), p) % p
        count += mod * ((res - count) * pow(mod, -1, p) % p)
        mod *= p
    return total, count


def _cycle_search(mats, adj):
    """
    Sum over the cycles of a sparse symmetric adjacency (few allowed steps, e.g. d = 1 on a fullerene) and their
    number. Glynn's 2^(n-2) terms do not shrink with the adjacency, so here the cycles are searched directly,
    dropping a path as soon as an unused atom cannot be entered and left any more.
    """
    mats = _project(mats)
    n = len(mats)
    _, U, W, V = _factors([mats[0]] + list(mats), False)
    P = np.einsum('uka,vkb->uvab', V, U) * W[None, :, None, :]
    nb = [[int(v) for v in np.flatnonzero(adj[u])] for u in range(n)]
    visited, avail = [False] * n, [len(x) for x in nb]
    visited[0] = True
    total, count = [0.0], [0]

    def search(end, depth, Y):
        if depth == n:
            if adj[end, 0]:
                total[0] += np.sum(Y * P[end, 0].T)
                count[0] += 1
            return
        for v in nb[end]:
            if visited[v]:
                continue
            # end becomes interior: its other unused neighbours lose one way in or out (atom 0 stays open)
            touched = [w for w in nb[end] if not visited[w] and w != v] if end else []
            ok = True
            for w in touched:
                avail[w] -= 1
                ok = ok and avail[w] >= 2
            if ok and depth + 1 < n and v not in nb[0] and not any(not visited[w] and w != v for w in nb[0]):
                ok = False  # atom 0 needs a neighbour left to close the cycle
            if ok:
                visited[v] = True
                search(v, depth + 1, P[end, v] if Y is None else Y @ P[end, v])
                visited[v] = False
            for w in touched:
                avail[w] += 1

    search(0, 1, None)
    return total[0], count[0]


def _allowed_steps(arr, aom, alg, d, rings_thres, connec, delta):
    """Adjacency matrices of the allowed steps, with their sign (mcialg 2 and 4: all <= d minus all <= d - 1)."""
    n = len(arr)
    if alg == 6:
        di = find_di_no if wf_type(aom) == "no" else find_di
        return [(1, np.array([[i != j and abs(di(aom, arr[i], arr[j])) > delta for j in range(n)] for i in range(n)]))]
    if connec is None:
        connec = build_connectivity(aom, rings_thres)
    ring_connec = {atom: [nb for nb in connec.get(atom, []) if nb in arr] for atom in arr}
    full_dists = find_node_distances_onlyodd(ring_connec) if alg in (4, 5) else find_node_distances(ring_connec)
    dists = np.array([[full_dists[arr[i]].get(arr[j], 999999) for j in range(n)] for i in range(n)])
    np.fill_diagonal(dists, 999999)
    if alg in (3, 5):
        return [(1, (dists <= d) & (dists % 2 == 1))]
    if alg in (2, 4):
        return [(1, dists <= d), (-1, dists <= d - 1)]
    return [(1, dists <= d)]


def _restricted_sum(mats, adj, n_cores, max_mem):
    """Sum and number of the permutations allowed by adj: cycle search if sparse, Glynn otherwise."""
    n = len(mats)
    if not (adj.any(axis=0).all() and adj.any(axis=1).all()):
        return 0.0, 0  # an atom without an allowed step in or out
    if adj.sum() <= 4 * n and (adj == adj.T).all():
        return _cycle_search(mats, adj)
    return _glynn_sum(mats, None if adj.sum() == n * (n - 1) else adj, n_cores, max_mem)


def compute_mci(arr, aom, partition='mulliken', n_cores=None, max_mem=None):
    """
    Computes the exact MCI with Glynn's identity (2^(n-2) terms, constant memory).

    Parameters
    ----------
    arr : list
        Atom indices involved in the ring.
    aom : np.ndarray or tuple
        Atomic Overlap Matrices. If tuple (AOMs, Occ), assumes Natural Orbitals.
    partition : str
        'mulliken', 'symmetric', etc.
    n_cores : int, optional
        Number of processes. Defaults to cpu_count.
    max_mem : float, optional
        Memory (MB) of each batch of terms per process. Default 64.
    """
    n = len(arr)
    if n < 3: return 0.0
    total_trace, _ = _glynn_sum(_prep_matrices(arr, aom), None, n_cores or mp.cpu_count(), max_mem or 64)

    # Glynn sums every ordering, as the DFS with Mulliken; for symmetric AOMs the reversed ones are equal
    is_sd = wf_type(aom) in ["rest", "unrest"]
    prefactor = 2 ** (n - 1) if is_sd else 2 ** (n / 2)
    return prefactor * 0.5 * total_trace


def mci_approx(arr, aom, partition=None, alg=0, d=1, n_cores=1,
               rings_thres=0.3, connec=None, **kwargs):
    """
    Computes Approximate MCI with topological filtering, with Glynn's identity on the allowed steps.

    alg (int): 0=Exact, 1=Dist, 2=MaxDist, 3=Odd, 4=MaxDistOdd, 5=Odd(Alternate), 6=Active (kwarg delta)
    Returns the MCI, the number of permutations and the time.
    """
    t0 = time()
    n = len(arr)
    is_mulliken = (partition in ["mulliken", "non-symmetric"])

    if alg == 0:
        val = compute_mci(arr, aom, partition, n_cores, max_mem=kwargs.get('max_mem'))
        nperms = factorial(n - 1)
        if not is_mulliken:
            nperms *= 0.5
        return val, nperms, time() - t0
    if n < 3: return 0.0, 0, time() - t0

    mats = _prep_matrices(arr, aom)
    total_trace, total_perms = 0.0, 0
    for sign, adj in _allowed_steps(arr, aom, alg, d, rings_thres, connec, kwargs.get('delta', 1e-4)):
        t, p = _restricted_sum(mats, adj, n_cores, kwargs.get('max_mem') or 64)
        total_trace += sign * t
        total_perms += sign * p

    # Each cycle is counted in both directions; for symmetric AOMs they are equal
    is_sd = wf_type(aom) in ["rest", "unrest"]
    prefactor = 2 ** (n - 1) if is_sd else 2 ** (n / 2)
    if not is_mulliken:
        total_perms //= 2
    return prefactor * 0.5 * total_trace, total_perms, time() - t0


def mci_profile(arr, aom, partition=None, alg=4, n_cores=1, rings_thres=0.3, connec=None, max_mem=64):
    """
    Contributions of mcialg 2 or 4 for every distance d at once (each cumulative sum, all steps <= d, is computed
    once); they add up to the exact MCI. Returns a list of (d, contribution, cumulative MCI, permutations, seconds).
    """
    n = len(arr)
    mats = _prep_matrices(arr, aom)
    if connec is None:
        connec = build_connectivity(aom, rings_thres)
    is_sd = wf_type(aom) in ["rest", "unrest"]
    prefactor = (2 ** (n - 1) if is_sd else 2 ** (n / 2)) * 0.5
    halve = 1 if partition in ["mulliken", "non-symmetric"] else 2
    profile, prev, prev_perms = [], 0.0, 0
    for d in range(1, 2 * n):  # odd-walk distances can exceed n - 1
        t0 = time()
        adj = _allowed_steps(arr, aom, alg, d, rings_thres, connec, None)[0][1]
        t, p = _restricted_sum(mats, adj, n_cores, max_mem)
        profile.append((d, prefactor * (t - prev), prefactor * t, (p - prev_perms) // halve, time() - t0))
        prev, prev_perms = t, p
        if adj.sum() == n * (n - 1):
            break
    return profile


def sequential_mci(arr, aom, partition='mulliken'):
    return compute_mci(arr, aom, partition, n_cores=1)


def multiprocessing_mci(arr, aom, ncores, partition='mulliken'):
    return compute_mci(arr, aom, partition, n_cores=ncores)


# Aliases for Natural Orbitals
sequential_mci_no = sequential_mci
multiprocessing_mci_no = multiprocessing_mci
aproxmci = mci_approx
