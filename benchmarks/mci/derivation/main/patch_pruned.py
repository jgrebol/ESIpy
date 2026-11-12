p="esipy/mci.py"
s=open(p).read()
new = '''def _pruned_sum(mats, adj):
    """
    Sum over the Hamiltonian cycles of a sparse symmetric adjacency, and their number, by a search that drops a
    path as soon as an unused atom cannot be entered and left any more. For the cases Glynn cannot reach (its
    2^(n-2) terms do not depend on the adjacency): few allowed steps and any number of atoms, like the
    distance-1 sum of a fullerene. Each step multiplies r x r blocks P_uv = V_u^T U_v w_v of the factored AOMs.
    """
    mats = _project(mats)
    n = len(mats)
    _, U, W, V = _factors([mats[0]] + list(mats), False)
    P = np.einsum('uka,vkb->uvab', V, U) * W[None, :, None, :]
    nb = [[int(v) for v in np.flatnonzero(adj[u])] for u in range(n)]
    visited, avail = [False] * n, [len(x) for x in nb]  # avail: neighbours still unused, the end or the start
    visited[0] = True
    total, count = [0.0], [0]

    def dfs(end, depth, Y):
        if depth == n:
            if adj[end, 0]:
                total[0] += np.sum(Y * P[end, 0].T)
                count[0] += 1
            return
        for v in nb[end]:
            if visited[v]:
                continue
            # end becomes interior: its other unused neighbours lose one option (the start stays open)
            touched = [w for w in nb[end] if not visited[w] and w != v] if end else []
            ok = True
            for w in touched:
                avail[w] -= 1
                ok = ok and avail[w] >= 2
            # the start needs an unused neighbour (or the new end) left to close the cycle
            if ok and depth + 1 < n and v not in nb[0] and not any(not visited[w] and w != v for w in nb[0]):
                ok = False
            if ok:
                visited[v] = True
                dfs(v, depth + 1, P[end, v] if Y is None else Y @ P[end, v])
                visited[v] = False
            for w in touched:
                avail[w] += 1

    dfs(0, 1, None)
    return total[0], count[0]


def _prep_matrices_sym('''
assert s.count("def _prep_matrices_sym(")==1
s=s.replace("def _prep_matrices_sym(", new, 1)
old="        t, p = _glynn_sum(mats, None if adj.sum() == n * (n - 1) else adj, n_cores, batch_mem)\n        total_trace += sign * t"
assert s.count(old)==1
s=s.replace(old,"""        if adj.sum() <= 4 * n and (adj == adj.T).all():
            t, p = _pruned_sum(mats, adj)  # few allowed steps: search the cycles, at any ring size
        else:
            t, p = _glynn_sum(mats, None if adj.sum() == n * (n - 1) else adj, n_cores, batch_mem)
        total_trace += sign * t""")
open(p,"w").write(s)
