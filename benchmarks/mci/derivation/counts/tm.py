"""Frontier (transfer-matrix) DP on circulant graphs C_n(S).
ham_count(n,S): # DIRECTED Hamiltonian cycles (=2x undirected), exact, any n with n > 2*max(S).
forest_poly(n,S): dict (j,p)->#linear forests (j edges, p nontrivial paths) of C_n(S).
The state space in the middle of the sweep does not depend on n => count(n) = u^T T^(n-c) w,
hence a linear recurrence in n (Cayley-Hamilton)."""
from math import factorial
from fractions import Fraction

def _edge(n, S, u, v):
    k = (v - u) % n
    return k in S or (n - k) in S

def _canon(st):
    m = {}; out = []
    for x in st:
        if x >= 1:
            if x not in m: m[x] = len(m) + 1
            out.append(m[x])
        else: out.append(x)
    return tuple(out)

def _sweep(n, S, mode):
    """mode 'ham' or 'forest'. state: tuple over active vertices: -1 deg0, 0 deg2, k>=1 deg1 with path label k.
    value: int (ham) or dict (j,p)->count (forest)."""
    S = set(S); s = max(S)
    assert n > 2 * s
    nbrs = [[w for w in range(n) if w != u and _edge(n, S, u, w)] for u in range(n)]
    last = [max(nb) for nb in nbrs]
    active = []
    states = {(): 1 if mode == 'ham' else {(0, 0): 1}}
    done = 0
    for v in range(n):
        active_new = active + [v]
        cand = [i for i, u in enumerate(active) if _edge(n, S, u, v)]
        iv = len(active)
        new = {}
        for st, val in states.items():
            st = list(st) + [-1]
            # choose up to 2 edges from cand, applied sequentially
            stack = [(st, 0, 0, 0, 0)]  # state, next cand idx, edges added, dj, dp
            while stack:
                cur, ci, ne, dj, dp = stack.pop()
                # record current
                key = _canon(cur)
                if mode == 'ham':
                    new[key] = new.get(key, 0) + val
                else:
                    dd = new.setdefault(key, {})
                    for (j, p), c in val.items():
                        kk = (j + dj, p + dp); dd[kk] = dd.get(kk, 0) + c
                if ne == 2: continue
                for cj in range(ci, len(cand)):
                    a = cand[cj]
                    xa, xb = cur[a], cur[iv]
                    if xa == 0 or xb == 0: continue
                    nxt = list(cur)
                    if xa >= 1 and xb >= 1 and xa == xb:
                        # closing a cycle
                        if mode != 'ham' or v != n - 1: continue
                        nxt[a] = 0; nxt[iv] = 0
                        if all(x == 0 for x in nxt):
                            # Hamiltonian cycle complete; must be no further edges => count once
                            done += val
                        continue
                    if xa == -1 and xb == -1:
                        lab = max([x for x in cur] + [0]) + 1
                        nxt[a] = lab; nxt[iv] = lab; ddp = 1
                    elif xa == -1:
                        nxt[a] = xb; nxt[iv] = 0 if False else xb; ddp = 0
                        # xb is deg1 endpoint, becomes deg2; a becomes endpoint with label xb
                        nxt[iv] = 0
                    elif xb == -1:
                        nxt[iv] = xa; nxt[a] = 0; ddp = 0
                    else:
                        nxt = [xa if x == xb else x for x in nxt]
                        nxt[a] = 0; nxt[iv] = 0; ddp = -1
                    stack.append((nxt, cj + 1, ne + 1, dj + 1, dp + ddp))
        # drop departing vertices
        keep = [i for i, u in enumerate(active_new) if last[u] > v]
        states = {}
        for st, val in new.items():
            ok = True
            for i in range(len(st)):
                if i not in keep and mode == 'ham' and st[i] != 0: ok = False; break
            if not ok: continue
            key = _canon(tuple(st[i] for i in keep))
            if mode == 'ham':
                states[key] = states.get(key, 0) + val
            else:
                dd = states.setdefault(key, {})
                for k2, c in val.items(): dd[k2] = dd.get(k2, 0) + c
        active = [active_new[i] for i in keep]
    if mode == 'ham':
        return 2 * done  # each undirected cycle found once (edge set) -> 2 directions
    return states[()]

def ham_count(n, S):
    return _sweep(n, S, 'ham')

def forest_poly(n, S):
    return _sweep(n, S, 'forest')

def ham_complement(n, F):
    """# directed Ham cycles of K_n minus circulant C_n(F), by inclusion-exclusion over linear forests of C_n(F)."""
    P = forest_poly(n, F)
    tot = 0
    for (j, p), c in P.items():
        k = n - j
        tot += (-1) ** j * c * factorial(k - 1) * 2 ** p
    # edge subsets of C_n(F) that are themselves Hamiltonian cycles (j = n)
    tot += (-1) ** n * ham_count(n, F)
    return tot
