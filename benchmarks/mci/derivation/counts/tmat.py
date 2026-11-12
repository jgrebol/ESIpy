"""Explicit transfer matrix for directed Hamiltonian cycles of C_n(S), n >= 3s (s = max S):
   H(n) = w^T T^(n-3s) v0.  Gives the linear recurrence (char/min poly) and growth constant."""
import numpy as np
from tm import _edge, _canon

def _step(n, S, active, states, v, last):
    cand = [i for i, u in enumerate(active) if _edge(n, S, u, v)]
    iv = len(active); new = {}; done = 0
    for st0, val in states.items():
        stack = [(list(st0) + [-1], 0, 0)]
        while stack:
            cur, ci, ne = stack.pop()
            key = tuple(cur); new[key] = new.get(key, 0) + val
            if ne == 2: continue
            for cj in range(ci, len(cand)):
                a = cand[cj]; xa, xb = cur[a], cur[iv]
                if xa == 0 or xb == 0: continue
                nxt = list(cur)
                if xa >= 1 and xa == xb:
                    if v != n - 1: continue
                    nxt[a] = 0; nxt[iv] = 0
                    if all(x == 0 for x in nxt): done += val
                    continue
                if xa == -1 and xb == -1:
                    lab = max(cur + [0]) + 1; nxt[a] = lab; nxt[iv] = lab
                elif xa == -1: nxt[a] = xb; nxt[iv] = 0
                elif xb == -1: nxt[iv] = xa; nxt[a] = 0
                else:
                    nxt = [xa if x == xb else x for x in nxt]; nxt[a] = 0; nxt[iv] = 0
                stack.append((nxt, cj + 1, ne + 1))
    active_new = active + [v]
    keep = [i for i, u in enumerate(active_new) if last[u] > v]
    out = {}
    for st, val in new.items():
        if any(st[i] != 0 for i in range(len(st)) if i not in keep): continue
        key = _canon(tuple(st[i] for i in keep)); out[key] = out.get(key, 0) + val
    return [active_new[i] for i in keep], out, done

def build(S):
    S = set(S); s = max(S); n = 4 * s + 2   # any n >= 3s gives the same opening/middle/closing
    nbrs = [[w for w in range(n) if w != u and _edge(n, S, u, w)] for u in range(n)]
    last = [max(nb) for nb in nbrs]
    active, states = [], {(): 1}
    for v in range(2 * s):
        active, states, _ = _step(n, S, active, states, v, last)
    act_mid = active
    # middle transition at v = 2s (pattern identical for all middle v)
    keys = list(states); idx = {k: i for i, k in enumerate(keys)}
    trans = []  # (from, to, coeff)
    q = list(keys); seen = set(keys)
    while q:
        k = q.pop()
        a2, out, _ = _step(n, S, act_mid, {k: 1}, 2 * s, last)
        for k2, c in out.items():
            if k2 not in idx: idx[k2] = len(keys); keys.append(k2)
            if k2 not in seen: seen.add(k2); q.append(k2)
            trans.append((idx[k], idx[k2], c))
    v0 = np.zeros(len(keys), dtype=object)
    for k, c in states.items(): v0[idx[k]] = c
    # closing functional: start window at v = n - s with relabelled actives
    nn = 3 * s + 0  # use n' = 3s+? : choose n' with same structure: early 0..s-1, window n'-2s..n'-s-1
    nn = 4 * s
    last2 = [max(w for w in range(nn) if w != u and _edge(nn, S, u, w)) for u in range(nn)]
    act_close = list(range(s)) + list(range(nn - 2 * s, nn - s))
    assert len(act_close) == len(act_mid)
    w = np.zeros(len(keys), dtype=object)
    for k, i in idx.items():
        a, st = act_close, {k: 1}; tot = 0
        for v in range(nn - s, nn):
            a, st, dn = _step(nn, S, a, st, v, last2); tot += dn
        w[i] = 2 * tot
    return keys, trans, v0, w

def sequence_mod(S, nmax, p):
    keys, trans, v0, w = build(S)
    s = max(S); D = len(keys)
    fr = np.array([t[0] for t in trans]); to = np.array([t[1] for t in trans]); cc = np.array([t[2] for t in trans], dtype=np.int64)
    vec = np.array([int(x) % p for x in v0], dtype=np.int64); ww = np.array([int(x) % p for x in w], dtype=np.int64)
    out = {}
    for n in range(3 * s, nmax + 1):
        out[n] = int((vec * ww % p).sum() % p)
        nv = np.zeros(D, dtype=np.int64); np.add.at(nv, to, cc * vec[fr] % p); vec = nv % p
    return out, D

def sequence_exact(S, nmax):
    keys, trans, v0, w = build(S)
    s = max(S); vec = list(v0); out = {}
    for n in range(3 * s, nmax + 1):
        out[n] = sum(int(a) * int(b) for a, b in zip(vec, w))
        nv = [0] * len(keys)
        for f, t, c in trans: nv[t] += c * vec[f]
        vec = nv
    return out, len(keys)

def bm_mod(a, p):
    C = [1]; B = [1]; L = 0; m = 1; b = 1
    for n in range(len(a)):
        d = a[n]
        for i in range(1, L + 1): d = (d + C[i] * a[n - i]) % p
        if d == 0: m += 1; continue
        T = C[:]; coef = d * pow(b, p - 2, p) % p
        C = C + [0] * (len(B) + m - len(C))
        for i in range(len(B)): C[i + m] = (C[i + m] - coef * B[i]) % p
        if 2 * L <= n: L = n + 1 - L; B = T; b = d; m = 1
        else: m += 1
    return [c if c <= p // 2 else c - p for c in C[:L + 1]], L
